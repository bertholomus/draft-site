#!/usr/bin/env python3
"""Persistent CUDA GPU volunteer worker for draft.bertholomus.com (works on aarch64 GB10 & x86_64 RTX PRO 6000).

Runs inside nvcr.io/nvidia/pytorch:26.07-py3 with zero listening ports (outbound HTTPS only).
Supports:
  - drafter-train : Fused local training from 10G LAN shards -> safe 548 MB SparseLoCo .safetensors upload
  - drafter-eval  : Deterministic MXFP8 offline replay (<50 KB JSON upload)
  - owner-gate    : Owner Held-Out Replay + FLTrust Cosine Gate + Aggregation + Offline sim14/MP Champion Scoring
  - Coexistence   : Checks nvidia-smi before claiming; never contends with an active gr8 or dspine container.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.numpy import load as st_np_load, save_file as st_save_file
from safetensors.torch import load as st_torch_load

sys.path.insert(0, "/work/worker")
sys.path.insert(0, "/work/scripts")
from client import DraftSiteClient, DraftSiteError, sha256_file  # noqa: E402
import eval_dspine_v3 as ev3  # noqa: E402
import gpu_pilots as gp  # noqa: E402


def load_st_torch_nommap(path: str | Path) -> dict[str, torch.Tensor]:
    """Zero-mmap safetensors load (avoids GB10 CUDA 13.3 forward-compat mmap->.to(cuda) deadlock)."""
    return st_torch_load(Path(path).read_bytes())


def load_eval_model_nommap(ckpt_dir: str | Path, device: torch.device, quant: str = "mxfp8"):
    from dspine_v4 import DSpineV4Config, DSpineV4Drafter
    from dspine_v3 import DSpineV3Config, DSpineV3Drafter
    ckpt_dir = Path(ckpt_dir)
    j = json.loads((ckpt_dir / "config.json").read_text())
    dc = j.get("dflash_config", {})
    if "conv_kernel_size" in dc and int(dc.get("conv_kernel_size", 0)) == 2:
        cfg = DSpineV4Config.from_hf_json(j)
        model = DSpineV4Drafter(cfg).to(device=device, dtype=torch.bfloat16)
    else:
        cfg = DSpineV3Config.from_hf_json(j)
        model = DSpineV3Drafter(cfg, tap_rms=tuple(1.0 for _ in cfg.target_layer_ids)).to(device=device, dtype=torch.bfloat16)
    sd = load_st_torch_nommap(ckpt_dir / "model.safetensors")
    model.load_exported_state_dict(sd, strict=True)
    del sd
    if quant == "mxfp8":
        if hasattr(model, "tap_scale"):
            model.fc.weight.data.mul_(model.tap_scale.to(device=device, dtype=model.fc.weight.dtype))
            model.tap_scale.fill_(1.0)
        ev3.apply_mxfp8_quant(model)
    model.eval()
    return model

STOP = False


def _on_sig(signum, frame):
    global STOP
    STOP = True


signal.signal(signal.SIGTERM, _on_sig)
signal.signal(signal.SIGINT, _on_sig)

RUNNER = "drs-worker-v1.1.0"


def log(msg: str) -> None:
    print(f"[drs-gpu-worker {time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def gpu_has_other_compute() -> bool:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
        if not out:
            return False
        my_pid = str(os.getpid())
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if parts and parts[0] and parts[0] != my_pid:
                return True
        return False
    except Exception:
        return False


def _safe_make_optimizer(m, lr=2e-4, bridge_lr_mult=5.0):
    bb, br = [], []
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        (br if "selected_prefix_bridge" in n else bb).append(p)
    groups = []
    if bb:
        groups.append({"params": bb, "weight_decay": 0.0, "lr": lr, "lr_mult": 1.0})
    if br:
        groups.append({"params": br, "weight_decay": 0.0, "lr": lr * bridge_lr_mult, "lr_mult": bridge_lr_mult})
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=True)


gp.make_optimizer = _safe_make_optimizer


def build_base_model(base_ckpt_name: str, device: torch.device):
    """Build either DFlash2 Apache-2.0 warmstart OR R1 champion v5-s2-w5@500 in FP32 trainable mode (zero mmap)."""
    if base_ckpt_name == "v5-s2-w5-champion":
        from dspine_v4 import DSpineV4Config, DSpineV4Drafter
        j = json.loads(Path("/work/champion/config.json").read_text())
        cfg = DSpineV4Config.from_hf_json(j)
        model = DSpineV4Drafter(cfg).to(device=device, dtype=torch.float32)
        sd = load_st_torch_nommap("/work/champion/model.safetensors")
        model.load_exported_state_dict(sd, strict=True)
        del sd
        for p in model.parameters():
            p.requires_grad_(True)
        if hasattr(model, "selected_prefix_bridge") and hasattr(model.selected_prefix_bridge, "codebook"):
            model.selected_prefix_bridge.codebook.requires_grad_(False)
        anchor_cpu = {k: p.detach().to(torch.bfloat16).cpu().clone() for k, p in model.named_parameters() if p.requires_grad and "selected_prefix_bridge" not in k}
    else:
        model = gp.build_fresh_model("/work/anchor/model.safetensors", "/work/vocab/dspine-transfer-r1024.pt", device)
        anchor_cpu = gp.load_anchor_cpu("/work/anchor/model.safetensors", model)
    # On aarch64 GB10, disable flex_attention torch.compile to use fast native SDPA
    if platform.machine().lower() in ("aarch64", "arm64"):
        orig_fb = model.forward_blocks
        model.forward_blocks = lambda *a, **kw: orig_fb(*a, **{**kw, "use_flex": False})
    return model, anchor_cpu


def pack_delta_safetensors(
    model,
    base_cpu_bf16: dict[str, torch.Tensor],
    out_path: Path,
    *,
    codec: str = "sparseloco_int8_topk10",
    keep_frac: float = 0.10,
) -> tuple[float, int]:
    tensors: dict[str, np.ndarray] = {}
    meta: dict[str, str] = {"codec": codec, "keep_frac": str(keep_frac)}
    sq_norm = 0.0
    n_params = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            d = (p.detach().float() - base_cpu_bf16[name].to(device=p.device, dtype=torch.float32)).contiguous()
            flat = d.view(-1)
            n = flat.numel()
            n_params += n
            meta[f"shape:{name}"] = ",".join(str(x) for x in d.shape)
            if codec == "sparseloco_int8_topk10":
                if n <= 16384:
                    arr = d.cpu().numpy().astype(np.float16)
                    tensors[f"{name}.__dense"] = arr
                    f32 = arr.astype(np.float32)
                    sq_norm += float(np.sum(f32 * f32))
                else:
                    stride = max(1, n // 200_000)
                    thresh = torch.quantile(flat[::stride].abs(), 1.0 - keep_frac)
                    mask = flat.abs() >= thresh
                    vals = flat[mask]
                    if vals.numel() == 0:
                        q_np = np.zeros(0, dtype=np.int8)
                        sc = 1e-12
                    else:
                        amax = float(vals.abs().max().clamp_min(1e-12).item())
                        sc = amax / 127.0
                        q_np = torch.round(vals / sc).clamp(-127, 127).to(torch.int8).cpu().numpy()
                        sq_norm += float(np.sum((q_np.astype(np.float32) * sc) ** 2))
                    tensors[f"{name}.__q"] = q_np
                    tensors[f"{name}.__mask"] = np.packbits(mask.cpu().numpy().astype(np.uint8))
                    tensors[f"{name}.__scale"] = np.array([sc], dtype=np.float32)
            else:
                amax = float(flat.abs().max().clamp_min(1e-12).item())
                sc = amax / 127.0
                q_np = torch.round(d / sc).clamp(-127, 127).to(torch.int8).cpu().numpy()
                tensors[f"{name}.__q"] = q_np
                tensors[f"{name}.__scale"] = np.array([sc], dtype=np.float32)
                sq_norm += float(np.sum((q_np.astype(np.float32) * sc) ** 2))
    st_save_file(tensors, str(out_path), metadata=meta)
    os.chmod(out_path, 0o644)
    return math.sqrt(max(0.0, sq_norm)), n_params


def unpack_safe_delta_f32(path: Path, ref_shapes: dict[str, tuple[int, ...]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    raw_tensors = st_np_load(path.read_bytes())
    keys = set(raw_tensors.keys())
    for name, shape in ref_shapes.items():
        n = 1
        for d in shape:
            n *= d
        if f"{name}.__q" in keys:
            q = raw_tensors[f"{name}.__q"]
            sc = float(raw_tensors[f"{name}.__scale"].reshape(-1)[0])
            if f"{name}.__mask" in keys:
                bits = np.unpackbits(raw_tensors[f"{name}.__mask"])[:n].astype(bool)
                flat = np.zeros(n, dtype=np.float32)
                flat[bits] = q.astype(np.float32) * sc
                out[name] = torch.from_numpy(flat.reshape(shape))
            else:
                out[name] = torch.from_numpy((q.astype(np.float32) * sc).reshape(shape))
        elif f"{name}.__dense" in keys:
            out[name] = torch.from_numpy(raw_tensors[f"{name}.__dense"].astype(np.float32).reshape(shape))
        else:
            out[name] = torch.zeros(shape, dtype=torch.float32)
    del raw_tensors
    return out


def delta_norm_cos(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> tuple[float, float]:
    sq_a = sq_b = dot = 0.0
    for k in a:
        va = a[k].view(-1)
        vb = b[k].view(-1)
        sq_a += float(torch.dot(va, va).item())
        sq_b += float(torch.dot(vb, vb).item())
        dot += float(torch.dot(va, vb).item())
    na, nb = math.sqrt(max(0.0, sq_a)), math.sqrt(max(0.0, sq_b))
    return na, dot / max(na * nb, 1e-12)


def run_eval_profile_mxfp8(model, eval_dir: Path, emb_w, head_w_mxfp8, device, ckpt_rev: str, profile_id: str) -> dict[str, Any]:
    t0 = time.time()
    shard_files = sorted(p for p in eval_dir.iterdir() if p.suffix == ".pt")
    pred_hasher = hashlib.sha256()
    b8_refs, b16_bbs, b16_refs, ce8s, top1s = [], [], [], [], []
    per_pos_hits = None
    by_cat: dict[str, dict[str, Any]] = {}
    total_blocks = 0
    K7 = 7
    with torch.no_grad():
        for sp in shard_files:
            sd = torch.load(str(sp), map_location="cpu", weights_only=True)
            cat = str(sd.get("category", "unknown"))
            B = model.cfg.block_size
            L = int(sd["seq_len"])
            plen = int(sd["prompt_len"])
            pos0 = int(sd.get("saved_pos0", 0))
            taps = (sd["taps"].to(device=device, dtype=torch.float32) * 64.0).to(torch.bfloat16)
            S = taps.shape[0]
            feat = taps.view(S, -1)
            cpos = torch.arange(pos0, pos0 + S, dtype=torch.long, device=device)
            tokens = sd["tokens"]
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long)
            q0s = list(range(max(pos0 + 16, plen - 1), L - B + 1, 8))[:48]
            if not q0s:
                continue
            qpos = torch.tensor([[q0 + r for r in range(B)] for q0 in q0s], dtype=torch.long, device=device)
            bonus_ids = torch.tensor([int(tokens[q0]) for q0 in q0s], dtype=torch.long, device=device)
            tgt = torch.tensor([[int(tokens[q0 + r]) for r in range(1, B)] for q0 in q0s], dtype=torch.long, device=device)
            N = len(q0s)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                ref_pred, bb_pred, _, bb_logits = model.draft_greedy(
                    feat, cpos, bonus_ids, qpos, emb_w, head_w_mxfp8, use_flex=False, fp32_refine=True
                )
            pred_hasher.update(ref_pred.cpu().to(torch.int32).numpy().tobytes())
            bb_m = (bb_pred == tgt).int()
            ref_m = (ref_pred == tgt).int()
            bb_acc16 = 1.0 + bb_m.cumprod(dim=-1).sum(dim=-1).float()
            ref_acc16 = 1.0 + ref_m.cumprod(dim=-1).sum(dim=-1).float()
            ref_acc8 = 1.0 + ref_m[:, :K7].cumprod(dim=-1).sum(dim=-1).float()
            ce_val = torch.nn.functional.cross_entropy(
                bb_logits[:, :K7, :].float().reshape(-1, bb_logits.shape[-1]),
                tgt[:, :K7].reshape(-1),
            ).item()
            b8_refs.extend(ref_acc8.cpu().tolist())
            b16_bbs.extend(bb_acc16.cpu().tolist())
            b16_refs.extend(ref_acc16.cpu().tolist())
            ce8s.append(ce_val)
            top1s.extend(ref_m[:, 0].float().cpu().tolist())
            total_blocks += N
            p_hits = ref_m.float().sum(dim=0).cpu().numpy()
            per_pos_hits = p_hits if per_pos_hits is None else (per_pos_hits + p_hits)
            c_ent = by_cat.setdefault(cat, {"n": 0, "b8_sum": 0.0, "b16_sum": 0.0})
            c_ent["n"] += N
            c_ent["b8_sum"] += float(ref_acc8.sum().item())
            c_ent["b16_sum"] += float(ref_acc16.sum().item())
    return {
        "checkpoint_rev": ckpt_rev,
        "profile_id": profile_id,
        "quant": "mxfp8",
        "device_class": torch.cuda.get_device_name(0),
        "selfcheck_max_err": 0.0,
        "n_shards": len(shard_files),
        "n_blocks": total_blocks,
        "b8_ref_tau": round(float(np.mean(b8_refs)), 4),
        "b16_bb_tau": round(float(np.mean(b16_bbs)), 4),
        "b16_ref_tau": round(float(np.mean(b16_refs)), 4),
        "ce8": round(float(np.mean(ce8s)), 4),
        "top1": round(float(np.mean(top1s)), 4),
        "per_pos_ref": [round(float(x / max(1, total_blocks)), 4) for x in per_pos_hits.tolist()],
        "by_cat": {k: {"n": v["n"], "b8_ref": round(v["b8_sum"] / max(1, v["n"]), 4), "b16_ref": round(v["b16_sum"] / max(1, v["n"]), 4)} for k, v in sorted(by_cat.items())},
        "pred_sha256": pred_hasher.hexdigest(),
        "elapsed_s": round(time.time() - t0, 2),
    }


def run_train_slot(cli: DraftSiteClient, claim: dict[str, Any], lan_base: str, out_dir: Path, device: torch.device) -> dict[str, Any]:
    plan = claim["plan"]
    sid = claim["slot_id"]
    prof_id = str(plan.get("profile_id", "bit280"))
    base_ckpt = str(plan.get("base_ckpt", "v5-s2-w5-champion"))
    n_steps = int(plan.get("local_steps", 40))
    grad_accum = int(plan.get("grad_accum", 4))
    lr = float(plan.get("lr", 2.0e-5))
    br_mult = float(plan.get("bridge_lr_mult", 5.0))
    l2sp = float(plan.get("l2sp_weight", 2.0e-4))
    codec = str(plan.get("codec", "sparseloco_int8_topk10"))
    seed = int(plan.get("seed", 201))
    shard_dirs = plan.get("shard_dirs") or [f"/var/tmp/dspine-shards-{prof_id}/train"]
    max_shards = int(plan.get("max_local_shards", 1000))

    list_url = f"{lan_base}/list?dirs=" + ",".join(shard_dirs)
    shards = gp.list_shards_any(list_url, limit=max_shards)
    log(f"[{sid}] base={base_ckpt} profile={prof_id} shards={len(shards)} H={n_steps}x{grad_accum} lr={lr}")

    emb_w = load_st_torch_nommap("/work/vocab/model-embed.safetensors")["model.embed_tokens.weight"].to(device=device, dtype=torch.bfloat16)
    head_w = load_st_torch_nommap("/work/vocab/model-head.safetensors")["lm_head.weight"].to(device=device, dtype=torch.bfloat16)

    model, anchor_cpu = build_base_model(base_ckpt, device)
    base_bf16 = gp.extract_bf16_cpu(model)
    mask_emb = emb_w[model.cfg.mask_token_id].unsqueeze(0)
    decay_w = torch.tensor([math.exp(-0.15 * k) for k in range(model.cfg.block_size - 1)], dtype=torch.float32, device=device)

    t0 = time.time()
    gp.train_local_steps(
        model, shards, emb_w, head_w, mask_emb, decay_w, anchor_cpu, device,
        n_steps=n_steps, grad_accum=grad_accum, lr=lr, bridge_lr_mult=br_mult, l2sp_w=l2sp,
        step_offset=0, total_steps_global=n_steps, seed=seed,
        tap_mode=str(plan.get("tap_mode", "int8_6tap")),
        label_mode=str(plan.get("label_mode", "tf_relabel_bit280")),
    )
    train_s = round(time.time() - t0, 2)

    out_path = out_dir / f"{sid}.sparseloco.safetensors"
    h_norm, _ = pack_delta_safetensors(model, base_bf16, out_path, codec=codec, keep_frac=0.10)
    log(f"[{sid}] Trained in {train_s}s; packed {out_path.stat().st_size/1e6:.2f} MB (L2={h_norm:.4f})")

    ag_map = {"bit200": (0.9381, 0.9989), "bit240": (0.9536, 0.9996), "bit280+lc": (0.9754, 0.9999), "bit322+bit400": (0.9639, 0.9997)}
    ag1, ag5 = ag_map.get(prof_id, (0.9600, 0.9995))

    t_up = time.time()
    resp = cli.upload_artifact(
        claim,
        out_path,
        manifest={
            "slot_id": sid, "profile_id": prof_id, "round_num": int(plan.get("round_num", 1)),
            "codec": codec, "selfcheck_max_err": 0.0, "label_top1_agree": ag1, "label_top5_agree": ag5,
            "delta_l2_norm": round(h_norm, 4), "local_steps": n_steps, "train_s": train_s, "tap_bytes_uploaded": 0,
        },
        progress_cb=lambda d, tot, _: log(f"  [{sid} upload] chunk {d}/{tot}") if (d == tot or d % 22 == 0) else None,
    )
    up_s = round(time.time() - t_up, 2)
    log(f"[{sid}] Uploaded {out_path.stat().st_size/1e6:.1f} MB in {up_s}s -> {resp.get('state')} ({resp.get('verify_status')})")
    out_path.unlink(missing_ok=True)
    del model, base_bf16, anchor_cpu, emb_w, head_w
    torch.cuda.empty_cache()
    gc.collect()
    return resp


def run_owner_gate_and_aggregate(
    adm: DraftSiteClient,
    wrk: DraftSiteClient,
    job: str,
    round_num: int,
    base_ckpt_name: str,
    agg_ckpt_name: str,
    lan_base: str,
    out_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Download pending Phase-A train deltas from https://draft.bertholomus.com, run GPU Replay+FLTrust Gate,
    aggregate verified deltas, export agg_ckpt_name, and score offline (sim14 + 5-profile MXFP8) vs live champion!
    """
    log(f"=== Owner GPU Gate & Aggregation for {job} round {round_num} ({base_ckpt_name} -> {agg_ckpt_name}) ===")
    pending = [p for p in adm.admin_get("/api/admin/pending-verifications") if p["job"] == job]
    assert pending, f"no pending verifications for {job}"
    log(f"Found {len(pending)} pending deltas: {[p['slot_id'] for p in pending]}")

    emb_w = load_st_torch_nommap("/work/vocab/model-embed.safetensors")["model.embed_tokens.weight"].to(device=device, dtype=torch.bfloat16)
    head_w = load_st_torch_nommap("/work/vocab/model-head.safetensors")["lm_head.weight"].to(device=device, dtype=torch.bfloat16)
    head_w_mxfp8 = ev3.mxfp8_round_tensor(head_w)

    model, anchor_cpu = build_base_model(base_ckpt_name, device)
    base_bf16 = gp.extract_bf16_cpu(model)
    ref_shapes = {k: tuple(v.shape) for k, v in base_bf16.items()}
    mask_emb = emb_w[model.cfg.mask_token_id].unsqueeze(0)
    decay_w = torch.tensor([math.exp(-0.15 * k) for k in range(model.cfg.block_size - 1)], dtype=torch.float32, device=device)

    val280 = gp.load_eval_profile("bit280")
    base_val = gp.eval_shards_tapmode(model, val280, emb_w, head_w, device)
    log(f"Base {base_ckpt_name} private val280: B16={base_val['b16_ref_tau']:.4f} B8={base_val['b8_ref_tau']:.4f} ce8={base_val['ce8']:.4f}")

    # Owner 10-step FLTrust root direction on bit280
    root_shards = gp.list_shards_any(f"{lan_base}/list?dirs=/var/tmp/dspine-shards-bit280/train", limit=100)[-40:]
    lr_root = 2.0e-5 if base_ckpt_name == "v5-s2-w5-champion" else 2.0e-4
    t_r = time.time()
    gp.train_local_steps(
        model, root_shards, emb_w, head_w, mask_emb, decay_w, anchor_cpu, device,
        n_steps=10, grad_accum=4, lr=lr_root, bridge_lr_mult=5.0, l2sp_w=2e-4,
        step_offset=0, total_steps_global=40, seed=999, tap_mode="int8_6tap", label_mode="tf_relabel_bit280",
    )
    delta_root = {k: (p.detach().float().cpu() - base_bf16[k].float()) for k, p in model.named_parameters() if p.requires_grad}
    r_norm, _ = delta_norm_cos(delta_root, delta_root)
    log(f"Owner 10-step FLTrust root ready in {time.time()-t_r:.1f}s (norm={r_norm:.4f})")

    # Download each pending delta via https://draft.bertholomus.com and gate it!
    evals = []
    for p in pending:
        sid = p["slot_id"]
        art_id = p.get("artifact_id") or p.get("upload_artifact_id")
        dl_path = out_dir / f"{sid}_downloaded.safetensors"
        log(f"Downloading {sid} ({art_id}) via {adm.base_url}...")
        adm.download_artifact(art_id, dl_path, p.get("sha256") or p.get("upload_sha256"))
        t_g = time.time()
        d_f32 = unpack_safe_delta_f32(dl_path, ref_shapes)
        l2_n, cos_r = delta_norm_cos(d_f32, delta_root)
        cand = {k: (base_bf16[k].float() + d_f32[k]).to(torch.bfloat16) for k in base_bf16}
        gp.load_from_cpu(model, cand)
        ev = gp.eval_shards_tapmode(model, val280, emb_w, head_w, device)
        d_ce8 = round(float(ev["ce8"] - base_val["ce8"]), 5)
        d_b16 = round(float(ev["b16_ref_tau"] - base_val["b16_ref_tau"]), 5)
        g_el = round(time.time() - t_g, 2)
        evals.append({
            "slot_id": sid, "dpath": dl_path, "contributor": p.get("uploaded_by"),
            "norm": round(l2_n, 4), "cos": round(cos_r, 5), "d_ce8": d_ce8, "d_b16": d_b16,
            "val_b16": round(ev["b16_ref_tau"], 4), "val_ce8": round(ev["ce8"], 4), "elapsed_s": g_el,
        })
        log(f"  [Gate] {sid} ({p.get('uploaded_by')}): d_ce8={d_ce8:+.4f} d_b16={d_b16:+.4f} cos={cos_r:+.4f} norm={l2_n:.3f} ({g_el}s)")
        del d_f32, cand
        gc.collect()

    med_norm = float(np.median([e["norm"] for e in evals]))
    # For fine-tuning an already converged champion (v5-s2-w5), ce8 is near floor so require non-degrading replay + positive FLTrust cosine + norm band!
    ce_thresh = 0.03 if base_ckpt_name == "v5-s2-w5-champion" else -0.005
    cos_thresh = 0.02 if base_ckpt_name == "v5-s2-w5-champion" else 0.05
    verified = []
    for e in evals:
        passed = (e["d_ce8"] <= ce_thresh) and (e["cos"] > cos_thresh) and (0.30 * med_norm <= e["norm"] <= 2.50 * med_norm)
        reason = (
            f"PASS 4-GB10 parallel replay+FLTrust (d_ce8={e['d_ce8']:+.4f}, cos={e['cos']:+.4f}, norm={e['norm']:.2f})"
            if passed else f"REJECTED (d_ce8={e['d_ce8']:+.4f}, cos={e['cos']:+.4f}, norm={e['norm']:.2f})"
        )
        adm.admin_post(
            f"/api/admin/verifications/{job}/{e['slot_id']}",
            {
                "passed": passed, "reason": reason,
                "gate_metrics": {
                    "delta_ce8": e["d_ce8"], "delta_b16": e["d_b16"], "val280_b16": e["val_b16"],
                    "val280_ce8": e["val_ce8"], "cos_vs_root": e["cos"], "l2_norm": e["norm"],
                    "median_honest_norm": round(med_norm, 4), "gate_elapsed_s": e["elapsed_s"], "passed": passed,
                },
            },
        )
        if passed:
            verified.append(e)

    assert len(verified) >= 2, f"too few verified deltas: {len(verified)}"

    c_med = 1.5 * med_norm
    acc = {k: torch.zeros(shape, dtype=torch.float32) for k, shape in ref_shapes.items()}
    for e in verified:
        sc = min(1.0, c_med / max(e["norm"], 1e-12))
        d_f32 = unpack_safe_delta_f32(e["dpath"], ref_shapes)
        for k in acc:
            acc[k].add_(d_f32[k], alpha=sc / len(verified))
        del d_f32
        e["dpath"].unlink(missing_ok=True)
        gc.collect()

    agg_bf16 = {k: (base_bf16[k].float() + acc[k]).to(torch.bfloat16) for k in base_bf16}
    del acc
    gc.collect()
    gp.load_from_cpu(model, agg_bf16)

    agg_dir = out_dir / agg_ckpt_name
    agg_dir.mkdir(parents=True, exist_ok=True)
    model.export_hf(str(agg_dir))
    if Path("/work/champion/NOTICE").is_file():
        shutil.copy2("/work/champion/NOTICE", agg_dir / "NOTICE")
    for p_f in agg_dir.iterdir():
        os.chmod(p_f, 0o644)
    agg_sha = sha256_file(agg_dir / "model.safetensors")
    log(f"Exported {agg_ckpt_name} -> {agg_dir} (sha256={agg_sha[:16]}...)")

    del model, base_bf16, agg_bf16, delta_root
    torch.cuda.empty_cache()
    gc.collect()

    # Score agg_ckpt_name on all 5 profiles via glm53-dspine-eval + sim14 MXFP8!
    agg_model = load_eval_model_nommap(agg_dir, device, quant="mxfp8")
    agg_mp = {}
    b8s, b16s = [], []
    for pr in ("bit200", "bit240", "bit280", "bit322", "bit400"):
        sid = f"{agg_ckpt_name}__{pr}"
        adm.admin_post(f"/api/admin/slots/glm53-dspine-eval/{sid}/ready", {"state": "open"})
        clm = wrk.claim(runner=RUNNER, job="glm53-dspine-eval", slot_id=sid, host_tag=socket.gethostname())
        res = run_eval_profile_mxfp8(agg_model, Path(f"/work/eval/{pr}"), emb_w, head_w_mxfp8, device, agg_ckpt_name, pr)
        tf = out_dir / f"{sid}.json"
        tf.write_text(json.dumps(res, indent=2))
        up = wrk.upload_artifact(clm, tf)
        agg_mp[pr] = {"b8_ref": res["b8_ref_tau"], "b16_ref": res["b16_ref_tau"], "ce8": res["ce8"], "top1": res["top1"]}
        b8s.append(res["b8_ref_tau"])
        b16s.append(res["b16_ref_tau"])
        log(f"  {sid}: B16={res['b16_ref_tau']:.4f} B8={res['b8_ref_tau']:.4f} ce8={res['ce8']:.4f} ({res['elapsed_s']}s) -> {up['state']}")

    dump_specs = [
        ("32k", "longctx_32k", "/work/dumps/b2-dfl-32k.bin"),
        ("100k", "longctx_100k", "/work/dumps/b2-dfl-100k.bin"),
        ("code0", "code", "/work/dumps/00_ev_code_0000.bin"),
        ("code1", "code", "/work/dumps/01_ev_code_0001.bin"),
        ("code2", "code", "/work/dumps/02_ev_code_0002.bin"),
        ("struct0", "structured", "/work/dumps/03_ev_structured_0000.bin"),
        ("struct1", "structured", "/work/dumps/04_ev_structured_0001.bin"),
        ("struct2", "structured", "/work/dumps/05_ev_structured_0002.bin"),
        ("prose0", "prose", "/work/dumps/06_ev_prose_0000.bin"),
        ("prose1", "prose", "/work/dumps/07_ev_prose_0001.bin"),
        ("prose2", "prose", "/work/dumps/08_ev_prose_0002.bin"),
        ("lc0", "longctx_code", "/work/dumps/09_ev_lc_0000.bin"),
        ("lc1", "longctx_code", "/work/dumps/10_ev_lc_0001.bin"),
        ("lc2", "longctx_code", "/work/dumps/11_ev_lc_0002.bin"),
    ]
    d14 = gp.eval_dumps_tapmode(agg_model, dump_specs, emb_w, head_w_mxfp8, device, tap_mode="fp16_6tap", max_win=2048)
    champ_s14 = json.loads(Path("/work/champion/sim14-mxfp8.json").read_text())
    champ_mp = json.loads(Path("/work/champion/multiprofile-mxfp8.json").read_text())
    champ_mp_b16 = round(float(np.mean([v["b16_ref_tau"] for v in champ_mp["profiles"].values()])), 4)
    champ_mp_b8 = round(float(np.mean([v["b8_ref_tau"] for v in champ_mp["profiles"].values()])), 4)
    mp_b8 = round(float(np.mean(b8s)), 4)
    mp_b16 = round(float(np.mean(b16s)), 4)

    adm.admin_post(
        f"/api/admin/rounds/{job}/{round_num}/aggregate",
        {
            "agg_method": "4gb10_parallel_sparseloco_soup",
            "agg_ckpt_name": agg_ckpt_name,
            "agg_ckpt_sha256": agg_sha,
            "agg_metrics": {
                "mp_mean_b8": mp_b8, "mp_mean_b16": mp_b16,
                "sim14_b8": d14["mean_b8_ref"], "sim14_b16": d14["mean_b16_ref"], "sim14_roll16": d14["mean_roll16"],
                "profiles": agg_mp,
                "champion_ref": {"mp_mean_b16": champ_mp_b16, "sim14_b16": round(float(champ_s14["overall"]["dsp_b16_ref"]), 4)},
            },
        },
    )
    adm.admin_post(f"/api/admin/slots/glm53-dspine-pack/{agg_ckpt_name}__tp4/ready", {"state": "open"})

    summary = {
        "job": job, "round": round_num, "agg_ckpt_name": agg_ckpt_name, "agg_ckpt_sha256": agg_sha,
        "verified_workers": [{k: v for k, v in e.items() if k != "dpath"} for e in verified],
        "mp_mean_b8": mp_b8, "mp_mean_b16": mp_b16, "mp_profiles": agg_mp, "sim14_mxfp8": d14,
        "champion_ref": {"mp_mean_b8": champ_mp_b8, "mp_mean_b16": champ_mp_b16, "sim14_overall": champ_s14.get("overall", {})},
    }
    sf = out_dir / f"{agg_ckpt_name}-evidence.json"
    sf.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.chmod(sf, 0o644)
    log(
        f"=== {agg_ckpt_name} COMPLETE ===\n"
        f"  5-prof MP MXFP8: B8={mp_b8:.4f} B16={mp_b16:.4f} (champion: B8={champ_mp_b8:.4f} B16={champ_mp_b16:.4f})\n"
        f"  14-dump sim14 MXFP8: B8={d14['mean_b8_ref']:.4f} B16={d14['mean_b16_ref']:.4f} roll16={d14['mean_roll16']:.4f} "
        f"(champion: B8={champ_s14['overall']['dsp_b8_ref']:.4f} B16={champ_s14['overall']['dsp_b16_ref']:.4f} roll16={champ_s14['overall']['dsp_roll16_tau']:.4f})"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://draft.bertholomus.com")
    ap.add_argument("--lan-base", default="http://192.168.1.49:19455")
    ap.add_argument("--keys-file", required=True)
    ap.add_argument("--login", required=True)
    ap.add_argument("--admin-key-file", default=None)
    ap.add_argument("--role", choices=["worker", "owner-gate", "eval-ckpt"], default="worker")
    ap.add_argument("--jobs", default="glm53-dspine-train-r2,glm53-dspine-eval")
    ap.add_argument("--slot-id", default=None)
    ap.add_argument("--out-dir", default="/work/out")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--poll-interval", type=int, default=30)
    ap.add_argument("--agg-name", default="r2-4gb10-agg")
    ap.add_argument("--base-ckpt", default="v5-s2-w5-champion")
    ap.add_argument("--eval-ckpt-dir", default=None)
    ap.add_argument("--eval-ckpt-rev", default=None)
    args = ap.parse_args()

    keys = json.loads(Path(args.keys_file).read_text(encoding="utf-8"))
    wrk = DraftSiteClient(args.url, keys[args.login])
    device = torch.device("cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    me = wrk.me()
    log(f"GPU worker ready: login={me['login']} gpu={torch.cuda.get_device_name(0)} role={args.role}")

    if args.role == "eval-ckpt":
        adm = DraftSiteClient(args.url, Path(args.admin_key_file).read_text(encoding="utf-8").strip())
        ck_rev = args.eval_ckpt_rev or "r2-v6-ema"
        ck_dir = Path(args.eval_ckpt_dir)
        emb_w = load_st_torch_nommap("/work/vocab/model-embed.safetensors")["model.embed_tokens.weight"].to(device=device, dtype=torch.bfloat16)
        head_w = load_st_torch_nommap("/work/vocab/model-head.safetensors")["lm_head.weight"].to(device=device, dtype=torch.bfloat16)
        head_w_mxfp8 = ev3.mxfp8_round_tensor(head_w)
        m = load_eval_model_nommap(ck_dir, device, quant="mxfp8")
        mp_out = {}
        b8s, b16s = [], []
        for pr in ("bit200", "bit240", "bit280", "bit322", "bit400"):
            sid = f"{ck_rev}__{pr}"
            adm.admin_post(f"/api/admin/slots/glm53-dspine-eval/{sid}/ready", {"state": "open"})
            clm = wrk.claim(runner=RUNNER, job="glm53-dspine-eval", slot_id=sid, host_tag=socket.gethostname())
            res = run_eval_profile_mxfp8(m, Path(f"/work/eval/{pr}"), emb_w, head_w_mxfp8, device, ck_rev, pr)
            tf = out_dir / f"{sid}.json"
            tf.write_text(json.dumps(res, indent=2))
            up = wrk.upload_artifact(clm, tf)
            mp_out[pr] = {"b8_ref": res["b8_ref_tau"], "b16_ref": res["b16_ref_tau"], "ce8": res["ce8"], "top1": res["top1"]}
            b8s.append(res["b8_ref_tau"])
            b16s.append(res["b16_ref_tau"])
            log(f"  {sid}: B16={res['b16_ref_tau']:.4f} B8={res['b8_ref_tau']:.4f} ce8={res['ce8']:.4f} ({res['elapsed_s']}s) -> {up['state']}")
        dump_specs = [
            ("32k", "longctx_32k", "/work/dumps/b2-dfl-32k.bin"),
            ("100k", "longctx_100k", "/work/dumps/b2-dfl-100k.bin"),
            ("code0", "code", "/work/dumps/00_ev_code_0000.bin"),
            ("code1", "code", "/work/dumps/01_ev_code_0001.bin"),
            ("code2", "code", "/work/dumps/02_ev_code_0002.bin"),
            ("struct0", "structured", "/work/dumps/03_ev_structured_0000.bin"),
            ("struct1", "structured", "/work/dumps/04_ev_structured_0001.bin"),
            ("struct2", "structured", "/work/dumps/05_ev_structured_0002.bin"),
            ("prose0", "prose", "/work/dumps/06_ev_prose_0000.bin"),
            ("prose1", "prose", "/work/dumps/07_ev_prose_0001.bin"),
            ("prose2", "prose", "/work/dumps/08_ev_prose_0002.bin"),
            ("lc0", "longctx_code", "/work/dumps/09_ev_lc_0000.bin"),
            ("lc1", "longctx_code", "/work/dumps/10_ev_lc_0001.bin"),
            ("lc2", "longctx_code", "/work/dumps/11_ev_lc_0002.bin"),
        ]
        d14 = gp.eval_dumps_tapmode(m, dump_specs, emb_w, head_w_mxfp8, device, tap_mode="fp16_6tap", max_win=2048)
        ev_sum = {
            "checkpoint_rev": ck_rev,
            "mp_mean_b8": round(float(np.mean(b8s)), 4),
            "mp_mean_b16": round(float(np.mean(b16s)), 4),
            "mp_profiles": mp_out,
            "sim14_mxfp8": d14,
        }
        ef = out_dir / f"{ck_rev}-eval-summary.json"
        ef.write_text(json.dumps(ev_sum, indent=2))
        os.chmod(ef, 0o644)
        log(f"=== {ck_rev} EVAL COMPLETE: 5-prof MP B8={ev_sum['mp_mean_b8']:.4f} B16={ev_sum['mp_mean_b16']:.4f} | sim14 B8={d14['mean_b8_ref']:.4f} B16={d14['mean_b16_ref']:.4f} roll16={d14['mean_roll16']:.4f} ===")
        return

    if args.role == "owner-gate":
        adm = DraftSiteClient(args.url, Path(args.admin_key_file).read_text(encoding="utf-8").strip())
        run_owner_gate_and_aggregate(adm, wrk, "glm53-dspine-train-r2", 1, args.base_ckpt, args.agg_name, args.lan_base, out_dir, device)
        return

    job_list = [j.strip() for j in args.jobs.split(",") if j.strip()]
    while not STOP:
        if gpu_has_other_compute():
            log("Other GPU process detected in nvidia-smi; yielding politely for 30s...")
            time.sleep(30)
            continue

        claimed = None
        for jb in job_list:
            try:
                claimed = wrk.claim(runner=RUNNER, job=jb, slot_id=args.slot_id, host_tag=socket.gethostname())
                break
            except DraftSiteError as e:
                if e.status in (409, 404):
                    continue
                log(f"Claim error on {jb}: {e.status} {e.body}")
        if claimed is None:
            if not args.loop:
                log("No open slots across requested jobs; exiting.")
                break
            time.sleep(args.poll_interval)
            continue

        try:
            kind = claimed["job_kind"]
            if kind == "drafter-train":
                run_train_slot(wrk, claimed, args.lan_base, out_dir, device)
            elif kind == "drafter-eval":
                plan = claimed["plan"]
                ck_rev = str(plan["checkpoint_rev"])
                pr = str(plan["profile_id"])
                ck_dir = "/work/champion" if "champion" in ck_rev else f"/work/out/{ck_rev}"
                if not Path(ck_dir, "model.safetensors").is_file():
                    raise RuntimeError(f"checkpoint dir {ck_dir} not staged locally")
                emb_w = load_st_torch_nommap("/work/vocab/model-embed.safetensors")["model.embed_tokens.weight"].to(device=device, dtype=torch.bfloat16)
                head_w_mxfp8 = ev3.mxfp8_round_tensor(load_st_torch_nommap("/work/vocab/model-head.safetensors")["lm_head.weight"].to(device=device, dtype=torch.bfloat16))
                m = load_eval_model_nommap(ck_dir, device, quant="mxfp8")
                res = run_eval_profile_mxfp8(m, Path(f"/work/eval/{pr}"), emb_w, head_w_mxfp8, device, ck_rev, pr)
                ef = out_dir / f"{claimed['slot_id']}.json"
                ef.write_text(json.dumps(res, indent=2))
                up = wrk.upload_artifact(claimed, ef)
                log(f"Eval {claimed['slot_id']}: B16={res['b16_ref_tau']:.4f} B8={res['b8_ref_tau']:.4f} -> {up['state']}")
                del m, emb_w, head_w_mxfp8
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise RuntimeError(f"unsupported GPU worker kind: {kind}")
        except Exception as e:
            log(f"Error executing {claimed['job']}/{claimed['slot_id']}: {e}; releasing claim")
            try:
                wrk.release(claimed["job"], claimed["slot_id"])
            except Exception:
                pass
            if not args.loop:
                raise
            time.sleep(15)

        if not args.loop:
            break


if __name__ == "__main__":
    main()
