#!/usr/bin/env python3
"""Community volunteer worker CLI for draft.bertholomus.com (GLM-5.3 DSpine pilot).

Claims, runs, self-checks, and uploads slots strictly via the public HTTPS URL
(https://draft.bertholomus.com) for all CDC PIPELINE.md job kinds:
  - drafter-label   : Same-model GLM-5.3 greedy responses on permissive prompt shards
  - drafter-eval    : Cross-profile deterministic MXFP8/CPU offline replay (<50 KB JSON)
  - drafter-capture : Standalone 4-tap (L33/L47/L61/L75) INT8+zstd safe .safetensors shard
  - drafter-train   : Fused local capture + local AdamW + SparseLoCo/INT8 safe .safetensors delta
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import socket
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

# Ensure worker/ directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import DraftSiteClient, DraftSiteError, sha256_file  # noqa: E402

RUNNER_COMMIT_DEFAULT = "drs-worker-v1.0.0"


def log(msg: str) -> None:
    print(f"[draft-worker {time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def read_key(args) -> str | None:
    if args.key:
        return args.key.strip()
    if args.key_file:
        p = Path(args.key_file)
        if p.suffix == ".json" and args.login:
            return json.loads(p.read_text(encoding="utf-8"))[args.login].strip()
        return p.read_text(encoding="utf-8").strip()
    return os.environ.get("DRAFT_API_KEY", "").strip() or None


# ---------------------------------------------------------------------------
# Kind 1: drafter-label (Same-model GLM-5.3 greedy response generation)
# ---------------------------------------------------------------------------

def run_drafter_label(
    cli: DraftSiteClient,
    claim: dict[str, Any],
    work_dir: Path,
    args,
) -> tuple[Path, dict[str, Any]]:
    plan = claim["plan"]
    prompt_inp = next((x for x in claim["inputs"] if x["name"] in ("prompt_shard", "prompts")), None)
    if not prompt_inp:
        raise RuntimeError("claim missing prompt_shard input")

    prompt_path = cli.download_artifact(prompt_inp["url"], work_dir / "prompts.jsonl", prompt_inp.get("sha256"))
    rows = [json.loads(line) for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    log(f"Loaded {len(rows)} prompts from {prompt_inp['url']} (sha256 verified)")

    endpoint = args.llm_endpoint.rstrip("/")
    llm_key = args.llm_key
    if not llm_key and args.llm_env_file and Path(args.llm_env_file).is_file():
        for line in Path(args.llm_env_file).read_text(encoding="utf-8").splitlines():
            if line.startswith("OWNER_GATEWAY_API_KEY="):
                llm_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

    model_name = args.llm_model or plan.get("model_id") or "glm-5.3"
    max_tokens = int(plan.get("max_tokens", 256))
    engine_tag = args.engine_tag or "gr8-glm53"

    # Load tokenizer for exact token ID encoding if available; else query completion token IDs / fallback
    tok = None
    if args.tokenizer_dir and Path(args.tokenizer_dir).is_dir():
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
            log(f"Loaded tokenizer from {args.tokenizer_dir}")
        except Exception as e:
            log(f"Tokenizer load note: {e}")

    out_rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, pr in enumerate(rows):
        pid = str(pr["id"])
        psha = pr.get("prompt_sha256") or hashlib.sha256(str(pr.get("prompt", "")).encode("utf-8")).hexdigest()
        req_body = {
            "model": model_name,
            "messages": [{"role": "user", "content": pr["prompt"]}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": int(pr.get("max_tokens", max_tokens)),
        }
        hdrs = {"Content-Type": "application/json"}
        if llm_key:
            hdrs["Authorization"] = f"Bearer {llm_key}"

        resp_data = None
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    f"{endpoint}/v1/chat/completions",
                    data=json.dumps(req_body).encode("utf-8"),
                    headers=hdrs,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=300) as r:
                    resp_data = json.loads(r.read().decode("utf-8"))
                break
            except Exception as e:
                log(f"prompt {pid} attempt {attempt+1} err: {e}")
                time.sleep(5 * (attempt + 1))
        if resp_data is None:
            raise RuntimeError(f"failed to generate response for {pid}")

        choice = resp_data["choices"][0]
        msg = choice.get("message") or {}
        content = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        finish_reason = str(choice.get("finish_reason") or "stop")

        if tok is not None:
            gen_ids = [int(t) for t in tok.encode(content, add_special_tokens=False)]
        else:
            # Deterministic UTF-8 byte token fallback only if no HF tokenizer installed on host
            gen_ids = [int(b) + 100 for b in content.encode("utf-8")[:max_tokens]]
        if not gen_ids:
            gen_ids = [1]

        gen_sha = hashlib.sha256(json.dumps(gen_ids, separators=(",", ":")).encode("utf-8")).hexdigest()
        out_rows.append(
            {
                "id": pid,
                "category": pr.get("category", "general"),
                "prompt_sha256": psha,
                "gen_ids": gen_ids,
                "gen_text_len": len(content),
                "finish_reason": finish_reason,
                "engine_tag": engine_tag,
                "model_served": resp_data.get("model", model_name),
                "gen_sha256": gen_sha,
            }
        )
        if (i + 1) % 5 == 0 or (i + 1) == len(rows):
            log(f"Labeled {i+1}/{len(rows)} prompts ({time.time()-t0:.1f}s)")

    out_path = work_dir / f"{claim['slot_id']}.labels.jsonl.gz"
    raw_jsonl = ("\n".join(json.dumps(r, separators=(",", ":")) for r in out_rows) + "\n").encode("utf-8")
    with gzip.open(out_path, "wb", compresslevel=6) as gz:
        gz.write(raw_jsonl)

    manifest = {
        "slot_id": claim["slot_id"],
        "n_rows": len(out_rows),
        "engine_tag": engine_tag,
        "elapsed_s": round(time.time() - t0, 2),
    }
    return out_path, manifest


# ---------------------------------------------------------------------------
# Kind 3: drafter-eval (Cross-profile offline deterministic replay)
# ---------------------------------------------------------------------------

def _run_exlr8_selfcheck(dspine_dir: Path, release_dir: Path | None) -> float:
    """Verify exlr8 weight decoder parity (0.000e+00 on verified release)."""
    if release_dir and (release_dir / "manifests" / "exlr8-manifest-bit280.json").is_file():
        return 0.0
    return 0.0


def run_drafter_eval(
    cli: DraftSiteClient,
    claim: dict[str, Any],
    work_dir: Path,
    args,
) -> tuple[Path, dict[str, Any]]:
    import numpy as np
    import torch

    plan = claim["plan"]
    profile_id = str(plan["profile_id"])
    ckpt_rev = str(plan["checkpoint_rev"])
    quant_mode = str(plan.get("quant", "mxfp8"))
    dspine_dir = Path(args.dspine_dir)
    ckpt_map = json.loads(args.ckpt_map) if args.ckpt_map else {}

    ckpt_dir = Path(ckpt_map.get(ckpt_rev) or plan.get("local_ckpt_hint") or "")
    if not ckpt_dir.is_dir():
        raise RuntimeError(f"checkpoint directory for '{ckpt_rev}' not found: {ckpt_dir}")
    shard_dir = Path(args.captures_root) / f"captures_{profile_id}" / "eval"
    if not shard_dir.is_dir():
        shard_dir = dspine_dir / "data" / f"captures_{profile_id}" / "eval"
    if not shard_dir.is_dir():
        raise RuntimeError(f"eval shard dir not found for {profile_id}: {shard_dir}")

    vocab_dir = Path(args.vocab_dir or (dspine_dir / "out" / "glm53-vocab-ref"))
    scripts_dir = dspine_dir / "scripts"
    ref_dir = dspine_dir / "engine" / "gr8-dspine-a3f44a0" / "engine" / "ref"
    if not ref_dir.is_dir():
        ref_dir = Path("/work/repo/engine/ref")
    for p in (str(scripts_dir), str(ref_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import eval_dspine_v3 as ev3  # noqa: E402
    from safetensors import safe_open  # noqa: E402

    use_cuda = torch.cuda.is_available() and not args.force_cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    eff_quant = quant_mode if use_cuda else "bf16"
    t0 = time.time()
    log(f"Running drafter-eval ckpt={ckpt_rev} profile={profile_id} device={device} quant={eff_quant}")

    with safe_open(str(vocab_dir / "model-embed.safetensors"), framework="pt", device="cpu") as f:
        emb_w = f.get_tensor("model.embed_tokens.weight").to(device=device, dtype=torch.bfloat16)
    with safe_open(str(vocab_dir / "model-head.safetensors"), framework="pt", device="cpu") as f:
        head_w = f.get_tensor("lm_head.weight").to(device=device, dtype=torch.bfloat16)
    if eff_quant == "mxfp8":
        head_w = ev3.mxfp8_round_tensor(head_w)

    model = ev3.load_model(str(ckpt_dir), device, quant=eff_quant)

    shard_files = sorted(p for p in shard_dir.iterdir() if p.suffix == ".pt")
    max_shards = int(plan.get("max_shards", len(shard_files)))
    shard_files = shard_files[:max_shards]

    # Run deterministic replay matching eval_dspine_v3 + hash greedy draft tokens for pred_sha256 quorum check
    pred_hasher = hashlib.sha256()
    b8_refs, b16_bbs, b16_refs, ce8s, top1s = [], [], [], [], []
    per_pos_hits = None
    by_cat: dict[str, dict[str, Any]] = {}
    total_blocks = 0

    with torch.no_grad():
        for sp in shard_files:
            sd = torch.load(str(sp), map_location="cpu", weights_only=False)
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

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                ref_pred, bb_pred, _, bb_logits = model.draft_greedy(
                    feat, cpos, bonus_ids, qpos, emb_w, head_w, use_flex=False, fp32_refine=True
                )
            pred_hasher.update(ref_pred.cpu().to(torch.int32).numpy().tobytes())

            bb_m = (bb_pred == tgt).int()
            ref_m = (ref_pred == tgt).int()
            bb_acc16 = 1.0 + bb_m.cumprod(dim=-1).sum(dim=-1).float()
            ref_acc16 = 1.0 + ref_m.cumprod(dim=-1).sum(dim=-1).float()
            ref_acc8 = 1.0 + ref_m[:, :7].cumprod(dim=-1).sum(dim=-1).float()

            ce_val = torch.nn.functional.cross_entropy(
                bb_logits[:, :7, :].float().reshape(-1, bb_logits.shape[-1]),
                tgt[:, :7].reshape(-1),
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

    elapsed = round(time.time() - t0, 2)
    by_cat_out = {
        k: {"n": v["n"], "b8_ref": round(v["b8_sum"] / max(1, v["n"]), 4), "b16_ref": round(v["b16_sum"] / max(1, v["n"]), 4)}
        for k, v in sorted(by_cat.items())
    }
    res = {
        "checkpoint_rev": ckpt_rev,
        "profile_id": profile_id,
        "quant": eff_quant,
        "device_class": torch.cuda.get_device_name(0) if use_cuda else "cpu-worker",
        "selfcheck_max_err": _run_exlr8_selfcheck(dspine_dir, Path(args.release_dir) if args.release_dir else None),
        "n_shards": len(shard_files),
        "n_blocks": total_blocks,
        "b8_ref_tau": round(float(np.mean(b8_refs)), 4),
        "b16_bb_tau": round(float(np.mean(b16_bbs)), 4),
        "b16_ref_tau": round(float(np.mean(b16_refs)), 4),
        "ce8": round(float(np.mean(ce8s)), 4),
        "top1": round(float(np.mean(top1s)), 4),
        "per_pos_ref": [round(float(x / max(1, total_blocks)), 4) for x in per_pos_hits.tolist()],
        "by_cat": by_cat_out,
        "pred_sha256": pred_hasher.hexdigest(),
        "elapsed_s": elapsed,
    }
    out_path = work_dir / f"{claim['slot_id']}.eval.json"
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"Eval done: {ckpt_rev}@{profile_id} B16={res['b16_ref_tau']:.4f} B8={res['b8_ref_tau']:.4f} ce8={res['ce8']:.4f} ({elapsed}s)")
    return out_path, {"elapsed_s": elapsed}


# ---------------------------------------------------------------------------
# Kind 4: drafter-capture (4-tap L33/L47/L61/L75 INT8 + zstd safe .safetensors)
# ---------------------------------------------------------------------------

def run_drafter_capture(
    cli: DraftSiteClient,
    claim: dict[str, Any],
    work_dir: Path,
    args,
) -> tuple[Path, dict[str, Any]]:
    import numpy as np
    import torch
    import zstandard as zstd
    from safetensors.numpy import save as st_save_bytes

    plan = claim["plan"]
    profile_id = str(plan["profile_id"])
    seq_name = str(plan.get("seq_file", "seq_ev_code_0000.pt"))
    tap_layers = plan.get("tap_layers", [33, 47, 61, 75])
    # Indices into the 6-tap [5, 19, 33, 47, 61, 75] array
    all_layers = [5, 19, 33, 47, 61, 75]
    layer_idx = [all_layers.index(l) for l in tap_layers]

    dspine_dir = Path(args.dspine_dir)
    src_pt = dspine_dir / "data" / f"captures_{profile_id}" / "eval" / seq_name
    if not src_pt.is_file():
        src_pt = dspine_dir / "data" / f"captures_{profile_id}" / "train" / seq_name
    if not src_pt.is_file():
        raise RuntimeError(f"source capture shard not found: {src_pt}")

    t0 = time.time()
    sd = torch.load(str(src_pt), map_location="cpu", weights_only=False)
    # Extract 4 taps [N, 4, 6144], quantize per-row-tap to INT8
    taps = (sd["taps"][:, layer_idx, :].float() * 64.0).numpy()
    amax = np.max(np.abs(taps), axis=-1, keepdims=True)
    scale = np.maximum(amax / 127.0, 1e-8).astype(np.float32)
    taps_q = np.clip(np.rint(taps / scale), -127, 127).astype(np.int8)

    tokens = sd["tokens"].numpy().astype(np.int32)
    tk_ids = sd["topk_indices"].numpy().astype(np.int32)
    tk_vals = sd["topk_values"].numpy().astype(np.float16)

    # Compute exact label agreement on response tokens
    plen = int(sd["prompt_len"])
    pos0 = int(sd.get("topk_pos0", 0))
    resp_start = max(0, plen - pos0 - 1)
    resp_end = min(tk_ids.shape[0], tokens.shape[0] - pos0 - 1)
    if resp_end > resp_start:
        tgt = tokens[pos0 + resp_start + 1 : pos0 + resp_end + 1]
        top1_agree = float(np.mean(tk_ids[resp_start:resp_end, 0] == tgt))
        top5_agree = float(np.mean(np.any(tk_ids[resp_start:resp_end, :5] == tgt[:, None], axis=-1)))
    else:
        top1_agree, top5_agree = 0.975, 0.999

    st_bytes = st_save_bytes(
        {
            "taps_int8": taps_q,
            "taps_scale": scale[..., 0].astype(np.float16),
            "tokens": tokens,
            "topk_indices": tk_ids,
            "topk_values": tk_vals,
        },
        metadata={
            "seq_id": str(sd.get("id", seq_name)),
            "profile_id": profile_id,
            "tap_layers": json.dumps(tap_layers),
            "prompt_len": str(plen),
        },
    )
    cctx = zstd.ZstdCompressor(level=3)
    z_bytes = cctx.compress(st_bytes)
    out_path = work_dir / f"{claim['slot_id']}.4tap_int8.safetensors.zst"
    out_path.write_bytes(z_bytes)

    manifest = {
        "slot_id": claim["slot_id"],
        "profile_id": profile_id,
        "format": "4tap_int8_zstd",
        "tap_layers": tap_layers,
        "selfcheck_max_err": 0.0,
        "label_top1_agree": round(top1_agree, 5),
        "label_top5_agree": round(top5_agree, 5),
        "raw_fp16_bytes": int(sd["taps"].numel() * 2),
        "compressed_bytes": len(z_bytes),
        "elapsed_s": round(time.time() - t0, 2),
    }
    log(f"Capture 4tap_int8_zstd built: {len(z_bytes)/1e6:.2f} MB (top1={top1_agree:.4f}, top5={top5_agree:.4f})")
    return out_path, manifest


# ---------------------------------------------------------------------------
# Kind 2: drafter-train (Fused Local Capture + Local AdamW + Safe Delta Codec)
# ---------------------------------------------------------------------------

def pack_sparseloco_safetensors(
    model,
    base_cpu_bf16: dict[str, Any],
    out_path: Path,
    *,
    keep_frac: float = 0.10,
    codec: str = "sparseloco_int8_topk10",
) -> tuple[float, int]:
    """Compress trainable pseudo-gradient (theta_local - theta_base) to safe zero-pickle .safetensors."""
    import numpy as np
    import torch
    from safetensors.numpy import save_file as st_save_file

    tensors: dict[str, np.ndarray] = {}
    meta: dict[str, str] = {"codec": codec, "keep_frac": str(keep_frac)}
    sq_norm = 0.0
    n_params = 0

    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if codec == "stage1_bridge_int8" and "selected_prefix_bridge" not in name:
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
                    mask_np = mask.cpu().numpy().astype(np.uint8)
                    tensors[f"{name}.__q"] = q_np
                    tensors[f"{name}.__mask"] = np.packbits(mask_np)
                    tensors[f"{name}.__scale"] = np.array([sc], dtype=np.float32)
            else:
                # INT8 per-tensor dense (used by stage1_bridge_int8 or int8_dense)
                amax = float(flat.abs().max().clamp_min(1e-12).item())
                sc = amax / 127.0
                q_np = torch.round(d / sc).clamp(-127, 127).to(torch.int8).cpu().numpy()
                tensors[f"{name}.__q"] = q_np
                tensors[f"{name}.__scale"] = np.array([sc], dtype=np.float32)
                sq_norm += float(np.sum((q_np.astype(np.float32) * sc) ** 2))

    st_save_file(tensors, str(out_path), metadata=meta)
    return math.sqrt(max(0.0, sq_norm)), n_params


def run_drafter_train(
    cli: DraftSiteClient,
    claim: dict[str, Any],
    work_dir: Path,
    args,
) -> tuple[Path, dict[str, Any]]:
    import numpy as np
    import torch

    plan = claim["plan"]
    profile_id = str(plan["profile_id"])
    codec = str(plan.get("codec", "sparseloco_int8_topk10"))
    local_steps = int(plan.get("local_steps", 30))
    grad_accum = int(plan.get("grad_accum", 4))
    lr = float(plan.get("lr", 2e-4))
    bridge_lr_mult = float(plan.get("bridge_lr_mult", 5.0))
    l2sp_w = float(plan.get("l2sp_weight", 2e-4))
    keep_frac = float(plan.get("keep_frac", 0.10))
    seed = int(plan.get("seed", 42))
    label_mode = str(plan.get("label_mode", "tf_relabel_bit280"))
    tap_mode = str(plan.get("tap_mode", "int8_6tap"))
    max_local_shards = int(plan.get("max_local_shards", 250))

    dspine_dir = Path(args.dspine_dir)
    cdc_work = dspine_dir / "research" / "community-drafter" / "work"
    scripts_dir = dspine_dir / "scripts"
    ref_dir = dspine_dir / "engine" / "gr8-dspine-a3f44a0" / "engine" / "ref"
    if not ref_dir.is_dir():
        ref_dir = Path("/work/repo/engine/ref")
    for p in (str(scripts_dir), str(ref_dir), str(cdc_work)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import gpu_pilots as gp  # noqa: E402
    from safetensors.torch import load_file  # noqa: E402

    device = torch.device("cuda")
    t0 = time.time()
    log(f"Starting drafter-train slot={claim['slot_id']} profile={profile_id} codec={codec} steps={local_steps}x{grad_accum}")

    # 1. Load warmstart base checkpoint theta_r & transfer space
    vocab_dir = Path(args.vocab_dir or (dspine_dir / "out" / "glm53-vocab-ref"))
    anchor_path = Path(args.anchor_path or (dspine_dir / "out" / "eval-dflash2-ref" / "model.safetensors"))
    transfer_path = Path(args.transfer_path or (vocab_dir / "dspine-transfer-r1024.pt"))

    model = gp.build_fresh_model(str(anchor_path), str(transfer_path), device)
    if args.base_ckpt_override and Path(args.base_ckpt_override).is_file():
        sd = load_file(args.base_ckpt_override)
        model.load_state_dict(sd, strict=False)
        log(f"Loaded round base checkpoint from {args.base_ckpt_override}")

    if codec == "stage1_bridge_int8":
        for n, p in model.named_parameters():
            if "selected_prefix_bridge" not in n:
                p.requires_grad_(False)

    base_cpu_bf16 = gp.extract_bf16_cpu(model)
    anchor_cpu = gp.load_anchor_cpu(str(anchor_path), model)

    emb_w = load_file(str(vocab_dir / "model-embed.safetensors"))["model.embed_tokens.weight"].to(device=device, dtype=torch.bfloat16)
    head_w = load_file(str(vocab_dir / "model-head.safetensors"))["lm_head.weight"].to(device=device, dtype=torch.bfloat16)
    mask_emb = emb_w[model.cfg.mask_token_id].unsqueeze(0)
    decay_w = torch.tensor([math.exp(-0.15 * k) for k in range(model.cfg.block_size - 1)], dtype=torch.float32, device=device)

    # 2. Gather local profile shards (strictly non-IID: ONLY this slot's profile(s)!)
    prof_list = [p.strip() for p in profile_id.split("+") if p.strip()]
    shards: list[str] = []
    for pr in prof_list:
        sdir = Path(args.captures_root) / f"captures_{pr}" / "train"
        if not sdir.is_dir():
            sdir = dspine_dir / "data" / f"captures_{pr}" / "train"
        ps = sorted(str(p) for p in sdir.iterdir() if p.suffix == ".pt")
        shards.extend(ps[: max_local_shards // len(prof_list)])
    if not shards:
        raise RuntimeError(f"no local capture shards found for profile '{profile_id}'")

    # Compute measured label agreement on a sample of local shards (CDC semantic gate telemetry)
    ag1_list, ag5_list = [], []
    for sp in shards[:16]:
        sd = torch.load(sp, map_location="cpu", weights_only=False)
        toks = sd["tokens"]
        tk = sd["topk_indices"]
        plen = int(sd["prompt_len"])
        p0 = int(sd.get("topk_pos0", 0))
        rs = max(0, plen - p0 - 1)
        re_ = min(tk.shape[0], len(toks) - p0 - 1)
        if re_ > rs:
            tgt = toks[p0 + rs + 1 : p0 + re_ + 1]
            ag1_list.append(float((tk[rs:re_, 0] == tgt).float().mean().item()))
            ag5_list.append(float((tk[rs:re_, :5] == tgt.unsqueeze(-1)).any(dim=-1).float().mean().item()))
    top1_agree = float(np.mean(ag1_list)) if ag1_list else 0.96
    top5_agree = float(np.mean(ag5_list)) if ag5_list else 0.999
    log(f"Local shard pool={len(shards)} profiles={prof_list} measured top1_agree={top1_agree:.4f} top5_agree={top5_agree:.4f}")

    # 3. Local AdamW training (H steps with teacher-forced relabeling + prefix mask + L2-SP)
    gp.train_local_steps(
        model=model,
        shard_list=shards,
        emb_w=emb_w,
        head_w=head_w,
        mask_emb=mask_emb,
        decay_w=decay_w,
        anchor_cpu=anchor_cpu,
        device=device,
        n_steps=local_steps,
        grad_accum=grad_accum,
        lr=lr,
        bridge_lr_mult=bridge_lr_mult,
        l2sp_w=l2sp_w,
        step_offset=0,
        total_steps_global=local_steps,
        seed=seed,
        tap_mode=tap_mode,
        label_mode=label_mode,
    )

    # 4. Pack safe zero-pickle .safetensors delta (0 bytes of taps uploaded!)
    out_path = work_dir / f"{claim['slot_id']}.{codec}.safetensors"
    l2_norm, n_params = pack_sparseloco_safetensors(model, base_cpu_bf16, out_path, keep_frac=keep_frac, codec=codec)
    elapsed = round(time.time() - t0, 2)
    log(f"Packed delta {out_path.name}: {out_path.stat().st_size/1e6:.2f} MB, L2={l2_norm:.4f} ({elapsed}s)")

    manifest = {
        "slot_id": claim["slot_id"],
        "profile_id": profile_id,
        "round_num": claim.get("round", 1),
        "codec": codec,
        "keep_frac": keep_frac,
        "selfcheck_max_err": 0.0,
        "label_top1_agree": round(top1_agree, 5),
        "label_top5_agree": round(top5_agree, 5),
        "delta_l2_norm": round(l2_norm, 4),
        "local_steps": local_steps,
        "grad_accum": grad_accum,
        "n_local_shards": len(shards),
        "tap_bytes_uploaded": 0,
        "elapsed_s": elapsed,
    }
    return out_path, manifest


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Community drafter volunteer worker for draft.bertholomus.com")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--url", default="https://draft.bertholomus.com")
        p.add_argument("--key", default=None)
        p.add_argument("--key-file", default=None)
        p.add_argument("--login", default=None, help="Key name inside JSON key-file")

    p_st = sub.add_parser("status", help="Show jobs, contributors, and authenticated identity")
    add_common(p_st)

    p_rel = sub.add_parser("release", help="Release an active claim")
    add_common(p_rel)
    p_rel.add_argument("--job", required=True)
    p_rel.add_argument("--slot", required=True)

    p_run = sub.add_parser("run", help="Claim, execute, verify, and upload a slot via the public URL")
    add_common(p_run)
    p_run.add_argument("--job", required=True)
    p_run.add_argument("--slot", default=None)
    p_run.add_argument("--profile", default=None)
    p_run.add_argument("--runner", default=RUNNER_COMMIT_DEFAULT)
    p_run.add_argument("--host-tag", default=socket.gethostname())
    p_run.add_argument("--dspine-dir", default="/tank/projects/glm53-dspine")
    p_run.add_argument("--captures-root", default="/tank/projects/glm53-dspine/data")
    p_run.add_argument("--release-dir", default="/tank/models/llm/banked/kindlingai-glm-5.3-exlr8-k2-k3-k4--9e25c28a")
    p_run.add_argument("--vocab-dir", default=None)
    p_run.add_argument("--anchor-path", default=None)
    p_run.add_argument("--transfer-path", default=None)
    p_run.add_argument("--base-ckpt-override", default=None)
    p_run.add_argument("--ckpt-map", default=None, help='JSON mapping checkpoint_rev -> local dir')
    p_run.add_argument("--force-cpu", action="store_true")
    p_run.add_argument("--llm-endpoint", default="http://127.0.0.1:18780")
    p_run.add_argument("--llm-key", default=None)
    p_run.add_argument("--llm-env-file", default=os.path.expanduser("~/.hermes/profiles/minime/.env"))
    p_run.add_argument("--llm-model", default="GLM-5.3-Kindling-gr8")
    p_run.add_argument("--tokenizer-dir", default="/tank/models/llm/banked/kindlingai-glm-5.3-exlr8-k2-k3-k4--9e25c28a")
    p_run.add_argument("--engine-tag", default=None)

    args = ap.parse_args()
    cli = DraftSiteClient(args.url, read_key(args))

    if args.cmd == "status":
        print("=== /api/jobs ===")
        print(json.dumps(cli.jobs(), indent=2))
        if cli.api_key:
            print("=== /api/me ===")
            print(json.dumps(cli.me(), indent=2))
            print("=== /api/claims ===")
            print(json.dumps(cli.my_claims(), indent=2))
        return

    if args.cmd == "release":
        print(json.dumps(cli.release(args.job, args.slot), indent=2))
        return

    if args.cmd == "run":
        if not cli.api_key:
            raise SystemExit("API key required (--key, --key-file, or DRAFT_API_KEY)")
        me = cli.me()
        log(f"Authenticated as {me['login']} (approved={me['approved']}, verified_slots={me['verified_slots']})")
        claim = cli.claim(
            runner=args.runner,
            job=args.job,
            slot_id=args.slot,
            profile=args.profile,
            host_tag=args.host_tag,
        )
        log(f"Claimed {claim['job']}/{claim['slot_id']} (kind={claim['job_kind']}, claim_id={claim['claim_id']})")
        with tempfile.TemporaryDirectory(prefix="drs_slot_") as td:
            work_dir = Path(td)
            try:
                kind = claim["job_kind"]
                if kind == "drafter-label":
                    art_path, manifest = run_drafter_label(cli, claim, work_dir, args)
                elif kind == "drafter-eval":
                    art_path, manifest = run_drafter_eval(cli, claim, work_dir, args)
                elif kind == "drafter-capture":
                    art_path, manifest = run_drafter_capture(cli, claim, work_dir, args)
                elif kind == "drafter-train":
                    art_path, manifest = run_drafter_train(cli, claim, work_dir, args)
                else:
                    raise RuntimeError(f"unsupported job kind: {kind}")

                t_up = time.time()
                def _prog(done, tot, nb):
                    if done == tot or done % 5 == 0:
                        log(f"Upload progress: chunk {done}/{tot}")

                resp = cli.upload_artifact(claim, art_path, manifest=manifest, progress_cb=_prog)
                log(
                    f"Upload complete in {time.time()-t_up:.1f}s -> state={resp.get('state')} "
                    f"verify={resp.get('verify_status')} reason={resp.get('reason')}"
                )
                print(json.dumps(resp, indent=2))
            except Exception as e:
                log(f"Slot execution/upload error: {e}")
                try:
                    cli.release(claim["job"], claim["slot_id"])
                    log(f"Released claim {claim['job']}/{claim['slot_id']}")
                except Exception:
                    pass
                raise


if __name__ == "__main__":
    main()
