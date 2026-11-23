#!/usr/bin/env python3
"""Owner GPU Verification Gate & Outer-Round Aggregator for draft.bertholomus.com.

Implements CDC PIPELINE.md §§5.1-5.2:
1. Pulls pending uploaded drafter-train .safetensors deltas from the coordinator.
2. NEVER uses pickle (loads strictly via safetensors.numpy/torch safe loader).
3. Computes Owner's 10-step root gradient Delta_root on private held-out anchor shards.
4. Evaluates every submitted delta theta_r + Delta_i on the private 26-shard held-out eval set (~13s/delta):
   - Zero NaN/Inf
   - Strict loss reduction: delta_ce8 < -0.005
   - FLTrust cosine vs Delta_root: cos > 0.05
   - L2 norm within [0.3x, 2.5x] honest median
5. Posts accept/reject verdicts to /api/admin/verifications/<job>/<slot_id> via https://draft.bertholomus.com.
6. Aggregates verified honest deltas (norm-clipped coordinate-wise trimmed mean / soup),
   saves model.safetensors + config.json + NOTICE, and runs full offline scoring
   (sim14 14 served dumps + 5-profile multiprofile MXFP8) against live champion v5-s2-w5@500!
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import DraftSiteClient, sha256_file  # noqa: E402


def log(msg: str) -> None:
    print(f"[owner-gate {time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def load_safe_delta_f32(path: Path, ref_shapes: dict[str, tuple[int, ...]]) -> dict[str, torch.Tensor]:
    """Safely dequantize a SparseLoCo/INT8 .safetensors delta into CPU FP32 tensors (zero pickle)."""
    out: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="numpy") as f:
        meta = f.metadata() or {}
        keys = set(f.keys())
        codec = meta.get("codec", "sparseloco_int8_topk10")
        for name, shape in ref_shapes.items():
            n = 1
            for d in shape:
                n *= d
            if f"{name}.__q" in keys:
                q = f.get_tensor(f"{name}.__q")
                sc = float(f.get_tensor(f"{name}.__scale").reshape(-1)[0])
                if f"{name}.__mask" in keys:
                    packed = f.get_tensor(f"{name}.__mask")
                    bits = np.unpackbits(packed)[:n].astype(bool)
                    flat = np.zeros(n, dtype=np.float32)
                    nnz = int(bits.sum())
                    if nnz != q.size:
                        raise ValueError(f"mask popcount {nnz} != q size {q.size} on {name}")
                    flat[bits] = q.astype(np.float32) * sc
                    out[name] = torch.from_numpy(flat.reshape(shape))
                elif f"{name}.__idx" in keys:
                    idx = f.get_tensor(f"{name}.__idx")
                    flat = np.zeros(n, dtype=np.float32)
                    flat[idx] = q.astype(np.float32) * sc
                    out[name] = torch.from_numpy(flat.reshape(shape))
                else:
                    out[name] = torch.from_numpy((q.astype(np.float32) * sc).reshape(shape))
            elif f"{name}.__dense" in keys:
                arr = f.get_tensor(f"{name}.__dense").astype(np.float32)
                out[name] = torch.from_numpy(arr.reshape(shape))
            elif name in keys:
                arr = f.get_tensor(name).astype(np.float32)
                out[name] = torch.from_numpy(arr.reshape(shape))
            else:
                out[name] = torch.zeros(shape, dtype=torch.float32)
    return out


def delta_l2_and_cos(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> tuple[float, float]:
    sq_a = 0.0
    sq_b = 0.0
    dot = 0.0
    for k in a:
        va = a[k].view(-1)
        vb = b[k].view(-1)
        sq_a += float(torch.dot(va, va).item())
        sq_b += float(torch.dot(vb, vb).item())
        dot += float(torch.dot(va, vb).item())
    na = math.sqrt(max(0.0, sq_a))
    nb = math.sqrt(max(0.0, sq_b))
    cos = dot / max(na * nb, 1e-12)
    return na, cos


def main() -> None:
    ap = argparse.ArgumentParser(description="Owner GPU Held-Out Replay + FLTrust Gate and Aggregator")
    ap.add_argument("--url", default="http://127.0.0.1:8742")
    ap.add_argument("--admin-key-file", required=True)
    ap.add_argument("--origin-auth-file", default="/tank/projects/draft-site/secrets/origin_auth.secret")
    ap.add_argument("--job", default="glm53-dspine-train")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--data-dir", default="/tank/projects/draft-site/data")
    ap.add_argument("--dspine-dir", default="/tank/projects/glm53-dspine")
    ap.add_argument("--captures-root", default="/tank/projects/glm53-dspine/data")
    ap.add_argument("--vocab-dir", default=None)
    ap.add_argument("--anchor-path", default=None)
    ap.add_argument("--transfer-path", default=None)
    ap.add_argument("--dumps-b3-dir", default=None)
    ap.add_argument("--dumps-mc-dir", default=None)
    ap.add_argument("--out-ckpt-dir", default="/tank/projects/draft-site/data/evals/pilot-r1-agg")
    ap.add_argument("--eval-out", default="/tank/projects/draft-site/data/evals/pilot-r1-offline-scores.json")
    ap.add_argument("--outer-lr", type=float, default=1.0)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    dspine_dir = Path(args.dspine_dir)
    for p in (
        str(dspine_dir / "scripts"),
        str(dspine_dir / "engine" / "gr8-dspine-a3f44a0" / "engine" / "ref"),
        str(dspine_dir / "research" / "community-drafter" / "work"),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)

    import gpu_pilots as gp  # noqa: E402

    admin_key = Path(args.admin_key_file).read_text(encoding="utf-8").strip()
    origin_auth = Path(args.origin_auth_file).read_text(encoding="utf-8").strip() if args.origin_auth_file and Path(args.origin_auth_file).is_file() else None
    cli = DraftSiteClient(args.url, admin_key, origin_auth=origin_auth)

    pending = [x for x in cli.admin_get("/api/admin/pending-verifications") if x["job"] == args.job and x["round_num"] == args.round]
    log(f"Found {len(pending)} pending uploaded deltas for {args.job} round {args.round}")
    if not pending:
        return

    device = torch.device("cuda")
    vocab_dir = Path(args.vocab_dir) if args.vocab_dir else (dspine_dir / "out" / "glm53-vocab-ref")
    anchor_path = Path(args.anchor_path) if args.anchor_path else (dspine_dir / "out" / "eval-dflash2-ref" / "model.safetensors")
    transfer_path = Path(args.transfer_path) if args.transfer_path else (vocab_dir / "dspine-transfer-r1024.pt")

    emb_w = load_file(str(vocab_dir / "model-embed.safetensors"))["model.embed_tokens.weight"].to(device=device, dtype=torch.bfloat16)
    head_w = load_file(str(vocab_dir / "model-head.safetensors"))["lm_head.weight"].to(device=device, dtype=torch.bfloat16)

    model = gp.build_fresh_model(str(anchor_path), str(transfer_path), device)
    base_bf16 = gp.extract_bf16_cpu(model)
    anchor_cpu = gp.load_anchor_cpu(str(anchor_path), model)
    ref_shapes = {k: tuple(v.shape) for k, v in base_bf16.items()}
    mask_emb = emb_w[model.cfg.mask_token_id].unsqueeze(0)
    decay_w = torch.tensor([math.exp(-0.15 * k) for k in range(model.cfg.block_size - 1)], dtype=torch.float32, device=device)

    # 1. Load private held-out bit280 eval set & baseline score of theta_0
    val280_dir = Path(args.captures_root) / "captures_bit280" / "eval"
    val280 = gp.load_eval_profile(str(val280_dir))
    base_eval = gp.eval_shards_tapmode(model, val280, emb_w, head_w, device)
    log(f"Base theta_0 private val280: B16={base_eval['b16_ref_tau']:.4f} B8={base_eval['b8_ref_tau']:.4f} ce8={base_eval['ce8']:.4f}")

    # 2. Compute Owner's 10-step root direction Delta_root on private anchor shards (FLTrust)
    t_root = time.time()
    root_shards = sorted(str(p) for p in (Path(args.captures_root) / "captures_bit280" / "train").iterdir() if p.suffix == ".pt")[-40:]
    gp.train_local_steps(
        model=model, shard_list=root_shards, emb_w=emb_w, head_w=head_w, mask_emb=mask_emb, decay_w=decay_w,
        anchor_cpu=anchor_cpu, device=device, n_steps=10, grad_accum=4, lr=2e-4, bridge_lr_mult=5.0,
        l2sp_w=2e-4, step_offset=0, total_steps_global=30, seed=999, tap_mode="int8_6tap", label_mode="tf_relabel_bit280",
    )
    delta_root = {
        k: (p.detach().float().cpu() - base_bf16[k].float())
        for k, p in model.named_parameters()
        if p.requires_grad
    }
    root_norm, _ = delta_l2_and_cos(delta_root, delta_root)
    log(f"Computed Owner 10-step FLTrust root direction in {time.time()-t_root:.1f}s (norm={root_norm:.4f})")

    # 3. Evaluate every pending delta (first compute norms + eval scores, then apply norm band against median)
    evaluated: list[dict[str, Any]] = []
    for item in pending:
        art_path = Path(args.data_dir) / item["rel_path"]
        t_ev = time.time()
        try:
            d_f32 = load_safe_delta_f32(art_path, ref_shapes)
            l2_norm, cos_root = delta_l2_and_cos(d_f32, delta_root)
            # Load theta_0 + Delta_i onto GPU and run held-out replay
            cand_bf16 = {k: (base_bf16[k].float() + d_f32[k]).to(torch.bfloat16) for k in base_bf16}
            gp.load_from_cpu(model, cand_bf16)
            ev = gp.eval_shards_tapmode(model, val280, emb_w, head_w, device)
            d_ce8 = round(float(ev["ce8"] - base_eval["ce8"]), 5)
            d_b16 = round(float(ev["b16_ref_tau"] - base_eval["b16_ref_tau"]), 5)
            elapsed = round(time.time() - t_ev, 2)
            evaluated.append({
                "item": item, "delta": d_f32, "norm": round(l2_norm, 4), "cos": round(cos_root, 5),
                "d_ce8": d_ce8, "d_b16": d_b16, "val_b16": round(ev["b16_ref_tau"], 4), "val_ce8": round(ev["ce8"], 4),
                "elapsed_s": elapsed, "err": None,
            })
            log(f"  Evaluated {item['slot_id']} (by {item['uploaded_by']}): d_ce8={d_ce8:+.4f} d_b16={d_b16:+.4f} cos={cos_root:+.4f} norm={l2_norm:.3f} ({elapsed}s)")
        except Exception as e:
            evaluated.append({"item": item, "delta": None, "err": str(e), "norm": 0.0, "cos": 0.0, "d_ce8": 99.0, "d_b16": -99.0, "elapsed_s": round(time.time() - t_ev, 2)})

    # Compute median norm of positive-cosine candidates (so a single noise attack doesn't skew median!)
    good_norms = [e["norm"] for e in evaluated if e["err"] is None and e["cos"] > 0.05 and e["d_ce8"] < -0.005]
    med_norm = float(np.median(good_norms)) if good_norms else float(np.median([e["norm"] for e in evaluated if e["norm"] > 0] or [1.0]))

    accepted_deltas: list[dict[str, torch.Tensor]] = []
    accepted_meta: list[dict[str, Any]] = []
    gate_summary: list[dict[str, Any]] = []

    for e in evaluated:
        it = e["item"]
        reasons = []
        if e["err"]:
            reasons.append(f"parse_error: {e['err']}")
        else:
            if e["d_ce8"] >= -0.005:
                reasons.append(f"d_ce8={e['d_ce8']:+.4f} >= -0.005")
            if e["cos"] <= 0.05:
                reasons.append(f"cos_vs_root={e['cos']:+.4f} <= 0.05")
            if e["norm"] < 0.30 * med_norm or e["norm"] > 2.50 * med_norm:
                reasons.append(f"norm={e['norm']:.3f} outside [0.3x, 2.5x] median ({med_norm:.3f})")
        passed = len(reasons) == 0
        reason_str = (
            f"PASS held-out replay+FLTrust (d_ce8={e['d_ce8']:+.4f}, cos={e['cos']:+.4f}, norm={e['norm']:.2f})"
            if passed
            else "REJECTED by Held-Out Replay+FLTrust Gate: " + "; ".join(reasons)
        )
        gate_metrics = {
            "delta_ce8": e["d_ce8"],
            "delta_b16": e["d_b16"],
            "val280_b16": e.get("val_b16"),
            "val280_ce8": e.get("val_ce8"),
            "cos_vs_root": e["cos"],
            "l2_norm": e["norm"],
            "median_honest_norm": round(med_norm, 4),
            "gate_elapsed_s": e["elapsed_s"],
            "passed": passed,
        }
        vresp = cli.admin_post(
            f"/api/admin/verifications/{args.job}/{it['slot_id']}",
            {"passed": passed, "reason": reason_str, "gate_metrics": gate_metrics, "reopen_on_reject": True},
        )
        log(f"Posted verdict for {it['slot_id']}: passed={passed} -> server state={vresp.get('state')}")
        gate_summary.append({"slot_id": it["slot_id"], "uploaded_by": it["uploaded_by"], "passed": passed, "reason": reason_str, **gate_metrics})
        if passed:
            accepted_deltas.append(e["delta"])
            accepted_meta.append(e)

    if args.verify_only or not accepted_deltas:
        log(f"Verify-only or no accepted deltas (accepted={len(accepted_deltas)}). Done.")
        return

    # 4. Aggregate verified honest deltas (norm-clip + coordinate-wise trimmed mean when M>=5, else mean)
    M = len(accepted_deltas)
    c_med = 1.5 * med_norm
    log(f"Aggregating {M} verified honest deltas (c_med={c_med:.3f}, outer_lr={args.outer_lr})")
    agg_bf16: dict[str, torch.Tensor] = {}
    for k in base_bf16:
        stack = []
        for idx, d in enumerate(accepted_deltas):
            nm = accepted_meta[idx]["norm"]
            scale = min(1.0, c_med / max(nm, 1e-12))
            stack.append(d[k] * scale)
        stacked = torch.stack(stack, dim=0)  # [M, ...]
        if M >= 5:
            trim = max(1, int(0.1 * M))
            sorted_s, _ = torch.sort(stacked, dim=0)
            mean_d = sorted_s[trim : M - trim].mean(dim=0)
            agg_method = f"trimmed_mean_{M}w"
        else:
            mean_d = stacked.mean(dim=0)
            agg_method = f"soup_mean_{M}w_noniid"
        agg_bf16[k] = (base_bf16[k].float() + args.outer_lr * mean_d).to(torch.bfloat16)

    gp.load_from_cpu(model, agg_bf16)

    # 5. Save aggregated checkpoint (with NOTICE preserved!)
    out_dir = Path(args.out_ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.export_hf(str(out_dir))
    notice_src = dspine_dir / "NOTICE"
    if notice_src.is_file():
        shutil.copy2(notice_src, out_dir / "NOTICE")
    ckpt_sha = sha256_file(out_dir / "model.safetensors")
    log(f"Saved aggregated checkpoint to {out_dir} (sha256={ckpt_sha[:16]}...)")

    # 6. Full offline evaluation on EXACT SAME rows as live champion (sim14 MXFP8 + 5-profile multiprofile MXFP8)!
    log("Running full offline evaluation (sim14 14 served dumps + 5-profile multiprofile MXFP8)...")
    import eval_dspine_v3 as ev3  # noqa: E402
    model_eval = ev3.load_model(str(out_dir), device, quant="mxfp8")
    head_w_mxfp8 = ev3.mxfp8_round_tensor(head_w)

    b3_dir = Path(args.dumps_b3_dir) if args.dumps_b3_dir else (dspine_dir / "rows" / "b2r-slot-20261120T071919Z" / "b3")
    mc_dir = Path(args.dumps_mc_dir) if args.dumps_mc_dir else (dspine_dir / "rows" / "eval-mc-20261120T083643Z" / "split")
    dump_specs = [
        ("32k", "longctx_32k", str(b3_dir / "b2-dfl-32k.bin")),
        ("100k", "longctx_100k", str(b3_dir / "b2-dfl-100k.bin")),
        ("code0", "code", str(mc_dir / "00_ev_code_0000.bin")),
        ("code1", "code", str(mc_dir / "01_ev_code_0001.bin")),
        ("code2", "code", str(mc_dir / "02_ev_code_0002.bin")),
        ("struct0", "structured", str(mc_dir / "03_ev_structured_0000.bin")),
        ("struct1", "structured", str(mc_dir / "04_ev_structured_0001.bin")),
        ("struct2", "structured", str(mc_dir / "05_ev_structured_0002.bin")),
        ("prose0", "prose", str(mc_dir / "06_ev_prose_0000.bin")),
        ("prose1", "prose", str(mc_dir / "07_ev_prose_0001.bin")),
        ("prose2", "prose", str(mc_dir / "08_ev_prose_0002.bin")),
        ("lc0", "longctx_code", str(mc_dir / "09_ev_lc_0000.bin")),
        ("lc1", "longctx_code", str(mc_dir / "10_ev_lc_0001.bin")),
        ("lc2", "longctx_code", str(mc_dir / "11_ev_lc_0002.bin")),
    ]
    # Evaluate 5 profiles + 14 served dumps in MXFP8
    mp_profiles = {}
    b8s, b16s = [], []
    for pr in ("bit200", "bit240", "bit280", "bit322", "bit400"):
        vc = gp.load_eval_profile(str(Path(args.captures_root) / f"captures_{pr}" / "eval"))
        vr = gp.eval_shards_tapmode(model_eval, vc, emb_w, head_w_mxfp8, device)
        mp_profiles[pr] = {"b8_ref": vr["b8_ref_tau"], "b16_ref": vr["b16_ref_tau"], "ce8": vr["ce8"], "top1": vr["top1_ref"]}
        b8s.append(vr["b8_ref_tau"])
        b16s.append(vr["b16_ref_tau"])
        del vc
        gc.collect()
    d14 = gp.eval_dumps_tapmode(model_eval, dump_specs, emb_w, head_w_mxfp8, device, tap_mode="fp16_6tap", max_win=2048)
    full_eval = {
        "profiles": mp_profiles,
        "mp_mean_b8": round(float(np.mean(b8s)), 4),
        "mp_mean_b16": round(float(np.mean(b16s)), 4),
        "dumps14": d14,
    }

    # Compare against live champion (out/dspine-v5-s2-w5/sim14-mxfp8.json & multiprofile-mxfp8.json)
    champ_s14 = json.loads((dspine_dir / "out" / "dspine-v5-s2-w5" / "sim14-mxfp8.json").read_text())
    champ_mp = json.loads((dspine_dir / "out" / "dspine-v5-s2-w5" / "multiprofile-mxfp8.json").read_text())
    champ_mp_b16 = round(float(np.mean([v["b16_ref_tau"] for v in champ_mp["profiles"].values()])), 4)
    champ_mp_b8 = round(float(np.mean([v["b8_ref_tau"] for v in champ_mp["profiles"].values()])), 4)

    report = {
        "job": args.job,
        "round": args.round,
        "agg_method": agg_method,
        "n_verified_workers": M,
        "ckpt_dir": str(out_dir),
        "ckpt_sha256": ckpt_sha,
        "base_theta0_val280": {"b16_ref": base_eval["b16_ref_tau"], "b8_ref": base_eval["b8_ref_tau"], "ce8": base_eval["ce8"]},
        "gate_verdicts": gate_summary,
        "offline_eval_mxfp8": full_eval,
        "champion_v5_s2_w5_mxfp8": {
            "mp_mean_b8": champ_mp_b8,
            "mp_mean_b16": champ_mp_b16,
            "mp_profiles": {k: {"b8_ref": v["b8_ref_tau"], "b16_ref": v["b16_ref_tau"], "ce8": v["ce8"]} for k, v in champ_mp["profiles"].items()},
            "sim14_overall": champ_s14.get("overall", {}),
        },
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Path(args.eval_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.eval_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(
        f"OFFLINE EVAL COMPLETE: agg MP B16={full_eval['mp_mean_b16']} (B8={full_eval['mp_mean_b8']}) | "
        f"14dumps B16={full_eval['dumps14']['mean_b16_ref']} roll16={full_eval['dumps14']['mean_roll16']} | "
        f"Champion MP B16={champ_mp_b16}"
    )

    # 7. Publish aggregated round result to draft.bertholomus.com via Admin API
    cli.admin_post(
        f"/api/admin/rounds/{args.job}/{args.round}/aggregate",
        {
            "agg_method": agg_method,
            "agg_ckpt_name": "pilot-r1-agg",
            "agg_ckpt_sha256": ckpt_sha,
            "agg_metrics": {
                "mp_mean_b8": full_eval["mp_mean_b8"],
                "mp_mean_b16": full_eval["mp_mean_b16"],
                "sim14_b8": full_eval["dumps14"]["mean_b8_ref"],
                "sim14_b16": full_eval["dumps14"]["mean_b16_ref"],
                "sim14_roll16": full_eval["dumps14"]["mean_roll16"],
                "profiles": full_eval["profiles"],
                "champion_ref": {"mp_mean_b16": champ_mp_b16, "sim14_b16": champ_s14.get("overall", {}).get("dsp_b16_ref", 5.75)},
            },
        },
    )
    log("Published round aggregate to https://draft.bertholomus.com!")


if __name__ == "__main__":
    main()
