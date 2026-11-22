#!/usr/bin/env python3
"""Seed the GLM-5.3 DSpine community drafter pilot jobs on https://draft.bertholomus.com.

Idempotently:
1. Stages permissive prompt shards (from open-perfectblend Apache-2.0/MIT train_prompts.jsonl) under /data/inputs/.
2. Registers accepted runner commit(s) via /api/admin/runners.
3. Defines all 4 CDC PIPELINE.md job kinds for kindlingai/glm-5.3-exlr8-k2-k3-k4@9e25c28a:
   - glm53-dspine-labels   (drafter-label)
   - glm53-dspine-train    (drafter-train, fused capture+SparseLoCo/Soup + readiness-gated slot)
   - glm53-dspine-eval     (drafter-eval, 11 mul1 profiles x checkpoints, duplicate quorum slot)
   - glm53-dspine-capture  (drafter-capture, optional 4-tap L33/L47/L61/L75 INT8+zstd archive)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import DraftSiteClient  # noqa: E402

RELEASE_REPO = "kindlingai/glm-5.3-exlr8-k2-k3-k4"
RELEASE_REV = "9e25c28a2e8808d8b2599d780521fd6af24e7558"
ALL_11_PROFILES = ["bit200", "bit220", "bit240", "bit250", "bit271", "bit280", "bit300", "bit322", "bit350", "bit375", "bit400"]
READY_5_PROFILES = {"bit200", "bit240", "bit280", "bit322", "bit400"}


def stage_prompt_shards(src_prompts: Path, inputs_dir: Path) -> list[dict]:
    inputs_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [json.loads(line) for line in src_prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    specs = [
        ("shard_0000", 0, 12, 160),   # Pilot quick shard 0 (12 permissive prompts, fast real GLM-5.3 generation)
        ("shard_0001", 12, 24, 160),  # Pilot quick shard 1 (12 permissive prompts, second host / red-team test)
        ("shard_0002", 24, 524, 768), # Production-sized 500-prompt shard
        ("shard_0003", 524, 1024, 768),
    ]
    staged = []
    for name, lo, hi, max_toks in specs:
        chunk = []
        for r in all_rows[lo:hi]:
            ptxt = r.get("prompt") or r.get("text") or ""
            chunk.append(
                {
                    "id": str(r["id"]),
                    "category": str(r.get("category", "code")),
                    "source": str(r.get("source", "open-perfectblend-permissive")),
                    "prompt": ptxt,
                    "prompt_sha256": hashlib.sha256(ptxt.encode("utf-8")).hexdigest(),
                    "max_tokens": max_toks,
                }
            )
        out_file = inputs_dir / f"prompts_{name}.jsonl"
        out_file.write_text("\n".join(json.dumps(x) for x in chunk) + "\n", encoding="utf-8")
        staged.append({"shard": name, "rel_path": f"inputs/prompts_{name}.jsonl", "n_prompts": len(chunk), "max_tokens": max_toks})
    return staged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://draft.bertholomus.com")
    ap.add_argument("--admin-key-file", default="/tank/projects/draft-site/secrets/admin_api_key.txt")
    ap.add_argument("--data-dir", default="/tank/projects/draft-site/data")
    ap.add_argument("--prompts-src", default="/tank/projects/glm53-dspine/data/prompts/train_prompts.jsonl")
    ap.add_argument("--runner-commit", default="drs-worker-v1.0.0")
    args = ap.parse_args()

    admin_key = Path(args.admin_key_file).read_text(encoding="utf-8").strip()
    cli = DraftSiteClient(args.url, admin_key)

    # 1. Register runner
    cli.admin_post(
        "/api/admin/runners",
        {
            "commit_id": args.runner_commit,
            "kind": "all",
            "note": "GLM-5.3 DSpine community drafter pilot worker (labels, fused SparseLoCo train, MXFP8 eval, 4-tap INT8 capture)",
        },
    )

    # 2. Stage & register permissive prompt shards
    staged = stage_prompt_shards(Path(args.prompts_src), Path(args.data_dir) / "inputs")
    label_slots = []
    for s in staged:
        art = cli.admin_post(
            "/api/admin/artifacts/register",
            {
                "artifact_id": f"inp_prompts_{s['shard']}",
                "rel_path": s["rel_path"],
                "kind": "input",
                "content_type": "application/x-ndjson",
                "job": "glm53-dspine-labels",
                "slot_id": s["shard"],
            },
        )
        label_slots.append(
            {
                "slot_id": s["shard"],
                "row_key": "prompt_pack_v1",
                "col_key": s["shard"],
                "round_num": 1,
                "profile_id": "any-glm53",
                "state": "open",
                "inputs": [
                    {
                        "name": "prompt_shard",
                        "artifact_id": art["artifact_id"],
                        "filename": Path(s["rel_path"]).name,
                        "sha256": art["sha256"],
                        "bytes": art["bytes"],
                    }
                ],
                "plan": {
                    "n_prompts": s["n_prompts"],
                    "max_tokens": s["max_tokens"],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "min_top1_agree": 0.92,
                    "min_top5_agree": 0.995,
                    "license": "Apache-2.0/MIT (open-perfectblend permissive slice)",
                },
            }
        )

    # 3. Job 1: drafter-label
    cli.admin_post(
        "/api/admin/jobs",
        {
            "job": "glm53-dspine-labels",
            "title": "GLM-5.3 DSpine · Same-Model Greedy Labels",
            "job_kind": "drafter-label",
            "model": "GLM-5.3",
            "release_repo": RELEASE_REPO,
            "release_revision": RELEASE_REV,
            "config": {"prompt_pack": "open-perfectblend-permissive", "verification": "schema+prompt_sha256+semantic_top1>=92%"},
            "slots": label_slots,
        },
    )

    # 4. Job 2: drafter-train (Fused Capture + SparseLoCo/Soup, 4 non-IID active profiles + 1 readiness-gated slot!)
    train_slots = []
    for idx, (prof, seed) in enumerate([("bit200", 101), ("bit240", 102), ("bit280", 103), ("bit322+bit400", 104)]):
        train_slots.append(
            {
                "slot_id": f"r1_w{idx}_{prof.replace('+', '_')}",
                "row_key": "round_1",
                "col_key": prof,
                "round_num": 1,
                "profile_id": prof,
                "state": "open",
                "plan": {
                    "round_num": 1,
                    "profile_id": prof,
                    "base_ckpt": "dflash2-apache2-warmstart",
                    "codec": "sparseloco_int8_topk10",
                    "keep_frac": 0.10,
                    "local_steps": 30,
                    "grad_accum": 4,
                    "lr": 2e-4,
                    "bridge_lr_mult": 5.0,
                    "l2sp_weight": 2e-4,
                    "tap_mode": "int8_6tap",
                    "label_mode": "tf_relabel_bit280",
                    "max_local_shards": 250,
                    "seed": seed,
                },
            }
        )
    # Readiness-gated slot: automatically opens ONLY when glm53-dspine-labels/shard_0000 passes verification!
    train_slots.append(
        {
            "slot_id": "r1_w4_gated_shard0",
            "row_key": "round_1",
            "col_key": "bit280_gated",
            "round_num": 1,
            "profile_id": "bit280",
            "state": "not_ready",
            "depends_on_job": "glm53-dspine-labels",
            "depends_on_slot": "shard_0000",
            "plan": {
                "round_num": 1,
                "profile_id": "bit280",
                "base_ckpt": "dflash2-apache2-warmstart",
                "codec": "stage1_bridge_int8",
                "local_steps": 15,
                "grad_accum": 4,
                "lr": 2e-4,
                "bridge_lr_mult": 5.0,
                "l2sp_weight": 2e-4,
                "tap_mode": "int8_6tap",
                "label_mode": "tf_relabel_bit280",
                "max_local_shards": 120,
                "seed": 105,
            },
        }
    )
    cli.admin_post(
        "/api/admin/jobs",
        {
            "job": "glm53-dspine-train",
            "title": "GLM-5.3 DSpine · Fused SparseLoCo / Soup Training",
            "job_kind": "drafter-train",
            "model": "GLM-5.3",
            "release_repo": RELEASE_REPO,
            "release_revision": RELEASE_REV,
            "active_round": 1,
            "config": {
                "codec": "sparseloco_int8_topk10 (~548 MB/slot, 0 B tap upload)",
                "gate": "Owner Held-Out Replay (d_ce8 < -0.005) + FLTrust Cosine (> 0.05) + L2 Norm [0.3x, 2.5x]",
                "warmstart": "incoai/GLM-5.3-DFlash2@8f02251d (Apache-2.0) + deterministic PCA CodeBridge",
            },
            "rounds": [
                {
                    "round_num": 1,
                    "base_ckpt_name": "dflash2-apache2-warmstart",
                    "quorum": 4,
                    "total_slots": 4,
                    "status": "open",
                }
            ],
            "slots": train_slots,
        },
    )

    # 5. Job 3: drafter-eval (11 mul1 profiles across champion baseline + pilot-r1-agg + duplicate quorum slot!)
    eval_slots = []
    for ckpt_rev, is_ready_ckpt in [("v5-s2-w5-champion", True), ("pilot-r1-agg", False)]:
        for prof in ALL_11_PROFILES:
            st = "open" if (is_ready_ckpt and prof in READY_5_PROFILES) else "not_ready"
            eval_slots.append(
                {
                    "slot_id": f"{ckpt_rev}__{prof}",
                    "row_key": ckpt_rev,
                    "col_key": prof,
                    "round_num": 1 if ckpt_rev == "v5-s2-w5-champion" else 2,
                    "profile_id": prof,
                    "state": st,
                    "plan": {"checkpoint_rev": ckpt_rev, "profile_id": prof, "quant": "mxfp8", "eval_pack": "eval26"},
                }
            )
    # Add an explicit duplicate-quorum verification slot on bit280 for champion
    eval_slots.append(
        {
            "slot_id": "v5-s2-w5-champion__bit280__dup",
            "row_key": "v5-s2-w5-champion",
            "col_key": "bit280",
            "round_num": 1,
            "profile_id": "bit280",
            "state": "open",
            "plan": {"checkpoint_rev": "v5-s2-w5-champion", "profile_id": "bit280", "quant": "mxfp8", "eval_pack": "eval26", "duplicate_quorum": True},
        }
    )
    cli.admin_post(
        "/api/admin/jobs",
        {
            "job": "glm53-dspine-eval",
            "title": "GLM-5.3 DSpine · 11-Profile Offline MXFP8 Eval Matrix",
            "job_kind": "drafter-eval",
            "model": "GLM-5.3",
            "release_repo": RELEASE_REPO,
            "release_revision": RELEASE_REV,
            "config": {"quant": "mxfp8", "duplicate_quorum_tol": 0.005, "profiles": ALL_11_PROFILES},
            "slots": eval_slots,
        },
    )

    # 6. Job 4: drafter-capture (optional 4-tap L33/L47/L61/L75 INT8+zstd archive)
    cap_slots = [
        {
            "slot_id": f"cap_{prof}_ev0",
            "row_key": "eval_anchor_4tap",
            "col_key": prof,
            "round_num": 1,
            "profile_id": prof,
            "state": "open",
            "plan": {
                "profile_id": prof,
                "seq_file": "seq_ev_code_0000.pt",
                "tap_layers": [33, 47, 61, 75],
                "format": "4tap_int8_zstd",
            },
        }
        for prof in ("bit200", "bit280", "bit400")
    ]
    cli.admin_post(
        "/api/admin/jobs",
        {
            "job": "glm53-dspine-capture",
            "title": "GLM-5.3 DSpine · Optional 4-Tap INT8 Archive Capture",
            "job_kind": "drafter-capture",
            "model": "GLM-5.3",
            "release_repo": RELEASE_REPO,
            "release_revision": RELEASE_REV,
            "config": {"format": "4tap_int8_zstd (L33/L47/L61/L75, ~3.8x smaller, 0.00% tau loss)"},
            "slots": cap_slots,
        },
    )
    print("Seeded all 4 GLM-5.3 DSpine pilot jobs cleanly!")


if __name__ == "__main__":
    main()
