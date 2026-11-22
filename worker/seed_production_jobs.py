#!/usr/bin/env python3
"""Seed real DSpine-sized production jobs & full-fleet neutral worker accounts on https://draft.bertholomus.com.

Coordinates with /tank/projects/glm53-dspine/data/labels/train_labels.jsonl so zero prompts are duplicated:
- glm53-dspine-labels-prod : ALL remaining prompts from train_prompts_diverse_v2.jsonl (200 prompts/slot),
                             prioritizing prose, longctx_code, tool_call, multilingual, fim_code, code, structured,
                             taken from the TAIL backward so DSpine's upward gen_lane_labels_polite.py never overlaps.
- glm53-dspine-train-r2    : 4-worker parallel GB10 multi-profile SparseLoCo/Soup round (w4..w7 simultaneously)
                             warm-started from R1 champion v5-s2-w5@500 across bit200/bit240/bit280+lc/bit322+400.
- glm53-dspine-capture-v2  : Wave-7 OUR-exlr8 captures (+1000 shards each on bit200/240/322/400 + bit280 + 6 uncaptured profiles).
- glm53-dspine-pack        : Deterministic CPU g6load dspine --nr 4 TP4 rank packing (0 GPU, bit-exact SHA-256).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import DraftSiteClient  # noqa: E402

RELEASE_REPO = "kindlingai/glm-5.3-exlr8-k2-k3-k4"
RELEASE_REV = "9e25c28a2e8808d8b2599d780521fd6af24e7558"
PRIORITY_CATS = ["prose", "longctx_code", "tool_call", "multilingual", "fim_code", "code", "structured"]


def register_fleet_workers(cli: DraftSiteClient, keys_path: Path) -> dict[str, str]:
    existing = json.loads(keys_path.read_text(encoding="utf-8")) if keys_path.is_file() else {}
    fleet_specs = [
        ("bertholomus-w1", ["tier-label", "tier-pack", "tier-cpu"], "private-host: host-1 (NAS CPU: C8 labels, g6load pack, harvester)"),
        ("bertholomus-w2", ["tier-label", "tier-eval", "tier-cpu"], "private-host: host-2 (macOS workstation CPU: labels & QA)"),
        ("bertholomus-w3", ["tier-train", "tier-eval", "tier-capture"], "private-host: host-3 (96GB Blackwell RTX PRO 6000 GPU)"),
        ("bertholomus-w4", ["tier-train", "tier-eval", "tier-capture", "tier-label"], "private-host: host-4 (128GB GB10 GPU #1 / ascent-05)"),
        ("bertholomus-w5", ["tier-train", "tier-eval", "tier-capture", "tier-label"], "private-host: host-5 (128GB GB10 GPU #2 / ascent-06)"),
        ("bertholomus-w6", ["tier-train", "tier-eval", "tier-capture", "tier-label"], "private-host: host-6 (128GB GB10 GPU #3 / ascent-07)"),
        ("bertholomus-w7", ["tier-train", "tier-eval", "tier-capture", "tier-label"], "private-host: host-7 (128GB GB10 GPU #4 / ascent-08)"),
        ("bertholomus-w8", ["tier-train", "tier-eval", "tier-label"], "private-host: host-8 (4xGB10 lane-window rig / A1-A4)"),
    ]
    changed = False
    for login, tiers, note in fleet_specs:
        raw_key = existing.get(login)
        if not raw_key:
            raw_key = "drs_wrk_" + secrets.token_hex(24)
            existing[login] = raw_key
            changed = True
            cli.admin_post(
                "/api/admin/contributors",
                {"login": login, "api_key": raw_key, "approved": True, "quarantined": False, "tiers": tiers, "note": note},
            )
        else:
            cli.admin_post(
                "/api/admin/contributors",
                {"login": login, "approved": True, "quarantined": False, "tiers": tiers, "note": note},
            )
    if changed:
        tmp = keys_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        os.chmod(tmp, 0o600)
        os.replace(tmp, keys_path)
    return existing


def stage_v2_label_shards(prompts_v2: Path, train_labels: Path, inputs_dir: Path, shard_size: int = 200) -> list[dict]:
    done_ids: set[str] = set()
    if train_labels.is_file():
        with train_labels.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_ids.add(str(json.loads(line)["id"]))
                except Exception:
                    pass

    by_cat: dict[str, list[dict]] = {c: [] for c in PRIORITY_CATS}
    other: list[dict] = []
    with prompts_v2.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if str(r["id"]) in done_ids:
                continue
            cat = str(r.get("category", "other"))
            if cat in by_cat:
                by_cat[cat].append(r)
            else:
                other.append(r)

    # Reverse each category so we claim from the TAIL backward -> 0% overlap with DSpine's upward script!
    ordered: list[dict] = []
    # Interleave high-priority categories requested by DSpine (prose, longctx_code, tool_call, multilingual, fim_code)
    hi_cats = ["prose", "longctx_code", "tool_call", "multilingual", "fim_code"]
    hi_iters = {c: list(reversed(by_cat[c])) for c in hi_cats}
    idx = {c: 0 for c in hi_cats}
    weights = {"prose": 4, "longctx_code": 2, "tool_call": 2, "multilingual": 1, "fim_code": 1}
    while any(idx[c] < len(hi_iters[c]) for c in hi_cats):
        for c in hi_cats:
            w = weights[c]
            arr = hi_iters[c]
            i = idx[c]
            if i < len(arr):
                ordered.extend(arr[i : i + w])
                idx[c] = min(len(arr), i + w)
    # Then remaining code & structured from tail backward
    for c in ("code", "structured"):
        ordered.extend(reversed(by_cat[c]))
    ordered.extend(reversed(other))

    out_dir = inputs_dir / "v2_labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for s_idx in range(0, len(ordered), shard_size):
        chunk_raw = ordered[s_idx : s_idx + shard_size]
        if not chunk_raw:
            continue
        sh_num = s_idx // shard_size
        sid = f"v2_s{sh_num:04d}"
        rel_path = f"inputs/v2_labels/prompts_{sid}.jsonl"
        out_f = inputs_dir.parent / rel_path
        lines = []
        cats_here: dict[str, int] = {}
        for r in chunk_raw:
            pids = r["prompt_ids"]
            psha = hashlib.sha256(json.dumps(pids, separators=(",", ":")).encode("utf-8")).hexdigest()
            cat = str(r.get("category", "code"))
            cats_here[cat] = cats_here.get(cat, 0) + 1
            rec = {
                "id": str(r["id"]),
                "category": cat,
                "source": str(r.get("source", "permissive")),
                "prompt_len": len(pids),
                "prompt_ids": pids,
                "prompt_sha256": psha,
                "max_tokens": 1024 if len(pids) > 4096 else 1536,
            }
            lines.append(json.dumps(rec, separators=(",", ":")))
        out_f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        dom_cat = max(cats_here.items(), key=lambda kv: kv[1])[0]
        staged.append({"slot_id": sid, "rel_path": rel_path, "n_prompts": len(lines), "dom_cat": dom_cat, "cats": cats_here})
    print(f"Staged {len(staged)} production label shards ({len(ordered)} remaining prompts, skipped {len(done_ids)} already labeled)")
    return staged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://draft.bertholomus.com")
    ap.add_argument("--admin-key-file", default="/tank/projects/draft-site/secrets/admin_api_key.txt")
    ap.add_argument("--keys-file", default="/tank/projects/draft-site/secrets/pilot_worker_keys.json")
    ap.add_argument("--data-dir", default="/tank/projects/draft-site/data")
    ap.add_argument("--dspine-dir", default="/tank/projects/glm53-dspine")
    ap.add_argument("--shard-size", type=int, default=200)
    args = ap.parse_args()

    admin_key = Path(args.admin_key_file).read_text(encoding="utf-8").strip()
    cli = DraftSiteClient(args.url, admin_key)

    # 1. Ensure all 8 fleet worker accounts exist with neutral public logins
    register_fleet_workers(cli, Path(args.keys_file))
    print("Registered neutral fleet accounts bertholomus-w1..w8")

    # 2. Register v1.1 persistent fleet runner
    cli.admin_post(
        "/api/admin/runners",
        {
            "commit_id": "drs-worker-v1.1.0",
            "kind": "all",
            "note": "Persistent parallel fleet worker (C8 token-ID labels, 4xGB10 parallel SparseLoCo train, MXFP8 eval, Wave-7 capture, CPU g6load pack)",
        },
    )

    # 3. Stage & seed glm53-dspine-labels-prod (ALL remaining prompts of 95k DSpine target!)
    data_dir = Path(args.data_dir)
    dspine_dir = Path(args.dspine_dir)
    staged = stage_v2_label_shards(
        dspine_dir / "data/prompts/train_prompts_diverse_v2.jsonl",
        dspine_dir / "data/labels/train_labels.jsonl",
        data_dir / "inputs",
        shard_size=args.shard_size,
    )

    label_slots = []
    for idx, s in enumerate(staged):
        art_id = f"inp_v2_{s['slot_id']}"
        art = cli.admin_post(
            "/api/admin/artifacts/register",
            {
                "artifact_id": art_id,
                "rel_path": s["rel_path"],
                "kind": "input",
                "content_type": "application/x-ndjson",
                "job": "glm53-dspine-labels-prod",
                "slot_id": s["slot_id"],
            },
        )
        row_bucket = f"batch_{idx // 50:02d}"
        label_slots.append(
            {
                "slot_id": s["slot_id"],
                "row_key": row_bucket,
                "col_key": s["dom_cat"],
                "round_num": 1,
                "profile_id": "any-glm53-c8",
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
                    "max_tokens_short": 1536,
                    "max_tokens_long": 1024,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "recommended_concurrency": 8,
                    "categories": s["cats"],
                    "min_top1_agree": 0.92,
                    "min_top5_agree": 0.995,
                    "license": "permissive-v2 (Apache-2.0/MIT/PSF)",
                },
            }
        )
        if (idx + 1) % 100 == 0:
            print(f"  registered {idx + 1}/{len(staged)} prompt shard inputs...")

    # Register job in chunks of 100 slots to stay well under HTTP body limits
    for c0 in range(0, len(label_slots), 100):
        cli.admin_post(
            "/api/admin/jobs",
            {
                "job": "glm53-dspine-labels-prod",
                "title": "GLM-5.3 DSpine R2 · Bulk C8 Labels (77k Remaining Prompts)",
                "job_kind": "drafter-label",
                "model": "GLM-5.3",
                "release_repo": RELEASE_REPO,
                "release_revision": RELEASE_REV,
                "active_round": 1,
                "config": {
                    "prompt_pack": "train_prompts_diverse_v2.jsonl (95k target, tail-first zero-overlap with DSpine)",
                    "slot_size": f"{args.shard_size} prompts/slot, C=8 serving concurrency",
                    "priority_order": "prose > longctx_code > tool_call > multilingual > fim_code > code > structured",
                    "export_path": "data/exports/drs_labels.jsonl",
                },
                "slots": label_slots[c0 : c0 + 100],
            },
        )
    print(f"Seeded glm53-dspine-labels-prod with {len(label_slots)} production slots!")

    # 4. Seed glm53-dspine-train-r2 (4 independent GB10 workers running in parallel from R1 champion v5-s2-w5@500!)
    r2_train_slots = [
        {
            "slot_id": "r2_w0_bit200",
            "row_key": "round_1",
            "col_key": "bit200",
            "round_num": 1,
            "profile_id": "bit200",
            "state": "open",
            "plan": {
                "round_num": 1, "profile_id": "bit200", "base_ckpt": "v5-s2-w5-champion",
                "codec": "sparseloco_int8_topk10", "local_steps": 40, "grad_accum": 4,
                "lr": 2.0e-5, "bridge_lr_mult": 5.0, "l2sp_weight": 2.0e-4,
                "tap_mode": "int8_6tap", "label_mode": "tf_relabel_bit280",
                "shard_dirs": ["/var/tmp/dspine-shards-bit200/train"], "max_local_shards": 1000, "seed": 201,
            },
        },
        {
            "slot_id": "r2_w1_bit240",
            "row_key": "round_1",
            "col_key": "bit240",
            "round_num": 1,
            "profile_id": "bit240",
            "state": "open",
            "plan": {
                "round_num": 1, "profile_id": "bit240", "base_ckpt": "v5-s2-w5-champion",
                "codec": "sparseloco_int8_topk10", "local_steps": 40, "grad_accum": 4,
                "lr": 2.0e-5, "bridge_lr_mult": 5.0, "l2sp_weight": 2.0e-4,
                "tap_mode": "int8_6tap", "label_mode": "tf_relabel_bit280",
                "shard_dirs": ["/var/tmp/dspine-shards-bit240/train"], "max_local_shards": 1000, "seed": 202,
            },
        },
        {
            "slot_id": "r2_w2_bit280_lc",
            "row_key": "round_1",
            "col_key": "bit280+lc",
            "round_num": 1,
            "profile_id": "bit280+lc",
            "state": "open",
            "plan": {
                "round_num": 1, "profile_id": "bit280+lc", "base_ckpt": "v5-s2-w5-champion",
                "codec": "sparseloco_int8_topk10", "local_steps": 40, "grad_accum": 4,
                "lr": 2.0e-5, "bridge_lr_mult": 5.0, "l2sp_weight": 2.0e-4,
                "tap_mode": "int8_6tap", "label_mode": "tf_relabel_bit280",
                "shard_dirs": ["/var/tmp/dspine-shards-bit280/train", "/var/tmp/dspine-shards-bit280-lc/train"],
                "max_local_shards": 2000, "seed": 203,
            },
        },
        {
            "slot_id": "r2_w3_bit322_400",
            "row_key": "round_1",
            "col_key": "bit322+bit400",
            "round_num": 1,
            "profile_id": "bit322+bit400",
            "state": "open",
            "plan": {
                "round_num": 1, "profile_id": "bit322+bit400", "base_ckpt": "v5-s2-w5-champion",
                "codec": "sparseloco_int8_topk10", "local_steps": 40, "grad_accum": 4,
                "lr": 2.0e-5, "bridge_lr_mult": 5.0, "l2sp_weight": 2.0e-4,
                "tap_mode": "int8_6tap", "label_mode": "tf_relabel_bit280",
                "shard_dirs": ["/var/tmp/dspine-shards-bit322/train", "/var/tmp/dspine-shards-bit400/train"],
                "max_local_shards": 2000, "seed": 204,
            },
        },
    ]
    cli.admin_post(
        "/api/admin/jobs",
        {
            "job": "glm53-dspine-train-r2",
            "title": "GLM-5.3 DSpine R2 · 4-Worker Parallel GB10 SparseLoCo",
            "job_kind": "drafter-train",
            "model": "GLM-5.3",
            "release_repo": RELEASE_REPO,
            "release_revision": RELEASE_REV,
            "active_round": 1,
            "config": {
                "topology": "4 independent 128GB GB10 workers in parallel (bit200 / bit240 / bit280+lc / bit322+400)",
                "warmstart": "v5-s2-w5-champion (R1 live champion) + Wave-6 17,246 OUR-exlr8 shards",
                "codec": "sparseloco_int8_topk10 (~548 MB/slot, 0 B tap upload)",
                "gate": "Owner Held-Out Replay + FLTrust Cosine + L2 Norm Band",
            },
            "rounds": [
                {
                    "round_num": 1,
                    "base_ckpt_name": "v5-s2-w5-champion",
                    "quorum": 4,
                    "total_slots": 4,
                    "status": "open",
                }
            ],
            "slots": r2_train_slots,
        },
    )
    print("Seeded glm53-dspine-train-r2 (4 parallel GB10 SparseLoCo slots)!")

    # 5. Add r2-4gb10-agg + r2-v6-ema rows to glm53-dspine-eval
    eval_new_slots = []
    for ckpt_rev, rnd in [("r2-4gb10-agg", 3), ("r2-v6-ema", 4)]:
        for prof in ("bit200", "bit240", "bit280", "bit322", "bit400"):
            eval_new_slots.append(
                {
                    "slot_id": f"{ckpt_rev}__{prof}",
                    "row_key": ckpt_rev,
                    "col_key": prof,
                    "round_num": rnd,
                    "profile_id": prof,
                    "state": "not_ready",
                    "plan": {"checkpoint_rev": ckpt_rev, "profile_id": prof, "quant": "mxfp8", "eval_pack": "eval26"},
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
            "active_round": 3,
            "config": {
                "quant": "mxfp8",
                "duplicate_quorum_tol": 0.005,
                "profiles": ["bit200", "bit220", "bit240", "bit250", "bit271", "bit280", "bit300", "bit322", "bit350", "bit375", "bit400"],
            },
            "slots": eval_new_slots,
        },
    )
    print("Added r2-4gb10-agg and r2-v6-ema rows to glm53-dspine-eval!")

    # 6. Seed glm53-dspine-capture-v2 (Wave-7 OUR-exlr8 captures requested by DSpine R2)
    cap_v2_slots = []
    for prof, n_target in [("bit200", 1000), ("bit240", 1000), ("bit280", 2000), ("bit280_lc", 1000), ("bit322", 1000), ("bit400", 1000)]:
        cap_v2_slots.append(
            {
                "slot_id": f"w7_{prof}_batch0",
                "row_key": "wave7_train_exlr8",
                "col_key": prof,
                "round_num": 1,
                "profile_id": prof,
                "state": "open",
                "plan": {
                    "profile_id": prof,
                    "target_new_shards": n_target,
                    "tap_layers": [33, 47, 61, 75],
                    "format": "4tap_int8_zstd",
                    "release": f"{RELEASE_REPO}@{RELEASE_REV[:8]}",
                },
            }
        )
    for prof in ("bit220", "bit250", "bit271", "bit300", "bit350", "bit375"):
        cap_v2_slots.append(
            {
                "slot_id": f"eval26_{prof}",
                "row_key": "matrix11_eval26",
                "col_key": prof,
                "round_num": 1,
                "profile_id": prof,
                "state": "open",
                "plan": {
                    "profile_id": prof,
                    "target_new_shards": 26,
                    "tap_layers": [33, 47, 61, 75],
                    "format": "4tap_int8_zstd",
                    "release": f"{RELEASE_REPO}@{RELEASE_REV[:8]}",
                },
            }
        )
    cli.admin_post(
        "/api/admin/jobs",
        {
            "job": "glm53-dspine-capture-v2",
            "title": "GLM-5.3 DSpine R2 · Wave-7 Multi-Profile & 11-Profile Eval Capture",
            "job_kind": "drafter-capture",
            "model": "GLM-5.3",
            "release_repo": RELEASE_REPO,
            "release_revision": RELEASE_REV,
            "active_round": 1,
            "config": {
                "format": "4tap_int8_zstd (L33/L47/L61/L75)",
                "requested_by": "DSpine Round 2 (+1k/profile bit200/240/322/400 + bit280 short/lc + 6 missing 11-profile eval26 anchors)",
            },
            "slots": cap_v2_slots,
        },
    )
    print("Seeded glm53-dspine-capture-v2 (12 Wave-7 & 11-profile slots)!")

    # 7. Seed glm53-dspine-pack (CPU deterministic g6load dspine --nr 4)
    pack_slots = [
        {
            "slot_id": "pilot-r1-agg__tp4",
            "row_key": "pilot-r1-agg",
            "col_key": "tp4",
            "round_num": 1,
            "profile_id": "tp4-mxfp8",
            "state": "open",
            "plan": {"checkpoint_rev": "pilot-r1-agg", "nr": 4, "format_version": "g6dspine-v1"},
        },
        {
            "slot_id": "v5-s2-w5-champion__tp4",
            "row_key": "v5-s2-w5-champion",
            "col_key": "tp4",
            "round_num": 1,
            "profile_id": "tp4-mxfp8",
            "state": "open",
            "plan": {"checkpoint_rev": "v5-s2-w5-champion", "nr": 4, "format_version": "g6dspine-v1"},
        },
        {
            "slot_id": "r2-4gb10-agg__tp4",
            "row_key": "r2-4gb10-agg",
            "col_key": "tp4",
            "round_num": 2,
            "profile_id": "tp4-mxfp8",
            "state": "not_ready",
            "plan": {"checkpoint_rev": "r2-4gb10-agg", "nr": 4, "format_version": "g6dspine-v1"},
        },
    ]
    cli.admin_post(
        "/api/admin/jobs",
        {
            "job": "glm53-dspine-pack",
            "title": "GLM-5.3 DSpine · Deterministic CPU TP4 Packing (g6load dspine)",
            "job_kind": "drafter-pack",
            "model": "GLM-5.3",
            "release_repo": RELEASE_REPO,
            "release_revision": RELEASE_REV,
            "active_round": 1,
            "config": {"tool": "g6load dspine --nr 4", "device": "CPU (0 GPU, ~37s, 100% bit-exact SHA-256)"},
            "slots": pack_slots,
        },
    )
    print("Seeded glm53-dspine-pack (CPU deterministic TP4 packing slots)!")


if __name__ == "__main__":
    main()
