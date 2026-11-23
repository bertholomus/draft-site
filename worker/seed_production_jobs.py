#!/usr/bin/env python3
"""Seed production jobs & owner-fleet neutral worker accounts on draft.bertholomus.com.

Fixes S12:
- Captures the server-issued `api_key_shown_once` from POST /api/admin/contributors instead
  of writing an ignored locally-generated key.
- Writes one mode-0600 key file per login under --keys-dir (never a shared multi-user JSON file
  unless --legacy-json-file is explicitly requested).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import DraftSiteClient  # noqa: E402

RELEASE_REPO = "kindlingai/glm-5.3-exlr8-k2-k3-k4"
RELEASE_REV = "9e25c28a2e8808d8b2599d780521fd6af24e7558"
PRIORITY_CATS = ["prose", "longctx_code", "tool_call", "multilingual", "fim_code", "code", "structured"]


def register_fleet_workers(cli: DraftSiteClient, keys_dir: Path) -> None:
    keys_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(keys_dir, 0o700)
    fleet_specs = [
        ("bertholomus-w1", ["tier-label", "tier-pack", "tier-cpu"], "owner-worker-1"),
        ("bertholomus-w2", ["tier-label", "tier-eval", "tier-cpu"], "owner-worker-2"),
        ("bertholomus-w3", ["tier-train", "tier-eval", "tier-capture"], "owner-worker-3"),
        ("bertholomus-w4", ["tier-train", "tier-eval", "tier-capture", "tier-label"], "owner-worker-4"),
        ("bertholomus-w5", ["tier-train", "tier-eval", "tier-capture", "tier-label"], "owner-worker-5"),
        ("bertholomus-w6", ["tier-train", "tier-eval", "tier-capture", "tier-label"], "owner-worker-6"),
        ("bertholomus-w7", ["tier-train", "tier-eval", "tier-capture", "tier-label"], "owner-worker-7"),
        ("bertholomus-w8", ["tier-train", "tier-eval", "tier-label"], "owner-worker-8"),
    ]
    for login, tiers, note in fleet_specs:
        resp = cli.admin_post(
            "/api/admin/contributors",
            {"login": login, "is_owner": True, "tiers": tiers, "note": note},
        )
        shown = resp.get("api_key_shown_once")
        if shown:
            kf = keys_dir / f"{login}.key"
            tmp = keys_dir / f".{login}.key.tmp"
            tmp.write_text(shown + "\n", encoding="utf-8")
            os.chmod(tmp, 0o600)
            os.replace(tmp, kf)


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

    ordered: list[dict] = []
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
    ap.add_argument("--url", default="http://127.0.0.1:8742")
    ap.add_argument("--admin-key-file", required=True)
    ap.add_argument("--origin-auth-file", default=None)
    ap.add_argument("--keys-dir", required=True, help="Directory for per-login mode-0600 key files")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dspine-dir", required=True)
    ap.add_argument("--shard-size", type=int, default=200)
    args = ap.parse_args()

    admin_key = Path(args.admin_key_file).read_text(encoding="utf-8").strip()
    origin_auth = Path(args.origin_auth_file).read_text(encoding="utf-8").strip() if args.origin_auth_file and Path(args.origin_auth_file).is_file() else None
    cli = DraftSiteClient(args.url, admin_key, origin_auth=origin_auth)

    register_fleet_workers(cli, Path(args.keys_dir))
    print("Registered neutral owner fleet accounts bertholomus-w1..w8")

    cli.admin_post(
        "/api/admin/runners",
        {
            "commit_id": "drs-label-ext-v1.0.0",
            "kind": "drafter-label",
            "note": "Clean standalone external drafter-label worker (/v1/completions prompt_ids, fail-closed token IDs, per-row provenance)",
        },
    )
    cli.admin_post(
        "/api/admin/runners",
        {
            "commit_id": "drs-worker-v1.1.0",
            "kind": "all",
            "note": "Owner fleet worker (labels, train, eval, capture, pack)",
        },
    )

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
                    "recommended_concurrency": 4,
                    "categories": s["cats"],
                    "gate": "owner_teacher_forced_8x10_spot_rescore",
                    "license": "permissive-v2 (Apache-2.0/MIT/PSF)",
                },
            }
        )

    for c0 in range(0, len(label_slots), 100):
        cli.admin_post(
            "/api/admin/jobs",
            {
                "job": "glm53-dspine-labels-prod",
                "title": "GLM-5.3 DSpine R2 · Bulk Labels (77k Remaining Prompts)",
                "job_kind": "drafter-label",
                "model": "GLM-5.3",
                "release_repo": RELEASE_REPO,
                "release_revision": RELEASE_REV,
                "active_round": 1,
                "config": {
                    "prompt_pack": "train_prompts_diverse_v2.jsonl (permissive Apache-2.0/MIT/PSF)",
                    "slot_size": f"{args.shard_size} prompts/slot",
                    "verification": "Phase-A schema/SHA256 + Phase-B owner HMAC-sampled 8x10 teacher-forced spot gate (>=85% top-1, >=96% top-5, >=50% per-prompt)",
                },
                "slots": label_slots[c0 : c0 + 100],
            },
        )
    print(f"Seeded glm53-dspine-labels-prod with {len(label_slots)} production slots!")


if __name__ == "__main__":
    main()
