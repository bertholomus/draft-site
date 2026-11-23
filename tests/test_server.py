"""Comprehensive pytest suite covering S1, S4-S10, S12, S14, and Fable 4.8."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import save as st_save_bytes


def _make_prompt_shard(data_dir: Path, filename: str, n: int = 4) -> tuple[list[dict[str, Any]], str]:
    (data_dir / "inputs").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        pids = [154822, 154824, 100 + i, 200 + i, 300 + i]
        psha = hashlib.sha256(json.dumps(pids, separators=(",", ":")).encode("utf-8")).hexdigest()
        rows.append(
            {
                "id": f"p{i}",
                "category": "code",
                "source": "permissive",
                "prompt_len": len(pids),
                "prompt_ids": pids,
                "prompt_sha256": psha,
                "max_tokens": 64,
            }
        )
    rel = f"inputs/{filename}"
    (data_dir / rel).write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n", encoding="utf-8")
    return rows, rel


def _make_label_gz(
    prompt_rows: list[dict[str, Any]],
    *,
    gen_ids_fn=None,
    finish_reason: str = "stop",
    override_prompt_sha: str | None = None,
    include_provenance: bool = True,
) -> bytes:
    lines = []
    for i, pr in enumerate(prompt_rows):
        gids = gen_ids_fn(i, pr) if gen_ids_fn else [10 + i, 20 + i, 30 + i, 40 + i, 50 + i]
        gsha = hashlib.sha256(json.dumps(gids, separators=(",", ":")).encode("utf-8")).hexdigest()
        rec: dict[str, Any] = {
            "id": pr["id"],
            "prompt_sha256": override_prompt_sha if override_prompt_sha is not None else pr["prompt_sha256"],
            "gen_ids": gids,
            "finish_reason": finish_reason,
            "engine_tag": "ext-gr8-bit280",
            "gen_sha256": gsha,
        }
        if include_provenance:
            rec.update(
                {
                    "engine_image_digest": "sha256:" + "a" * 64,
                    "engine_signature": "gr8-test-sig",
                    "quant_profile": "bit280",
                    "concurrency": 4,
                    "token_id_source": "engine",
                }
            )
        lines.append(json.dumps(rec, separators=(",", ":")))
    return gzip.compress(("\n".join(lines) + "\n").encode("utf-8"))


def _seed_all_kinds(site_env: dict[str, Any]) -> dict[str, Any]:
    client = site_env["client"]
    adm = site_env["adm_hdr"]
    data_dir = site_env["data_dir"]

    assert client.post("/api/admin/runners", headers=adm, json={"commit_id": "runner-all", "kind": "all"}).status_code == 200
    assert client.post("/api/admin/runners", headers=adm, json={"commit_id": "drs-label-ext-v1.0.0", "kind": "drafter-label"}).status_code == 200

    p_rows0, rel0 = _make_prompt_shard(data_dir, "prompts_s0.jsonl", n=4)
    p_rows1, rel1 = _make_prompt_shard(data_dir, "prompts_s1.jsonl", n=4)
    p_rows2, rel2 = _make_prompt_shard(data_dir, "prompts_s2.jsonl", n=4)
    a0 = client.post("/api/admin/artifacts/register", headers=adm, json={"artifact_id": "inp_p0", "rel_path": rel0}).json()
    a1 = client.post("/api/admin/artifacts/register", headers=adm, json={"artifact_id": "inp_p1", "rel_path": rel1}).json()
    a2 = client.post("/api/admin/artifacts/register", headers=adm, json={"artifact_id": "inp_p2", "rel_path": rel2}).json()

    assert client.post(
        "/api/admin/jobs",
        headers=adm,
        json={
            "job": "labels-prod",
            "title": "Labels Prod",
            "job_kind": "drafter-label",
            "slots": [
                {
                    "slot_id": "s0", "row_key": "b0", "col_key": "code", "profile_id": "bit280", "state": "open",
                    "inputs": [{"name": "prompt_shard", "artifact_id": "inp_p0", "sha256": a0["sha256"], "bytes": a0["bytes"]}],
                    "plan": {"n_prompts": 4, "max_tokens_short": 64},
                },
                {
                    "slot_id": "s1", "row_key": "b0", "col_key": "code", "profile_id": "bit280", "state": "open",
                    "inputs": [{"name": "prompt_shard", "artifact_id": "inp_p1", "sha256": a1["sha256"], "bytes": a1["bytes"]}],
                    "plan": {"n_prompts": 4, "max_tokens_short": 64},
                },
                {
                    "slot_id": "s2", "row_key": "b0", "col_key": "code", "profile_id": "bit280", "state": "open",
                    "inputs": [{"name": "prompt_shard", "artifact_id": "inp_p2", "sha256": a2["sha256"], "bytes": a2["bytes"]}],
                    "plan": {"n_prompts": 4, "max_tokens_short": 64},
                },
                {
                    "slot_id": "s_no_shard", "row_key": "b1", "col_key": "code", "profile_id": "bit280", "state": "open",
                    "inputs": [],
                    "plan": {"n_prompts": 4},
                },
            ],
        },
    ).status_code == 200

    assert client.post(
        "/api/admin/jobs",
        headers=adm,
        json={
            "job": "train-r1",
            "title": "Train R1",
            "job_kind": "drafter-train",
            "rounds": [{"round_num": 1, "base_ckpt_name": "theta_0", "quorum": 1, "total_slots": 1}],
            "slots": [
                {
                    "slot_id": "t_open", "row_key": "r1", "col_key": "bit280", "round_num": 1, "profile_id": "bit280",
                    "state": "open", "plan": {"codec": "sparseloco_int8_topk10", "profile_id": "bit280", "round_num": 1},
                },
                {
                    "slot_id": "t_gated", "row_key": "r1", "col_key": "gated", "round_num": 1, "profile_id": "bit280",
                    "state": "not_ready", "depends_on_job": "labels-prod", "depends_on_slot": "s0",
                    "plan": {"codec": "sparseloco_int8_topk10", "profile_id": "bit280", "round_num": 1},
                },
            ],
        },
    ).status_code == 200

    assert client.post(
        "/api/admin/jobs",
        headers=adm,
        json={
            "job": "eval-r1",
            "title": "Eval R1",
            "job_kind": "drafter-eval",
            "slots": [
                {"slot_id": "e0", "row_key": "c0", "col_key": "bit280", "profile_id": "bit280", "state": "open", "plan": {"checkpoint_rev": "c0", "profile_id": "bit280"}},
                {"slot_id": "e0_dup", "row_key": "c0", "col_key": "bit280", "profile_id": "bit280", "state": "open", "plan": {"checkpoint_rev": "c0", "profile_id": "bit280"}},
            ],
        },
    ).status_code == 200

    assert client.post(
        "/api/admin/jobs",
        headers=adm,
        json={
            "job": "cap-r1",
            "title": "Capture R1",
            "job_kind": "drafter-capture",
            "slots": [{"slot_id": "c0", "row_key": "cap", "col_key": "bit280", "profile_id": "bit280", "state": "open", "plan": {"profile_id": "bit280", "tap_layers": [33, 47, 61, 75]}}],
        },
    ).status_code == 200

    assert client.post(
        "/api/admin/jobs",
        headers=adm,
        json={
            "job": "pack-r1",
            "title": "Pack R1",
            "job_kind": "drafter-pack",
            "slots": [{"slot_id": "pk0", "row_key": "c0", "col_key": "tp4", "profile_id": "tp4", "state": "open", "plan": {"checkpoint_rev": "c0", "nr": 4}}],
        },
    ).status_code == 200

    return {"p_rows0": p_rows0, "p_rows1": p_rows1, "p_rows2": p_rows2, "a0": a0, "a1": a1, "a2": a2}


def test_origin_auth_default_deny_s4(site_env):
    client = site_env["client"]
    # Missing X-Origin-Auth MUST fail with 403 regardless of Host header
    for host in ("testserver", "draft.bertholomus.com", "evil.example", "127.0.0.1", "localhost"):
        r = client.get("/api/jobs", headers={"Host": host, "X-Quant": "1"})
        assert r.status_code == 403, f"Host={host} bypassed origin auth: {r.status_code}"
    # Wrong secret MUST fail with 403
    r_bad = client.get("/api/jobs", headers={"X-Quant": "1", "X-Origin-Auth": "wrong"})
    assert r_bad.status_code == 403
    # /healthz allowed without X-Origin-Auth
    assert client.get("/healthz").status_code == 200
    # Valid X-Origin-Auth succeeds
    assert client.get("/api/jobs", headers=site_env["pub_hdr"]).status_code == 200


def test_csp_no_inline_script_and_legal_routes_s14(site_env):
    client = site_env["client"]
    r = client.get("/", headers=site_env["origin_hdr"])
    assert r.status_code == 200
    csp = r.headers.get("content-security-policy", "")
    assert "script-src 'self'" in csp
    assert "script-src 'self' 'unsafe-inline'" not in csp
    assert '<script src="/static/app.js" defer></script>' in r.text
    assert "onclick=" not in r.text
    js = client.get("/static/app.js", headers=site_env["origin_hdr"])
    assert js.status_code == 200 and "selectJob" in js.text
    assert client.get("/LICENSE", headers=site_env["origin_hdr"]).status_code == 200
    assert client.get("/NOTICE", headers=site_env["origin_hdr"]).status_code == 200


def test_x_quant_and_rbac_boundaries(site_env):
    client = site_env["client"]
    _, ext_hdr = site_env["mint_user"]("ext-a", tiers=["tier-label"], is_owner=False)
    assert client.get("/api/jobs", headers=site_env["origin_hdr"]).status_code == 400
    assert client.post("/api/claims", headers=site_env["pub_hdr"], json={}).status_code == 401
    assert client.post("/api/admin/runners", headers=ext_hdr, json={"commit_id": "evil", "kind": "all"}).status_code == 403


def test_non_owner_label_only_and_tiers_enforced(site_env):
    client = site_env["client"]
    _seed_all_kinds(site_env)

    # Even if an admin accidentally gave a non-owner every tier, non-owner is restricted to drafter-label!
    _, ext_hdr = site_env["mint_user"](
        "ext-multi",
        tiers=["tier-label", "tier-train", "tier-eval", "tier-capture", "tier-pack"],
        is_owner=False,
    )
    for blocked_job in ("train-r1", "eval-r1", "cap-r1", "pack-r1"):
        r = client.post(
            "/api/claims",
            headers=ext_hdr,
            json={"runner": "runner-all", "job": blocked_job, "request": f"req_{blocked_job}"},
        )
        assert r.status_code == 403, f"non-owner should be blocked from {blocked_job}, got {r.status_code}: {r.text}"

    # Non-owner CAN claim drafter-label
    r_ok = client.post(
        "/api/claims",
        headers=ext_hdr,
        json={"runner": "drs-label-ext-v1.0.0", "job": "labels-prod", "slot_id": "s0", "request": "req_lbl_0"},
    )
    assert r_ok.status_code == 201, r_ok.text

    # Non-owner without tier-label is blocked from drafter-label too (tier enforcement)
    _, no_tier_hdr = site_env["mint_user"]("ext-notier", tiers=["tier-eval"], is_owner=False)
    r_nt = client.post(
        "/api/claims",
        headers=no_tier_hdr,
        json={"runner": "drs-label-ext-v1.0.0", "job": "labels-prod", "slot_id": "s1", "request": "req_nt"},
    )
    assert r_nt.status_code == 403

    # Owner with only tier-label cannot claim drafter-train (tiers enforced for owners too!)
    _, own_lbl_hdr = site_env["mint_user"]("owner-lbl-only", tiers=["tier-label"], is_owner=True)
    r_ol = client.post(
        "/api/claims",
        headers=own_lbl_hdr,
        json={"runner": "runner-all", "job": "train-r1", "slot_id": "t_open", "request": "req_ol"},
    )
    assert r_ol.status_code == 403

    # Owner with tier-train CAN claim drafter-train
    _, own_tr_hdr = site_env["mint_user"]("owner-trainer", tiers=["tier-train"], is_owner=True)
    r_ot = client.post(
        "/api/claims",
        headers=own_tr_hdr,
        json={"runner": "runner-all", "job": "train-r1", "slot_id": "t_open", "request": "req_ot"},
    )
    assert r_ot.status_code == 201, r_ot.text


def test_claim_cap_idempotency_and_release_s7(site_env):
    client = site_env["client"]
    _seed_all_kinds(site_env)
    _, ext_hdr = site_env["mint_user"]("ext-cap", tiers=["tier-label"], is_owner=False)

    c1 = client.post("/api/claims", headers=ext_hdr, json={"runner": "drs-label-ext-v1.0.0", "job": "labels-prod", "slot_id": "s0", "request": "req_1"})
    assert c1.status_code == 201
    # Idempotent retry with same request id succeeds
    c1_dup = client.post("/api/claims", headers=ext_hdr, json={"runner": "drs-label-ext-v1.0.0", "job": "labels-prod", "slot_id": "s0", "request": "req_1"})
    assert c1_dup.status_code == 201 and c1_dup.json()["claim_id"] == c1.json()["claim_id"]

    c2 = client.post("/api/claims", headers=ext_hdr, json={"runner": "drs-label-ext-v1.0.0", "job": "labels-prod", "slot_id": "s1", "request": "req_2"})
    assert c2.status_code == 201

    # 3rd active claim MUST fail with 429 (cap=2)
    c3 = client.post("/api/claims", headers=ext_hdr, json={"runner": "drs-label-ext-v1.0.0", "job": "labels-prod", "slot_id": "s2", "request": "req_3"})
    assert c3.status_code == 429, f"expected 429 on 3rd active claim, got {c3.status_code}: {c3.text}"

    # Releasing s1 allows claiming s2
    assert client.delete("/api/claims/labels-prod/s1", headers=ext_hdr).status_code == 200
    c3_ok = client.post("/api/claims", headers=ext_hdr, json={"runner": "drs-label-ext-v1.0.0", "job": "labels-prod", "slot_id": "s2", "request": "req_3b"})
    assert c3_ok.status_code == 201


def test_label_gate_pending_no_manifest_bypass_and_owner_verdict_s1(site_env):
    client = site_env["client"]
    adm = site_env["adm_hdr"]
    seeded = _seed_all_kinds(site_env)
    _, ext_hdr = site_env["mint_user"]("ext-kindling", tiers=["tier-label"], is_owner=False)

    clm = client.post(
        "/api/claims",
        headers=ext_hdr,
        json={"runner": "drs-label-ext-v1.0.0", "job": "labels-prod", "slot_id": "s0", "request": "lg1"},
    ).json()

    gz = _make_label_gz(seeded["p_rows0"])
    sha = hashlib.sha256(gz).hexdigest()
    # Attempt to bypass gate with forged semantic_top1_agree=0.999 in manifest
    up = client.post(
        f"/api/uploads/direct/{clm['claim_id']}?filename=s0.labels.jsonl.gz",
        headers={
            **ext_hdr,
            "X-Content-SHA256": sha,
            "X-Manifest-JSON": json.dumps({"semantic_top1_agree": 0.999, "semantic_top5_agree": 0.9999}),
        },
        content=gz,
    )
    assert up.status_code == 200, up.text
    body = up.json()
    assert body["accepted"] is True
    assert body["state"] == "uploaded"
    assert body["verify_status"] == "pending_label_gate"
    assert body["opened_downstream"] == []

    # Downstream train slot MUST still be not_ready!
    grid_tr = client.get("/api/grid?job=train-r1", headers=site_env["pub_hdr"]).json()
    t_gated = next(s for s in grid_tr["slots"] if s["slot_id"] == "t_gated")
    assert t_gated["state"] == "not_ready"

    # Pending verification list exposes slot + prompt_shard input metadata
    pend = client.get("/api/admin/pending-verifications", headers=adm).json()
    assert len(pend) == 1 and pend[0]["slot_id"] == "s0" and pend[0]["job_kind"] == "drafter-label"
    assert pend[0]["inputs"][0]["artifact_id"] == "inp_p0"

    # Owner gate passes it -> flips to verified and opens downstream train slot!
    ver = client.post(
        "/api/admin/verifications/labels-prod/s0",
        headers=adm,
        json={
            "passed": True,
            "verify_status": "verified_label_spot_gate",
            "reason": "PASS 78/80 top1, 80/80 top5",
            "gate_metrics": {"n_probes": 80, "top1_agree": 0.975, "top5_agree": 1.0, "min_prompt_top1": 0.9},
        },
    )
    assert ver.status_code == 200, ver.text
    assert ver.json()["state"] == "verified"
    assert ver.json()["opened_downstream"] == [{"job": "train-r1", "slot_id": "t_gated"}]

    # Trusted owner bertholomus-w1 still verifies directly without waiting on gate
    _, w1_hdr = site_env["mint_user"]("bertholomus-w1", tiers=["tier-label"], is_owner=True)
    clm_w1 = client.post(
        "/api/claims",
        headers=w1_hdr,
        json={"runner": "runner-all", "job": "labels-prod", "slot_id": "s1", "request": "w1_1"},
    ).json()
    gz1 = _make_label_gz(seeded["p_rows1"])
    up_w1 = client.post(
        f"/api/uploads/direct/{clm_w1['claim_id']}?filename=s1.labels.jsonl.gz",
        headers={**w1_hdr, "X-Content-SHA256": hashlib.sha256(gz1).hexdigest()},
        content=gz1,
    )
    assert up_w1.status_code == 200 and up_w1.json()["state"] == "verified"
    assert up_w1.json()["verify_status"] == "verified_label_trusted_owner"


def test_label_gate_reject_and_strike3_auto_quarantine(site_env):
    client = site_env["client"]
    adm = site_env["adm_hdr"]
    seeded = _seed_all_kinds(site_env)
    _, ext_hdr = site_env["mint_user"]("ext-bad", tiers=["tier-label"], is_owner=False)

    # Strike 1 & 2 via owner gate rejection, Strike 3 via Phase-A rejection -> auto-quarantine!
    for attempt in (1, 2):
        clm = client.post(
            "/api/claims",
            headers=ext_hdr,
            json={"runner": "drs-label-ext-v1.0.0", "job": "labels-prod", "slot_id": "s0", "request": f"bad_{attempt}"},
        ).json()
        gz = _make_label_gz(seeded["p_rows0"])
        up = client.post(
            f"/api/uploads/direct/{clm['claim_id']}?filename=s0.labels.jsonl.gz",
            headers={**ext_hdr, "X-Content-SHA256": hashlib.sha256(gz).hexdigest()},
            content=gz,
        )
        assert up.status_code == 200 and up.json()["state"] == "uploaded"
        rej = client.post(
            "/api/admin/verifications/labels-prod/s0",
            headers=adm,
            json={
                "passed": False,
                "verify_status": "rejected_label_spot_gate",
                "reason": "REJECT top1=0.12",
                "gate_metrics": {"top1_agree": 0.12},
                "reopen_on_reject": True,
            },
        )
        assert rej.status_code == 200 and rej.json()["state"] == "open"
        assert rej.json()["quarantined_actor"] is False

    # Strike 3 -> auto-quarantines contributor!
    clm3 = client.post(
        "/api/claims",
        headers=ext_hdr,
        json={"runner": "drs-label-ext-v1.0.0", "job": "labels-prod", "slot_id": "s0", "request": "bad_3"},
    ).json()
    gz_bad = _make_label_gz(seeded["p_rows0"], override_prompt_sha="0" * 64)
    up3 = client.post(
        f"/api/uploads/direct/{clm3['claim_id']}?filename=bad.labels.jsonl.gz",
        headers={**ext_hdr, "X-Content-SHA256": hashlib.sha256(gz_bad).hexdigest()},
        content=gz_bad,
    )
    assert up3.status_code == 422 and up3.json()["quarantined"] is True

    # Next claim MUST be blocked with 403!
    blocked = client.post(
        "/api/claims",
        headers=ext_hdr,
        json={"runner": "drs-label-ext-v1.0.0", "job": "labels-prod", "slot_id": "s0", "request": "bad_4"},
    )
    assert blocked.status_code == 403


def test_label_phase_a_structural_guards_and_fail_closed(site_env):
    client = site_env["client"]
    seeded = _seed_all_kinds(site_env)
    staging = site_env["data_dir"] / "uploads" / "staging"
    # Use owner login so strike-3 doesn't stop testing all 6 malformed payloads in one loop
    _, usr_hdr = site_env["mint_user"]("owner-qa-struct", tiers=["tier-label"], is_owner=True)

    # 1. Fail-closed when slot has no registered prompt shard
    c_ns = client.post(
        "/api/claims", headers=usr_hdr, json={"runner": "drs-label-ext-v1.0.0", "job": "labels-prod", "slot_id": "s_no_shard", "request": "ns1"}
    ).json()
    gz_ok = _make_label_gz(seeded["p_rows0"])
    r_ns = client.post(
        f"/api/uploads/direct/{c_ns['claim_id']}?filename=x.gz",
        headers={**usr_hdr, "X-Content-SHA256": hashlib.sha256(gz_ok).hexdigest()},
        content=gz_ok,
    )
    assert r_ns.status_code == 422 and r_ns.json()["verify_status"] == "failed_no_prompt_shard"

    bad_cases = [
        ("short_rows_fallback", _make_label_gz(seeded["p_rows0"], gen_ids_fn=lambda i, pr: [154829]), "failed_short_rows"),
        ("short_length_finish", _make_label_gz(seeded["p_rows0"], gen_ids_fn=lambda i, pr: [10, 20, 30, 40], finish_reason="length"), "failed_short_length_finish"),
        ("max_tokens_exceeded", _make_label_gz(seeded["p_rows0"], gen_ids_fn=lambda i, pr: [10] * 100), "failed_max_tokens_exceeded"),
        ("bool_token_ids", _make_label_gz(seeded["p_rows0"], gen_ids_fn=lambda i, pr: [True, False, 10, 20]), "failed_vocab"),
        ("invalid_utf8", gzip.compress(b"\xff\xfe\xfd"), "failed_utf8"),
    ]
    for name, payload, exp_status in bad_cases:
        c = client.post(
            "/api/claims", headers=usr_hdr, json={"runner": "drs-label-ext-v1.0.0", "job": "labels-prod", "slot_id": "s0", "request": f"st_{name}"}
        ).json()
        r = client.post(
            f"/api/uploads/direct/{c['claim_id']}?filename={name}.gz",
            headers={**usr_hdr, "X-Content-SHA256": hashlib.sha256(payload).hexdigest(), "X-Manifest-JSON": '{"semantic_top1_agree":"not-a-float"}'},
            content=payload,
        )
        assert r.status_code == 422, f"{name}: expected 422, got {r.status_code}: {r.text}"
        assert r.json()["verify_status"] == exp_status, f"{name}: got {r.json()}"
        assert list(staging.iterdir()) == [], f"leaked staged files after {name}"


def test_download_claim_binding_and_public_redaction_s5_s9(site_env):
    client = site_env["client"]
    adm = site_env["adm_hdr"]
    seeded = _seed_all_kinds(site_env)
    _, ext1_hdr = site_env["mint_user"]("ext-dl-1", tiers=["tier-label"], is_owner=False)
    _, ext2_hdr = site_env["mint_user"]("ext-dl-2", tiers=["tier-label"], is_owner=False)

    # Without holding any claim, ext1 CANNOT download inp_p0
    assert client.get("/api/artifacts/inp_p0", headers=ext1_hdr).status_code == 403

    # Claim s0 (inp_p0) -> ext1 CAN download inp_p0, CANNOT download inp_p1
    clm = client.post(
        "/api/claims", headers=ext1_hdr, json={"runner": "drs-label-ext-v1.0.0", "job": "labels-prod", "slot_id": "s0", "request": "dl1"}
    ).json()
    ok_dl = client.get("/api/artifacts/inp_p0", headers=ext1_hdr)
    assert ok_dl.status_code == 200 and ok_dl.headers["x-sha256"] == seeded["a0"]["sha256"]
    assert client.get("/api/artifacts/inp_p1", headers=ext1_hdr).status_code == 403

    # Upload s0 -> get artifact_id via admin pending list; ext2 CANNOT download ext1's uploaded artifact; admin CAN
    gz = _make_label_gz(seeded["p_rows0"])
    assert client.post(
        f"/api/uploads/direct/{clm['claim_id']}?filename=s0.gz",
        headers={**ext1_hdr, "X-Content-SHA256": hashlib.sha256(gz).hexdigest()},
        content=gz,
    ).status_code == 200
    pend = client.get("/api/admin/pending-verifications", headers=adm).json()
    up_art_id = pend[0]["artifact_id"]
    assert client.get(f"/api/artifacts/{up_art_id}", headers=ext2_hdr).status_code == 403
    assert client.get(f"/api/artifacts/{up_art_id}", headers=ext1_hdr).status_code == 403
    assert client.get(f"/api/artifacts/{up_art_id}", headers=adm).status_code == 200

    # Reject s0 -> moves to quarantine -> ext2 STILL gets 403 on quarantined artifact
    assert client.post(
        "/api/admin/verifications/labels-prod/s0",
        headers=adm,
        json={"passed": False, "verify_status": "rejected_label_spot_gate", "reason": "bad"},
    ).status_code == 200
    assert client.get(f"/api/artifacts/{up_art_id}", headers=ext2_hdr).status_code == 403

    # S5 + S9: Public /api/audit MUST NOT leak artifact_id, quarantine_id, claim_id, host_tag, reason
    audit = client.get("/api/audit?limit=50", headers=site_env["pub_hdr"]).json()
    assert len(audit) > 0
    forbidden_keys = {"artifact_id", "quarantine_id", "quarantined_artifact", "claim_id", "host_tag", "reason", "gate_metrics"}
    for ev in audit:
        det = ev["detail"]
        assert forbidden_keys.isdisjoint(det.keys()), f"leaked key in audit detail: {det}"
        assert up_art_id not in json.dumps(ev)

    # S9: Public /api/contributors MUST NOT expose approved, quarantined, strikes, tiers, last_seen_at
    contribs = client.get("/api/contributors", headers=site_env["pub_hdr"]).json()
    for c in contribs:
        for priv_k in ("approved", "quarantined", "strikes", "tiers", "last_seen_at"):
            assert priv_k not in c, f"private field '{priv_k}' exposed in /api/contributors"


def test_chunked_upload_floor_exact_size_sweep_and_subprocess_s6(site_env):
    client = site_env["client"]
    _seed_all_kinds(site_env)
    staging = site_env["data_dir"] / "uploads" / "staging"
    _, own_hdr = site_env["mint_user"]("owner-tr", tiers=["tier-train"], is_owner=True)

    clm = client.post(
        "/api/claims", headers=own_hdr, json={"runner": "runner-all", "job": "train-r1", "slot_id": "t_open", "request": "chk1"}
    ).json()

    # 1. Missing X-Content-SHA256 on direct upload -> 400
    r_nosha = client.post(f"/api/uploads/direct/{clm['claim_id']}?filename=a.bin", headers=own_hdr, content=b"abc")
    assert r_nosha.status_code == 400

    # 2. Chunk floor: chunk_bytes=1 on 600 KiB payload -> 400!
    r_floor = client.post(
        "/api/uploads/init",
        headers=own_hdr,
        json={
            "claim_id": clm["claim_id"],
            "filename": "delta.safetensors",
            "total_bytes": 600 * 1024,
            "total_sha256": "a" * 64,
            "chunk_bytes": 1,
        },
    )
    assert r_floor.status_code == 400, r_floor.text

    # 3. Valid multi-chunk (> 256 KiB) safetensors delta + exact chunk size check
    arr_q = np.ones(300_000, dtype=np.int8)
    arr_idx = np.arange(300_000, dtype=np.int32)
    delta_bytes = st_save_bytes(
        {
            "layer0.weight.__q": arr_q,
            "layer0.weight.__idx": arr_idx,
            "layer0.weight.__scale": np.array([0.01], dtype=np.float32),
        }
    )
    assert len(delta_bytes) > 256 * 1024
    dsha = hashlib.sha256(delta_bytes).hexdigest()
    chunk_sz = 256 * 1024
    init = client.post(
        "/api/uploads/init",
        headers=own_hdr,
        json={
            "claim_id": clm["claim_id"],
            "filename": "delta.safetensors",
            "total_bytes": len(delta_bytes),
            "total_sha256": dsha,
            "chunk_bytes": chunk_sz,
            "manifest": {"selfcheck_max_err": 0.0, "label_top1_agree": 0.975, "label_top5_agree": 0.999, "codec": "sparseloco_int8_topk10"},
        },
    )
    assert init.status_code == 201, init.text
    uid = init.json()["upload_id"]
    nchunks = init.json()["total_chunks"]
    assert nchunks >= 2

    # Missing X-Chunk-SHA256 -> 400
    p0 = delta_bytes[:chunk_sz]
    assert client.put(f"/api/uploads/{uid}/chunk/0", headers=own_hdr, content=p0).status_code == 400

    # Wrong chunk size (oversized/undersized vs declared chunk_bytes) -> 422
    wrong_sz = b"x" * 100
    r_wsz = client.put(
        f"/api/uploads/{uid}/chunk/0",
        headers={**own_hdr, "X-Chunk-SHA256": hashlib.sha256(wrong_sz).hexdigest()},
        content=wrong_sz,
    )
    assert r_wsz.status_code == 422, r_wsz.text

    # Upload all valid chunks
    for i in range(nchunks):
        part = delta_bytes[i * chunk_sz : (i + 1) * chunk_sz]
        pr = client.put(
            f"/api/uploads/{uid}/chunk/{i}",
            headers={**own_hdr, "X-Chunk-SHA256": hashlib.sha256(part).hexdigest()},
            content=part,
        )
        assert pr.status_code == 200, pr.text

    comp = client.post(f"/api/uploads/{uid}/complete", headers=own_hdr)
    assert comp.status_code == 200, comp.text
    assert comp.json()["state"] == "uploaded"
    assert comp.json()["verify_status"] == "pending_replay_gate"
    assert list(staging.iterdir()) == []

    # 4. Expired upload session & orphaned staging dir sweep
    from server import db

    orphan = staging / "upl_orphan_test"
    orphan.mkdir(parents=True, exist_ok=True)
    (orphan / "part_000000").write_bytes(b"junk")
    old_ts = time.time() - 7200
    os.utime(orphan, (old_ts, old_ts))
    with db.tx(immediate=True) as conn:
        swept = db.sweep_expired_uploads(conn)
    assert swept >= 1 and not orphan.exists()


def test_admin_upsert_preserves_omitted_fields_s10(site_env):
    client = site_env["client"]
    adm = site_env["adm_hdr"]

    r1 = client.post(
        "/api/admin/contributors",
        headers=adm,
        json={"login": "bob-hold", "approved": False, "quarantined": True, "tiers": ["tier-label"], "strikes": 2, "note": "initial"},
    )
    assert r1.status_code == 200
    assert r1.json()["approved"] is False and r1.json()["quarantined"] is True

    # Update ONLY note -> MUST NOT re-approve or un-quarantine or reset tiers/strikes!
    r2 = client.post("/api/admin/contributors", headers=adm, json={"login": "bob-hold", "note": "updated note"})
    assert r2.status_code == 200
    b2 = r2.json()
    assert b2["approved"] is False
    assert b2["quarantined"] is True
    assert b2["tiers"] == ["tier-label"]
    assert b2["strikes"] == 2

    # Key rotation without approval flags MUST also preserve quarantined/unapproved state
    r3 = client.post("/api/admin/contributors", headers=adm, json={"login": "bob-hold", "rotate_key": True})
    assert r3.status_code == 200 and "api_key_shown_once" in r3.json()
    assert r3.json()["approved"] is False and r3.json()["quarantined"] is True


def test_grid_pagination_and_delete_journal_mode_s8(site_env):
    client = site_env["client"]
    _seed_all_kinds(site_env)

    # Verify PRAGMA journal_mode=delete
    conn = sqlite3.connect(str(site_env["data_dir"] / "db" / "draft_site.sqlite3"))
    try:
        jm = conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
    finally:
        conn.close()
    assert jm == "delete"

    # Paginated grid returns requested slice + total_slots
    g = client.get("/api/grid?job=labels-prod&offset=1&limit=2", headers=site_env["pub_hdr"]).json()
    assert g["total_slots"] == 4
    assert g["offset"] == 1 and g["limit"] == 2
    assert len(g["slots"]) == 2


def test_eval_self_quorum_blocked_and_peer_quorum(site_env):
    client = site_env["client"]
    _seed_all_kinds(site_env)
    _, own1 = site_env["mint_user"]("owner-e1", tiers=["tier-eval"], is_owner=True)
    _, own2 = site_env["mint_user"]("owner-e2", tiers=["tier-eval"], is_owner=True)

    c1 = client.post("/api/claims", headers=own1, json={"runner": "runner-all", "job": "eval-r1", "slot_id": "e0", "request": "ev1"}).json()
    good_ev = json.dumps({
        "checkpoint_rev": "c0", "profile_id": "bit280", "quant": "mxfp8", "selfcheck_max_err": 0.0,
        "n_blocks": 1156, "b8_ref_tau": 4.83, "b16_ref_tau": 5.99, "ce8": 2.02, "top1": 0.875,
        "per_pos_ref": [0.87, 0.77, 0.66, 0.57, 0.50, 0.43, 0.37], "by_cat": {"code": {"b16_ref": 6.13}},
        "pred_sha256": "a" * 64,
    }).encode("utf-8")
    assert client.post(
        f"/api/uploads/direct/{c1['claim_id']}?filename=e0.json",
        headers={**own1, "X-Content-SHA256": hashlib.sha256(good_ev).hexdigest()},
        content=good_ev,
    ).status_code == 200

    # Same owner login cannot self-quorum the duplicate slot
    r_self = client.post("/api/claims", headers=own1, json={"runner": "runner-all", "job": "eval-r1", "slot_id": "e0_dup", "request": "ev1_dup"})
    assert r_self.status_code == 409

    # Different owner login submitting tampered metrics is rejected by peer quorum
    c2 = client.post("/api/claims", headers=own2, json={"runner": "runner-all", "job": "eval-r1", "slot_id": "e0_dup", "request": "ev2"}).json()
    bad_ev = json.dumps({
        "checkpoint_rev": "c0", "profile_id": "bit280", "quant": "mxfp8", "selfcheck_max_err": 0.0,
        "n_blocks": 1156, "b8_ref_tau": 4.83, "b16_ref_tau": 6.50, "ce8": 1.80, "top1": 0.91,
        "per_pos_ref": [0.91, 0.85, 0.75, 0.65, 0.55, 0.45, 0.37], "by_cat": {"code": {"b16_ref": 6.50}},
        "pred_sha256": "b" * 64,
    }).encode("utf-8")
    r_dup = client.post(
        f"/api/uploads/direct/{c2['claim_id']}?filename=e0_bad.json",
        headers={**own2, "X-Content-SHA256": hashlib.sha256(bad_ev).hexdigest()},
        content=bad_ev,
    )
    assert r_dup.status_code == 422 and r_dup.json()["verify_status"] == "failed_duplicate_quorum"


def test_capture_pack_pickle_and_provenance_enforcement(site_env):
    import zstandard as zstd

    client = site_env["client"]
    adm = site_env["adm_hdr"]
    seeded = _seed_all_kinds(site_env)
    _, own_all = site_env["mint_user"](
        "owner-cap-pack", tiers=["tier-capture", "tier-pack", "tier-train", "tier-label"], is_owner=True
    )

    # 1. Valid drafter-capture (4-tap INT8 + zstd, zero pickle) verified cleanly in subprocess
    c_cap = client.post(
        "/api/claims", headers=own_all, json={"runner": "runner-all", "job": "cap-r1", "slot_id": "c0", "request": "cap1"}
    ).json()
    rng = np.random.default_rng(7)
    st_bytes = st_save_bytes(
        {
            "taps_int8": rng.integers(-120, 120, size=(4, 4, 6144), dtype=np.int8),
            "taps_scale": np.full((4, 4), 0.05, dtype=np.float16),
            "tokens": np.arange(10, 18, dtype=np.int32),
            "topk_indices": np.arange(80, dtype=np.int32).reshape(4, 20),
            "topk_values": np.ones((4, 20), dtype=np.float16),
        }
    )
    z_bytes = zstd.ZstdCompressor(level=1).compress(st_bytes)
    r_cap = client.post(
        f"/api/uploads/direct/{c_cap['claim_id']}?filename=c0.4tap_int8.safetensors.zst",
        headers={
            **own_all,
            "X-Content-SHA256": hashlib.sha256(z_bytes).hexdigest(),
            "X-Manifest-JSON": json.dumps({"selfcheck_max_err": 0.0, "label_top1_agree": 0.97, "label_top5_agree": 0.999, "tap_layers": [33, 47, 61, 75]}),
        },
        content=z_bytes,
    )
    assert r_cap.status_code == 200 and r_cap.json()["state"] == "verified", r_cap.text

    # 2. Valid drafter-pack manifest verified cleanly; bad types return 422 (not 500)
    c_pk = client.post(
        "/api/claims", headers=own_all, json={"runner": "runner-all", "job": "pack-r1", "slot_id": "pk0", "request": "pk1"}
    ).json()
    bad_pk = json.dumps({"checkpoint_rev": "c0", "ckpt_sha256": "a" * 64, "nr": "not-an-int", "format_version": "g6dspine-v1", "ranks": []}).encode()
    r_pk_bad = client.post(
        f"/api/uploads/direct/{c_pk['claim_id']}?filename=bad_pk.json",
        headers={**own_all, "X-Content-SHA256": hashlib.sha256(bad_pk).hexdigest()},
        content=bad_pk,
    )
    assert r_pk_bad.status_code == 422

    c_pk2 = client.post(
        "/api/claims", headers=own_all, json={"runner": "runner-all", "job": "pack-r1", "slot_id": "pk0", "request": "pk2"}
    ).json()
    good_pk = json.dumps(
        {
            "checkpoint_rev": "c0",
            "ckpt_sha256": "a" * 64,
            "nr": 4,
            "format_version": "g6dspine-v1",
            "ranks": [{"rank": i, "pack_sha256": f"{i}" * 64, "pack_bytes": 1000 + i} for i in range(4)],
        }
    ).encode()
    r_pk_ok = client.post(
        f"/api/uploads/direct/{c_pk2['claim_id']}?filename=pk.json",
        headers={**own_all, "X-Content-SHA256": hashlib.sha256(good_pk).hexdigest()},
        content=good_pk,
    )
    assert r_pk_ok.status_code == 200 and r_pk_ok.json()["state"] == "verified", r_pk_ok.text

    # 3. Malicious pickle bytes on drafter-train rejected cleanly (422) with zero staged file leak
    c_tr = client.post(
        "/api/claims", headers=own_all, json={"runner": "runner-all", "job": "train-r1", "slot_id": "t_open", "request": "tr_pkl"}
    ).json()
    pkl = b"\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00cos\nsystem\n(Vecho pwned\ntR."
    r_pkl = client.post(
        f"/api/uploads/direct/{c_tr['claim_id']}?filename=evil.safetensors",
        headers={
            **own_all,
            "X-Content-SHA256": hashlib.sha256(pkl).hexdigest(),
            "X-Manifest-JSON": json.dumps({"selfcheck_max_err": 0.0, "label_top1_agree": 0.96, "label_top5_agree": 0.999, "codec": "sparseloco_int8_topk10"}),
        },
        content=pkl,
    )
    assert r_pkl.status_code == 422 and r_pkl.json()["verify_status"] == "failed_safetensors"
    assert list((site_env["data_dir"] / "uploads" / "staging").iterdir()) == []

    # 4. Provenance enforcement (Fable 4.4): require_provenance + allowed_quant_profiles
    assert client.post(
        "/api/admin/slots/labels-prod/s2/ready",
        headers=adm,
        json={
            "state": "open",
            "plan": {"n_prompts": 4, "max_tokens_short": 64, "require_provenance": True, "allowed_quant_profiles": ["bit280"]},
        },
    ).status_code == 200
    c_pv = client.post(
        "/api/claims", headers=own_all, json={"runner": "drs-label-ext-v1.0.0", "job": "labels-prod", "slot_id": "s2", "request": "pv1"}
    ).json()
    gz_noprov = _make_label_gz(seeded["p_rows2"], include_provenance=False)
    r_np = client.post(
        f"/api/uploads/direct/{c_pv['claim_id']}?filename=noprov.gz",
        headers={**own_all, "X-Content-SHA256": hashlib.sha256(gz_noprov).hexdigest()},
        content=gz_noprov,
    )
    assert r_np.status_code == 422 and r_np.json()["verify_status"] == "failed_provenance"

