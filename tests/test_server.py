"""End-to-end API, security, chunked upload, and verification test suite for draft-site."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from safetensors.numpy import save as st_save_bytes

# Configure isolated temp data & secrets dirs BEFORE importing server
_TMP = Path(tempfile.mkdtemp(prefix="drs_test_"))
_DATA = _TMP / "data"
_SEC = _TMP / "secrets"
_SEC.mkdir(parents=True, exist_ok=True)
_DATA.mkdir(parents=True, exist_ok=True)
(_SEC / "origin_auth.secret").write_text("test_origin_secret_hex_1234567890", encoding="utf-8")
(_SEC / "admin_api_key.txt").write_text("drs_adm_test_admin_key", encoding="utf-8")
(_SEC / "pilot_worker_keys.json").write_text(
    json.dumps({"worker-a": "drs_wrk_alpha", "worker-b": "drs_wrk_beta"}),
    encoding="utf-8",
)

os.environ["DRAFT_DATA_DIR"] = str(_DATA)
os.environ["DRAFT_SECRETS_DIR"] = str(_SEC)
os.environ["DRAFT_PUBLIC_URL"] = "https://draft.bertholomus.com"
os.environ["DRAFT_ORIGIN_HOST"] = "draft-origin.albertlee.co"
os.environ["DRAFT_REQUIRE_ORIGIN_AUTH"] = "1"

from server import app as srv_app, db  # noqa: E402

db.init_db()
client = TestClient(srv_app.app)

ADM = {"X-Quant": "1", "Authorization": "Bearer drs_adm_test_admin_key"}
WA = {"X-Quant": "1", "Authorization": "Bearer drs_wrk_alpha"}
WB = {"X-Quant": "1", "Authorization": "Bearer drs_wrk_beta"}


def run_all() -> None:
    # 1. Origin auth guard
    r = client.get("/api/jobs", headers={"Host": "draft-origin.albertlee.co", "X-Quant": "1"})
    assert r.status_code == 403, f"expected 403 without origin auth, got {r.status_code}"
    r = client.get(
        "/api/jobs",
        headers={"Host": "draft-origin.albertlee.co", "X-Quant": "1", "X-Origin-Auth": "test_origin_secret_hex_1234567890"},
    )
    assert r.status_code == 200, r.text

    # 2. X-Quant header & Auth enforcement
    assert client.get("/api/jobs").status_code == 400
    assert client.post("/api/claims", headers={"X-Quant": "1"}, json={}).status_code == 401
    assert client.post("/api/admin/runners", headers=WA, json={"commit_id": "abc1234"}).status_code == 403
    assert client.get("/").status_code == 200
    assert client.get("/healthz").status_code == 200

    # 3. Register runner & seed jobs across all 4 kinds with readiness dependency
    assert client.post("/api/admin/runners", headers=ADM, json={"commit_id": "pilot-v1", "kind": "all"}).status_code == 200
    (_DATA / "inputs").mkdir(parents=True, exist_ok=True)
    p_rows = [
        {"id": "p0", "prompt": "Write fib(n)", "prompt_sha256": hashlib.sha256(b"Write fib(n)").hexdigest()},
        {"id": "p1", "prompt": "Sort list", "prompt_sha256": hashlib.sha256(b"Sort list").hexdigest()},
    ]
    (_DATA / "inputs" / "prompts_s0.jsonl").write_text("\n".join(json.dumps(x) for x in p_rows) + "\n", encoding="utf-8")
    art = client.post("/api/admin/artifacts/register", headers=ADM, json={"artifact_id": "inp_p0", "rel_path": "inputs/prompts_s0.jsonl"}).json()
    assert art["ok"]

    assert client.post(
        "/api/admin/jobs",
        headers=ADM,
        json={
            "job": "pilot-labels",
            "title": "Pilot Labels",
            "job_kind": "drafter-label",
            "slots": [
                {
                    "slot_id": "s0",
                    "row_key": "shard",
                    "col_key": "s0",
                    "profile_id": "bit280",
                    "state": "open",
                    "inputs": [{"name": "prompt_shard", "artifact_id": "inp_p0", "sha256": art["sha256"], "bytes": art["bytes"]}],
                    "plan": {"n_prompts": 2},
                }
            ],
        },
    ).status_code == 200

    assert client.post(
        "/api/admin/jobs",
        headers=ADM,
        json={
            "job": "pilot-train",
            "title": "Pilot Train",
            "job_kind": "drafter-train",
            "rounds": [{"round_num": 1, "base_ckpt_name": "theta_0", "quorum": 1, "total_slots": 1}],
            "slots": [
                {
                    "slot_id": "r1_bit280",
                    "row_key": "r1",
                    "col_key": "bit280",
                    "round_num": 1,
                    "profile_id": "bit280",
                    "state": "not_ready",
                    "depends_on_job": "pilot-labels",
                    "depends_on_slot": "s0",
                    "plan": {"codec": "sparseloco_int8_topk10", "profile_id": "bit280", "round_num": 1},
                }
            ],
        },
    ).status_code == 200

    assert client.post(
        "/api/admin/jobs",
        headers=ADM,
        json={
            "job": "pilot-eval",
            "title": "Pilot Eval",
            "job_kind": "drafter-eval",
            "slots": [
                {"slot_id": "c0_bit280_p1", "row_key": "c0", "col_key": "bit280", "profile_id": "bit280", "state": "open", "plan": {"checkpoint_rev": "c0", "profile_id": "bit280"}},
                {"slot_id": "c0_bit280_dup", "row_key": "c0", "col_key": "bit280", "profile_id": "bit280", "state": "open", "plan": {"checkpoint_rev": "c0", "profile_id": "bit280"}},
            ],
        },
    ).status_code == 200

    # 4. Label slot: reject bad prompt_sha first, then accept honest label -> verify readiness flips train slot open!
    clm = client.post("/api/claims", headers=WA, json={"runner": "pilot-v1", "job": "pilot-labels", "request": "r1"}).json()
    bad_lbl = gzip.compress(
        (json.dumps({"id": "p0", "prompt_sha256": "0" * 64, "gen_ids": [1, 2], "finish_reason": "stop", "engine_tag": "fake", "gen_sha256": hashlib.sha256(b"[1,2]").hexdigest()}) + "\n").encode()
    )
    rr = client.post(
        f"/api/uploads/direct/{clm['claim_id']}?filename=bad.jsonl.gz",
        headers={**WA, "X-Content-SHA256": hashlib.sha256(bad_lbl).hexdigest()},
        content=bad_lbl,
    )
    assert rr.status_code == 422 and rr.json()["accepted"] is False, rr.text

    # Re-claim reopened slot and upload honest labels
    clm2 = client.post("/api/claims", headers=WA, json={"runner": "pilot-v1", "job": "pilot-labels", "request": "r2"}).json()
    good_rows = []
    for pr in p_rows:
        gids = [10, 20, 30]
        good_rows.append({
            "id": pr["id"],
            "prompt_sha256": pr["prompt_sha256"],
            "gen_ids": gids,
            "finish_reason": "stop",
            "engine_tag": "gr8-bit280",
            "gen_sha256": hashlib.sha256(json.dumps(gids, separators=(",", ":")).encode()).hexdigest(),
        })
    good_lbl = gzip.compress(("\n".join(json.dumps(x) for x in good_rows) + "\n").encode())
    rr2 = client.post(
        f"/api/uploads/direct/{clm2['claim_id']}?filename=good.jsonl.gz",
        headers={**WA, "X-Content-SHA256": hashlib.sha256(good_lbl).hexdigest()},
        content=good_lbl,
    )
    assert rr2.status_code == 200 and rr2.json()["state"] == "verified", rr2.text
    assert rr2.json()["opened_downstream"] == [{"job": "pilot-train", "slot_id": "r1_bit280"}], rr2.json()

    # 5. Train slot: now OPEN via readiness gating! Test chunked upload of safetensors SparseLoCo delta + Owner Gate verdict!
    tclm = client.post("/api/claims", headers=WB, json={"runner": "pilot-v1", "job": "pilot-train", "request": "r3", "host_tag": "mastersim"}).json()
    assert tclm["slot_id"] == "r1_bit280"
    # Double claim must fail with 409
    assert client.post("/api/claims", headers=WA, json={"runner": "pilot-v1", "job": "pilot-train", "slot_id": "r1_bit280", "request": "r3b"}).status_code == 409

    delta_bytes = st_save_bytes({
        "layer0.weight.__q": np.array([12, -34, 56, -78], dtype=np.int8),
        "layer0.weight.__idx": np.array([0, 3, 7, 11], dtype=np.int32),
        "layer0.weight.__scale": np.array([0.01], dtype=np.float32),
    })
    dsha = hashlib.sha256(delta_bytes).hexdigest()
    init = client.post(
        "/api/uploads/init",
        headers=WB,
        json={
            "claim_id": tclm["claim_id"],
            "filename": "delta.safetensors",
            "total_bytes": len(delta_bytes),
            "total_sha256": dsha,
            "chunk_bytes": 128,
            "manifest": {"selfcheck_max_err": 0.0, "label_top1_agree": 0.975, "label_top5_agree": 0.999, "codec": "sparseloco_int8_topk10"},
        },
    ).json()
    for i in range(init["total_chunks"]):
        part = delta_bytes[i * 128 : (i + 1) * 128]
        pr = client.put(
            f"/api/uploads/{init['upload_id']}/chunk/{i}",
            headers={**WB, "X-Chunk-SHA256": hashlib.sha256(part).hexdigest()},
            content=part,
        )
        assert pr.status_code == 200, pr.text
    comp = client.post(f"/api/uploads/{init['upload_id']}/complete", headers=WB)
    assert comp.status_code == 200 and comp.json()["state"] == "uploaded", comp.text
    # Owner gate accepts it
    vg = client.post(
        "/api/admin/verifications/pilot-train/r1_bit280",
        headers=ADM,
        json={"passed": True, "reason": "delta_ce8=-0.22 cos=0.113", "gate_metrics": {"delta_ce8": -0.22, "cos_vs_root": 0.113}},
    )
    assert vg.status_code == 200 and vg.json()["state"] == "verified"

    # 6. Eval slot + duplicate quorum rejection test
    eclm1 = client.post("/api/claims", headers=WA, json={"runner": "pilot-v1", "job": "pilot-eval", "slot_id": "c0_bit280_p1", "request": "e1"}).json()
    good_ev = json.dumps({
        "checkpoint_rev": "c0", "profile_id": "bit280", "quant": "mxfp8", "selfcheck_max_err": 0.0,
        "n_blocks": 1156, "b8_ref_tau": 4.83, "b16_ref_tau": 5.99, "ce8": 2.02, "top1": 0.875,
        "per_pos_ref": [0.87, 0.77, 0.66, 0.57, 0.50, 0.43, 0.37], "by_cat": {"code": {"b16_ref": 6.13}},
        "pred_sha256": "a" * 64,
    }).encode()
    assert client.post(
        f"/api/uploads/direct/{eclm1['claim_id']}?filename=eval.json",
        headers={**WA, "X-Content-SHA256": hashlib.sha256(good_ev).hexdigest()},
        content=good_ev,
    ).status_code == 200

    eclm2 = client.post("/api/claims", headers=WB, json={"runner": "pilot-v1", "job": "pilot-eval", "slot_id": "c0_bit280_dup", "request": "e2"}).json()
    tampered_ev = json.dumps({
        "checkpoint_rev": "c0", "profile_id": "bit280", "quant": "mxfp8", "selfcheck_max_err": 0.0,
        "n_blocks": 1156, "b8_ref_tau": 4.83, "b16_ref_tau": 6.50, "ce8": 1.80, "top1": 0.91,
        "per_pos_ref": [0.91, 0.85, 0.75, 0.65, 0.55, 0.45, 0.37], "by_cat": {"code": {"b16_ref": 6.50}},
        "pred_sha256": "b" * 64,
    }).encode()
    rdup = client.post(
        f"/api/uploads/direct/{eclm2['claim_id']}?filename=eval_bad.json",
        headers={**WB, "X-Content-SHA256": hashlib.sha256(tampered_ev).hexdigest()},
        content=tampered_ev,
    )
    assert rdup.status_code == 422 and rdup.json()["verify_status"] == "failed_duplicate_quorum", rdup.text

    print("ALL SERVER TESTS PASSED (origin auth, RBAC, readiness gating, chunked safetensors, duplicate quorum, rejections)!")


if __name__ == "__main__":
    run_all()
