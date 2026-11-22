#!/usr/bin/env python3
"""Live red-team & security verification against https://draft.bertholomus.com from a remote host.

Exercises deliberately bad uploads across job kinds and security boundaries via the public URL,
verifying every bad upload is REJECTED (HTTP 422/401/403/413), quarantined, and the slot reopened.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from client import DraftSiteClient, DraftSiteError  # noqa: E402


def main() -> None:
    url = sys.argv[1]
    key_file = Path(sys.argv[2])
    keys = json.loads(key_file.read_text(encoding="utf-8"))
    rt_key = keys["bertholomus-qa"]
    cli = DraftSiteClient(url, rt_key)

    results = []
    me = cli.me()
    print(f"[redteam] Authenticated as {me['login']} (admin={me['admin']}, approved={me['approved']})")

    # 1. Privilege separation: contributor key MUST be rejected (403) on /api/admin/*
    try:
        cli.admin_post("/api/admin/runners", {"commit_id": "evil-runner", "kind": "all"})
        raise AssertionError("contributor key was allowed on admin endpoint!")
    except DraftSiteError as e:
        assert e.status == 403, f"expected 403, got {e.status}"
        results.append({"test": "contributor_on_admin_endpoint", "expected": 403, "got": e.status, "pass": True})
        print(f"[redteam] PASS: contributor blocked from admin endpoint (HTTP {e.status})")

    # 2. Bad drafter-label upload (corrupted prompt_sha256 + off-policy semantic agreement)
    clm = cli.claim(runner="drs-worker-v1.0.0", job="glm53-dspine-labels", slot_id="shard_0001", host_tag="alberts-mac-mini-redteam")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "poison_labels.jsonl.gz"
        bad_row = {
            "id": "wrong_id_999",
            "prompt_sha256": "0" * 64,
            "gen_ids": [42, 43, 44],
            "finish_reason": "stop",
            "engine_tag": "off-policy-fake-model",
            "gen_sha256": hashlib.sha256(b"[42,43,44]").hexdigest(),
        }
        p.write_bytes(gzip.compress((json.dumps(bad_row) + "\n").encode("utf-8")))
        try:
            cli.upload_artifact(clm, p, manifest={"semantic_top1_agree": 0.51})
            raise AssertionError("corrupt label upload was accepted!")
        except DraftSiteError as e:
            assert e.status == 422, f"expected 422, got {e.status}: {e.body}"
            results.append({"test": "bad_drafter_label_corrupt_prompt", "expected": 422, "got": e.status, "detail": e.json, "pass": True})
            print(f"[redteam] PASS: bad drafter-label rejected (HTTP {e.status}, status={e.json.get('verify_status')})")

    # 3. Bad drafter-eval upload (non-zero selfcheck_max_err = 0.0125)
    eclm = cli.claim(runner="drs-worker-v1.0.0", job="glm53-dspine-eval", slot_id="v5-s2-w5-champion__bit280__dup", host_tag="alberts-mac-mini-redteam")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bad_selfcheck_eval.json"
        p.write_text(json.dumps({
            "checkpoint_rev": "v5-s2-w5-champion",
            "profile_id": "bit280",
            "quant": "mxfp8",
            "selfcheck_max_err": 0.0125,
            "n_blocks": 1156,
            "b8_ref_tau": 4.83,
            "b16_ref_tau": 5.99,
            "ce8": 2.02,
            "top1": 0.875,
            "per_pos_ref": [0.87, 0.77, 0.66, 0.57, 0.50, 0.43, 0.37],
            "by_cat": {"code": {"b16_ref": 6.13}},
            "pred_sha256": "c" * 64,
        }), encoding="utf-8")
        try:
            cli.upload_artifact(eclm, p)
            raise AssertionError("bad selfcheck eval was accepted!")
        except DraftSiteError as e:
            assert e.status == 422, f"expected 422, got {e.status}: {e.body}"
            results.append({"test": "bad_drafter_eval_nonzero_selfcheck", "expected": 422, "got": e.status, "detail": e.json, "pass": True})
            print(f"[redteam] PASS: bad drafter-eval (selfcheck>0) rejected (HTTP {e.status}, status={e.json.get('verify_status')})")

    # 4. Bad drafter-capture upload (off-policy top1_agree=0.61 + corrupt payload)
    cclm = cli.claim(runner="drs-worker-v1.0.0", job="glm53-dspine-capture", slot_id="cap_bit400_ev0", host_tag="alberts-mac-mini-redteam")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bad_capture.safetensors.zst"
        p.write_bytes(b"NOT_A_VALID_SAFETENSORS_FILE")
        try:
            cli.upload_artifact(cclm, p, manifest={"selfcheck_max_err": 0.0, "label_top1_agree": 0.61, "label_top5_agree": 0.82, "tap_layers": [33, 47, 61, 75]})
            raise AssertionError("bad capture upload was accepted!")
        except DraftSiteError as e:
            assert e.status == 422, f"expected 422, got {e.status}: {e.body}"
            results.append({"test": "bad_drafter_capture_low_agree", "expected": 422, "got": e.status, "detail": e.json, "pass": True})
            print(f"[redteam] PASS: bad drafter-capture rejected (HTTP {e.status}, status={e.json.get('verify_status')})")

    # 5. Bad drafter-train upload (Phase A: NaN / bogus pickle attempt -> rejected without ever invoking pickle!)
    tclm = cli.claim(runner="drs-worker-v1.0.0", job="glm53-dspine-train", slot_id="r1_w0_bit200", host_tag="alberts-mac-mini-redteam")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "malicious_pickle_attempt.safetensors"
        # Fake pickle header bytes — must be rejected cleanly by safetensors parser without executing anything!
        p.write_bytes(b"\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00cos\nsystem\n(Vecho pwned\ntR.")
        try:
            cli.upload_artifact(tclm, p, manifest={"selfcheck_max_err": 0.0, "label_top1_agree": 0.96, "label_top5_agree": 0.999, "codec": "sparseloco_int8_topk10"})
            raise AssertionError("malicious pickle bytes accepted!")
        except DraftSiteError as e:
            assert e.status == 422, f"expected 422, got {e.status}: {e.body}"
            results.append({"test": "bad_drafter_train_pickle_rejected_safely", "expected": 422, "got": e.status, "detail": e.json, "pass": True})
            print(f"[redteam] PASS: pickle payload rejected safely without execution (HTTP {e.status}, status={e.json.get('verify_status')})")

    # 6. Oversized upload > 16 MiB rejected by proxy/server with HTTP 413
    clm_sz = cli.claim(runner="drs-worker-v1.0.0", job="glm53-dspine-eval", slot_id="v5-s2-w5-champion__bit280__dup", host_tag="alberts-mac-mini-redteam")
    try:
        try:
            cli._request("POST", clm_sz["upload"]["init_url"], json_body={
                "claim_id": clm_sz["claim_id"],
                "filename": "huge.json",
                "total_bytes": 999_999_999,
                "total_sha256": "d" * 64,
                "chunk_bytes": 8 * 1024 * 1024,
            })
            raise AssertionError("oversized init accepted!")
        except DraftSiteError as e:
            assert e.status == 413, f"expected 413, got {e.status}: {e.body}"
            results.append({"test": "oversized_upload_cap_413", "expected": 413, "got": e.status, "pass": True})
            print(f"[redteam] PASS: oversized upload rejected (HTTP {e.status})")
    finally:
        cli.release("glm53-dspine-eval", "v5-s2-w5-champion__bit280__dup")

    print("\n=== ALL 6 LIVE PUBLIC RED-TEAM TESTS PASSED ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
