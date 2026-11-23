#!/usr/bin/env python3
"""Live red-team & security verification against https://draft.bertholomus.com.

Exercises every hardened security boundary (S1, S4, S5, S6, S7, S9, S10, Fable 4.8)
using a disposable non-owner test login (auto-disabled at the end) so zero production
slots or owner worker credentials are harmed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from client import DraftSiteClient, DraftSiteError  # noqa: E402


def _raw_http_status(url: str, headers: dict[str, str]) -> int:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def main() -> None:
    ap = argparse.ArgumentParser(description="Live public red-team suite for draft.bertholomus.com")
    ap.add_argument("--url", default="https://draft.bertholomus.com")
    ap.add_argument("--admin-url", default="http://127.0.0.1:8742")
    ap.add_argument("--admin-key-file", required=True)
    ap.add_argument("--origin-auth-file", default=None)
    ap.add_argument("--test-login", default="redteam-ext-verify")
    ap.add_argument("--label-job", default="glm53-dspine-labels-prod")
    ap.add_argument("--slot-a", default="v2_s0382")
    ap.add_argument("--slot-b", default="v2_s0383")
    ap.add_argument("--slot-c", default="v2_s0384")
    args = ap.parse_args()

    admin_key = Path(args.admin_key_file).read_text(encoding="utf-8").strip()
    origin_auth = (
        Path(args.origin_auth_file).read_text(encoding="utf-8").strip()
        if args.origin_auth_file and Path(args.origin_auth_file).is_file()
        else None
    )
    adm = DraftSiteClient(args.admin_url, admin_key, origin_auth=origin_auth)
    results: list[dict] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        assert ok, f"REDTEAM FAIL {name}: {detail}"
        results.append({"test": name, "pass": True, "detail": detail})
        print(f"[redteam] PASS: {name} ({detail})")

    # 1. S4: Direct origin without X-Origin-Auth MUST be denied (403) even with arbitrary Host
    st_no_orig = _raw_http_status(f"{args.admin_url.rstrip('/')}/api/jobs", {"X-Quant": "1", "Host": "evil.example"})
    record("s4_origin_default_deny", st_no_orig == 403, f"HTTP {st_no_orig}")

    # 2. Mint disposable non-owner test login (even giving extra tiers to prove non-owner policy blocks non-label!)
    mint = adm.admin_post(
        "/api/admin/contributors",
        {
            "login": args.test_login,
            "is_owner": False,
            "approved": True,
            "quarantined": False,
            "strikes": 0,
            "tiers": ["tier-label", "tier-train", "tier-eval", "tier-capture", "tier-pack"],
            "rotate_key": True,
            "note": "disposable redteam test account",
        },
    )
    test_key = mint["api_key_shown_once"]
    cli = DraftSiteClient(args.url, test_key)
    claimed_to_release: list[tuple[str, str]] = []

    try:
        # 3. S10: Admin update omitting approved/quarantined MUST NOT re-approve/un-quarantine
        adm.admin_post("/api/admin/contributors", {"login": args.test_login, "approved": False, "quarantined": True})
        chk_s10 = adm.admin_post("/api/admin/contributors", {"login": args.test_login, "note": "s10 preserve check"})
        record(
            "s10_admin_omitted_fields_preserved",
            (chk_s10["approved"] is False and chk_s10["quarantined"] is True),
            f"approved={chk_s10['approved']} quarantined={chk_s10['quarantined']}",
        )
        # Re-enable for remaining tests
        adm.admin_post("/api/admin/contributors", {"login": args.test_login, "approved": True, "quarantined": False})

        # 4. Privilege separation: non-owner blocked from /api/admin/*
        try:
            cli.admin_post("/api/admin/runners", {"commit_id": "evil-runner", "kind": "all"})
            raise AssertionError("non-owner allowed on admin endpoint")
        except DraftSiteError as e:
            record("rbac_admin_blocked", e.status == 403, f"HTTP {e.status}")

        # 5. Fable 4.8: Non-owner blocked from train, eval, capture, pack
        for blocked_job in ("glm53-dspine-train-r2", "glm53-dspine-eval", "glm53-dspine-capture-v2", "glm53-dspine-pack"):
            try:
                cli.claim(runner="drs-worker-v1.1.0", job=blocked_job)
                raise AssertionError(f"non-owner allowed to claim {blocked_job}")
            except DraftSiteError as e:
                record(f"fable48_block_{blocked_job}", e.status == 403, f"HTTP {e.status}")

        # 6. S5 & S9: Public /api/audit & /api/contributors redaction
        audit = cli._request("GET", "/api/audit?limit=40")
        forbidden_audit = {"artifact_id", "quarantine_id", "quarantined_artifact", "claim_id", "host_tag", "reason", "gate_metrics"}
        leaks = [k for ev in audit for k in forbidden_audit if k in (ev.get("detail") or {})]
        record("s5_s9_audit_redacted", not leaks, f"checked {len(audit)} events, leaks={len(leaks)}")

        contribs = cli.contributors()
        forbidden_contrib = {"approved", "quarantined", "strikes", "tiers", "last_seen_at"}
        c_leaks = [k for c in contribs for k in forbidden_contrib if k in c]
        record("s9_contributors_redacted", not c_leaks, f"checked {len(contribs)} rows, leaks={len(c_leaks)}")

        # 7. S5: Unclaimed artifact download MUST be rejected (403)
        try:
            cli._request("GET", f"/api/artifacts/inp_v2_{args.slot_a}")
            raise AssertionError("unclaimed artifact download allowed")
        except DraftSiteError as e:
            record("s5_unclaimed_download_blocked", e.status == 403, f"HTTP {e.status}")

        # 8. S7: Active claim cap = 2 (3rd claim rejected with 429)
        c1 = cli.claim(runner="drs-label-ext-v1.0.0", job=args.label_job, slot_id=args.slot_a)
        claimed_to_release.append((c1["job"], c1["slot_id"]))
        c2 = cli.claim(runner="drs-label-ext-v1.0.0", job=args.label_job, slot_id=args.slot_b)
        claimed_to_release.append((c2["job"], c2["slot_id"]))
        try:
            cli.claim(runner="drs-label-ext-v1.0.0", job=args.label_job, slot_id=args.slot_c)
            raise AssertionError("3rd active claim allowed beyond cap")
        except DraftSiteError as e:
            record("s7_claim_cap_enforced", e.status == 429, f"HTTP {e.status}")

        # S5 positive check: CAN download claimed shard, CANNOT download unclaimed slot_c shard
        inp_a = c1["inputs"][0]
        with tempfile.TemporaryDirectory() as td:
            p_dl = cli.download_artifact(inp_a["url"], Path(td) / "a.jsonl", inp_a["sha256"])
            record("s5_claimed_download_allowed", p_dl.stat().st_size == inp_a["bytes"], f"{inp_a['bytes']} B verified")
        try:
            cli._request("GET", f"/api/artifacts/inp_v2_{args.slot_c}")
            raise AssertionError("unclaimed sibling shard download allowed")
        except DraftSiteError as e:
            record("s5_sibling_download_blocked", e.status == 403, f"HTTP {e.status}")

        # 9. S6: Chunk floor (chunk_bytes=1 -> 400) & oversized total_bytes (-> 413)
        try:
            cli._request(
                "POST",
                c1["upload"]["init_url"],
                json_body={"claim_id": c1["claim_id"], "filename": "x.gz", "total_bytes": 600_000, "total_sha256": "a" * 64, "chunk_bytes": 1},
            )
            raise AssertionError("chunk_bytes=1 accepted")
        except DraftSiteError as e:
            record("s6_chunk_floor_enforced", e.status == 400, f"HTTP {e.status}")

        try:
            cli._request(
                "POST",
                c1["upload"]["init_url"],
                json_body={"claim_id": c1["claim_id"], "filename": "huge.gz", "total_bytes": 999_999_999, "total_sha256": "d" * 64, "chunk_bytes": 8 * 1024 * 1024},
            )
            raise AssertionError("oversized upload init accepted")
        except DraftSiteError as e:
            record("s6_oversize_cap_413", e.status == 413, f"HTTP {e.status}")

        # 10. S1 + S7: Bad uploads rejected (422), pickle rejected safely, strike 3 auto-quarantines!
        # Release slot_b first so we can cycle slot_a cleanly
        cli.release(c2["job"], c2["slot_id"])
        claimed_to_release.remove((c2["job"], c2["slot_id"]))

        # Strike 1: fake pickle payload
        with tempfile.TemporaryDirectory() as td:
            p_pkl = Path(td) / "evil.labels.jsonl.gz"
            p_pkl.write_bytes(b"\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00cos\nsystem\n(Vecho pwned\ntR.")
            try:
                cli.upload_artifact(c1, p_pkl, manifest={"semantic_top1_agree": 0.999})
                raise AssertionError("pickle bytes accepted on label slot")
            except DraftSiteError as e:
                claimed_to_release.remove((c1["job"], c1["slot_id"]))
                record("s1_s6_pickle_rejected_safely", e.status == 422, f"HTTP {e.status} status={e.json.get('verify_status')}")

        # Strike 2: corrupt prompt_sha with forged semantic_top1_agree=0.999
        c_s2 = cli.claim(runner="drs-label-ext-v1.0.0", job=args.label_job, slot_id=args.slot_a)
        claimed_to_release.append((c_s2["job"], c_s2["slot_id"]))
        with tempfile.TemporaryDirectory() as td:
            p_bad = Path(td) / "bad.labels.jsonl.gz"
            bad_row = {"id": "wrong_id", "prompt_sha256": "0" * 64, "gen_ids": [1, 2, 3, 4], "finish_reason": "stop", "engine_tag": "fake", "gen_sha256": hashlib.sha256(b"[1,2,3,4]").hexdigest()}
            p_bad.write_bytes(gzip.compress((json.dumps(bad_row) + "\n").encode("utf-8")))
            try:
                cli.upload_artifact(c_s2, p_bad, manifest={"semantic_top1_agree": 0.999})
                raise AssertionError("corrupt label accepted")
            except DraftSiteError as e:
                claimed_to_release.remove((c_s2["job"], c_s2["slot_id"]))
                record("s1_bad_label_rejected", e.status == 422, f"HTTP {e.status} status={e.json.get('verify_status')}")

        # Strike 3: short-row / EOS-fallback payload -> triggers auto-quarantine!
        c_s3 = cli.claim(runner="drs-label-ext-v1.0.0", job=args.label_job, slot_id=args.slot_a)
        claimed_to_release.append((c_s3["job"], c_s3["slot_id"]))
        with tempfile.TemporaryDirectory() as td:
            p_short = Path(td) / "short.labels.jsonl.gz"
            p_short.write_bytes(gzip.compress(b"\n"))
            try:
                cli.upload_artifact(c_s3, p_short)
                raise AssertionError("empty label accepted")
            except DraftSiteError as e:
                claimed_to_release.remove((c_s3["job"], c_s3["slot_id"]))
                record("s7_strike3_auto_quarantine", e.status == 422 and bool(e.json.get("quarantined")), f"HTTP {e.status} quarantined={e.json.get('quarantined')}")

        # Verify quarantined account is now blocked with 403!
        try:
            cli.claim(runner="drs-label-ext-v1.0.0", job=args.label_job, slot_id=args.slot_a)
            raise AssertionError("quarantined account still allowed to claim")
        except DraftSiteError as e:
            record("s7_quarantined_account_blocked", e.status == 403, f"HTTP {e.status}")

        # 11. S14: CSP header has no 'unsafe-inline' in script-src
        req_home = urllib.request.Request(args.url, method="GET")
        with urllib.request.urlopen(req_home, timeout=15) as rh:
            csp = rh.headers.get("Content-Security-Policy", "")
            record("s14_csp_no_unsafe_inline_script", ("script-src 'self'" in csp and "script-src 'self' 'unsafe-inline'" not in csp), csp[:80])

    finally:
        for jb, sid in list(claimed_to_release):
            try:
                adm.release(jb, sid)
            except Exception:
                pass
        # Disable disposable test account cleanly
        adm.admin_post("/api/admin/contributors", {"login": args.test_login, "approved": False, "quarantined": True, "note": "redteam test completed (disabled)"})

    print(f"\n=== ALL {len(results)} LIVE PUBLIC RED-TEAM CHECKS PASSED ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
