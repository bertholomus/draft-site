#!/usr/bin/env python3
"""Live red-team & security verification against https://draft.bertholomus.com.

Exercises every hardened security boundary (S1, S4, S5, S6, S7, S9, S10, Fable 4.8) PLUS the
GitHub OAuth, session, CSRF, open-redirect, self-service key panel, XSS, and pending/denied/
quarantined surface using disposable non-owner test logins (auto-disabled at the end) so zero
production slots or owner worker credentials are harmed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from client import DraftSiteClient, DraftSiteError  # noqa: E402


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect())


def _raw_http(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> tuple[int, Any, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with _NO_REDIRECT_OPENER.open(req, timeout=15) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def _raw_http_status(url: str, headers: dict[str, str]) -> int:
    st, _, _ = _raw_http(url, headers=headers)
    return st


def _docker_create_test_session(container: str, login: str) -> tuple[str, str]:
    """Create a disposable session inside the running container (never opens SQLite from host)."""
    py = (
        "import json, sys; from server import db; "
        "login = sys.argv[1]; "
        "with db.tx(immediate=True) as conn: "
        "  sid, csrf = db.create_session(conn, login); "
        "print(json.dumps({'sid': sid, 'csrf': csrf}))"
    )
    out = subprocess.check_output(
        ["docker", "exec", container, "python3", "-c", py, login],
        text=True,
        timeout=15,
    )
    d = json.loads(out.strip())
    return str(d["sid"]), str(d["csrf"])


def _docker_exec_py(container: str, code: str, *args: str) -> str:
    return subprocess.check_output(
        ["docker", "exec", container, "python3", "-c", code, *args],
        text=True,
        timeout=15,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Live public red-team suite for draft.bertholomus.com")
    ap.add_argument("--url", default="https://draft.bertholomus.com")
    ap.add_argument("--admin-url", default="http://127.0.0.1:8742")
    ap.add_argument("--admin-key-file", required=True)
    ap.add_argument("--origin-auth-file", default=None)
    ap.add_argument("--docker-container", default=None, help="Container name (e.g. draft-site) for live session/CSRF/key-panel red-team checks")
    ap.add_argument("--test-login", default="redteam-ext-verify")
    ap.add_argument("--pending-login", default="redteam-pending-xss")
    ap.add_argument("--label-job", default="glm53-dspine-labels-prod")
    ap.add_argument("--slot-a", default="v2_s0382")
    ap.add_argument("--slot-b", default="v2_s0383")
    ap.add_argument("--slot-c", default="v2_s0384")
    args = ap.parse_args()

    pub_base = args.url.rstrip("/")
    adm_base = args.admin_url.rstrip("/")
    admin_key = Path(args.admin_key_file).read_text(encoding="utf-8").strip()
    origin_auth = (
        Path(args.origin_auth_file).read_text(encoding="utf-8").strip()
        if args.origin_auth_file and Path(args.origin_auth_file).is_file()
        else None
    )
    adm = DraftSiteClient(adm_base, admin_key, origin_auth=origin_auth)
    results: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        assert ok, f"REDTEAM FAIL {name}: {detail}"
        results.append({"test": name, "pass": True, "detail": detail})
        print(f"[redteam] PASS: {name} ({detail})")

    # 1. S4: Direct origin without X-Origin-Auth MUST be denied (403) on ALL non-/healthz paths (old + new)
    for path in ("/api/jobs", "/auth/login", "/auth/callback", "/auth/me", "/api/keys", "/waiting", "/admin", "/api/admin/access-requests"):
        st_no_orig = _raw_http_status(f"{adm_base}{path}", {"X-Quant": "1", "Host": "evil.example"})
        record(f"s4_origin_default_deny{path.replace('/', '_')}", st_no_orig == 403, f"HTTP {st_no_orig}")

    # 2. Mint disposable non-owner test login (even giving extra tiers to prove non-owner policy blocks non-label!)
    mint = adm.admin_post(
        "/api/admin/contributors",
        {
            "login": args.test_login,
            "is_owner": False,
            "approved": True,
            "denied": False,
            "quarantined": False,
            "strikes": 0,
            "tiers": ["tier-label", "tier-train", "tier-eval", "tier-capture", "tier-pack"],
            "rotate_key": True,
            "note": "disposable redteam test account",
        },
    )
    test_key = mint["api_key_shown_once"]
    cli = DraftSiteClient(pub_base, test_key)
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
        adm.admin_post("/api/admin/contributors", {"login": args.test_login, "approved": True, "quarantined": False})

        # 4. Privilege separation: non-owner blocked from /api/admin/* (old + new access-requests endpoints)
        try:
            cli.admin_post("/api/admin/runners", {"commit_id": "evil-runner", "kind": "all"})
            raise AssertionError("non-owner allowed on admin endpoint")
        except DraftSiteError as e:
            record("rbac_admin_blocked", e.status == 403, f"HTTP {e.status}")

        try:
            cli.admin_get("/api/admin/access-requests")
            raise AssertionError("non-owner allowed on /api/admin/access-requests")
        except DraftSiteError as e:
            record("rbac_admin_access_requests_blocked", e.status == 403, f"HTTP {e.status}")

        # 5. Fable 4.8: Non-owner blocked from train, eval, capture, pack
        for blocked_job in ("glm53-dspine-train-r2", "glm53-dspine-eval", "glm53-dspine-capture-v2", "glm53-dspine-pack"):
            try:
                cli.claim(runner="drs-worker-v1.1.0", job=blocked_job)
                raise AssertionError(f"non-owner allowed to claim {blocked_job}")
            except DraftSiteError as e:
                record(f"fable48_block_{blocked_job}", e.status == 403, f"HTTP {e.status}")

        # 6. S5 & S9: Public /api/audit & /api/contributors redaction
        audit = cli._request("GET", "/api/audit?limit=40")
        forbidden_audit = {
            "artifact_id", "quarantine_id", "quarantined_artifact", "claim_id", "host_tag",
            "reason", "gate_metrics", "key_id", "prefix", "api_key", "api_key_shown_once", "csrf_token",
        }
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

        # 9. S6: Chunk floor & oversize cap
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
        cli.release(c2["job"], c2["slot_id"])
        claimed_to_release.remove((c2["job"], c2["slot_id"]))

        with tempfile.TemporaryDirectory() as td:
            p_pkl = Path(td) / "evil.labels.jsonl.gz"
            p_pkl.write_bytes(b"\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00cos\nsystem\n(Vecho pwned\ntR.")
            try:
                cli.upload_artifact(c1, p_pkl, manifest={"semantic_top1_agree": 0.999})
                raise AssertionError("pickle bytes accepted on label slot")
            except DraftSiteError as e:
                claimed_to_release.remove((c1["job"], c1["slot_id"]))
                record("s1_s6_pickle_rejected_safely", e.status == 422, f"HTTP {e.status} status={e.json.get('verify_status')}")

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

        try:
            cli.claim(runner="drs-label-ext-v1.0.0", job=args.label_job, slot_id=args.slot_a)
            raise AssertionError("quarantined account still allowed to claim")
        except DraftSiteError as e:
            record("s7_quarantined_account_blocked", e.status == 403, f"HTTP {e.status}")

        # 11. S14: CSP header has no 'unsafe-inline' in script-src & no inline onclick
        st_home, hdr_home, body_home = _raw_http(pub_base + "/")
        csp = hdr_home.get("Content-Security-Policy", "")
        record(
            "s14_csp_no_unsafe_inline_script",
            (st_home == 200 and "script-src 'self'" in csp and "script-src 'self' 'unsafe-inline'" not in csp and b"onclick=" not in body_home),
            csp[:80],
        )

        # 12. GHLOGIN Surface: OAuth login redirect, __Host- state cookie, bad/missing/replayed state, open redirect
        st_me0, _, body_me0 = _raw_http(pub_base + "/auth/me")
        me0 = json.loads(body_me0.decode("utf-8"))
        oauth_configured = bool(me0.get("oauth_configured"))
        record("gh_auth_me_unauthenticated", st_me0 == 200 and me0.get("authenticated") is False, f"oauth_configured={oauth_configured}")

        # Unauthenticated /api/keys MUST be 401
        st_keys_unauth = _raw_http_status(pub_base + "/api/keys", {"X-Quant": "1"})
        record("gh_keys_unauth_401", st_keys_unauth == 401, f"HTTP {st_keys_unauth}")

        # Unauthenticated /admin MUST NOT expose admin UI (302 to login)
        st_adm_unauth, hdr_adm_unauth, _ = _raw_http(pub_base + "/admin")
        loc_adm = hdr_adm_unauth.get("Location", "")
        record(
            "gh_admin_page_unauth_redirect",
            st_adm_unauth == 302 and loc_adm.startswith(f"{pub_base}/auth/login"),
            f"HTTP {st_adm_unauth} -> {loc_adm}",
        )

        if oauth_configured:
            st_login, hdr_login, _ = _raw_http(pub_base + "/auth/login")
            loc_login = hdr_login.get("Location", "")
            parsed = urllib.parse.urlsplit(loc_login)
            qs = urllib.parse.parse_qs(parsed.query)
            set_cookies = hdr_login.get_all("Set-Cookie") or []
            st_cookie_hdr = [c for c in set_cookies if c.startswith("__Host-drs_oauth_state=")]
            ok_login = (
                st_login == 302
                and f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == "https://github.com/login/oauth/authorize"
                and qs.get("redirect_uri") == [f"{pub_base}/auth/callback"]
                and qs.get("allow_signup") == ["false"]
                and "scope" not in qs
                and len(st_cookie_hdr) == 1
                and "secure" in st_cookie_hdr[0].lower()
                and "httponly" in st_cookie_hdr[0].lower()
                and "samesite=lax" in st_cookie_hdr[0].lower()
                and "path=/" in st_cookie_hdr[0].lower()
                and "domain=" not in st_cookie_hdr[0].lower()
            )
            record("gh_oauth_login_redirect_and_host_cookie", ok_login, f"HTTP {st_login} redirect_uri={qs.get('redirect_uri')}")

            # Open redirect attempts on /auth/login?next=...
            for evil in ("https://evil.example", "//evil.example", "/\\evil.example"):
                st_ev, hdr_ev, _ = _raw_http(pub_base + "/auth/login?next=" + urllib.parse.quote(evil, safe=""))
                loc_ev = hdr_ev.get("Location", "")
                record(
                    f"gh_open_redirect_blocked_{hashlib.sha256(evil.encode()).hexdigest()[:6]}",
                    st_ev == 302 and loc_ev.startswith("https://github.com/login/oauth/authorize?") and "evil.example" not in loc_ev,
                    f"next={evil}",
                )

            # Bad / missing / replayed state on /auth/callback -> 400
            st_cb_miss = _raw_http_status(pub_base + "/auth/callback", {})
            record("gh_callback_missing_state_400", st_cb_miss == 400, f"HTTP {st_cb_miss}")

            state_val = (qs.get("state") or ["none"])[0]
            cookie_pair = st_cookie_hdr[0].split(";", 1)[0] if st_cookie_hdr else ""
            # Bad state with real cookie -> 400
            st_cb_bad = _raw_http_status(pub_base + "/auth/callback?code=fake&state=wrong_state", {"Cookie": cookie_pair})
            record("gh_callback_bad_state_400", st_cb_bad == 400, f"HTTP {st_cb_bad}")
            # Missing cookie with real state -> 400 (and burns state!)
            st_cb_noc = _raw_http_status(pub_base + f"/auth/callback?code=fake&state={urllib.parse.quote(state_val)}", {})
            record("gh_callback_missing_cookie_400", st_cb_noc == 400, f"HTTP {st_cb_noc}")
            # Replayed state with real cookie -> STILL 400 (already burned!)
            st_cb_rep = _raw_http_status(
                pub_base + f"/auth/callback?code=fake&state={urllib.parse.quote(state_val)}",
                {"Cookie": cookie_pair},
            )
            record("gh_callback_replayed_state_400", st_cb_rep == 400, f"HTTP {st_cb_rep}")
        else:
            st_cb_unconf = _raw_http_status(pub_base + "/auth/login", {})
            record("gh_oauth_unconfigured_503", st_cb_unconf == 503, f"HTTP {st_cb_unconf} (awaiting github_oauth.json)")

        # 13. Full live-proxy session, CSRF, key panel, XSS, pending/denied/quarantined & fixation checks
        if args.docker_container:
            # 13a. Pending + XSS user cannot mint keys, sees escaped /waiting page, blocked from /admin
            _docker_exec_py(
                args.docker_container,
                "import sys; from server import db; "
                "login = sys.argv[1]; "
                "with db.tx(immediate=True) as conn: "
                "  conn.execute('DELETE FROM contributors WHERE login=?', (login,)); "
                "  conn.execute("
                "    \"INSERT INTO contributors (login, key_hash, is_admin, is_owner, approved, denied, tiers_json, strikes, quarantined, note, github_id, github_login, github_name, created_at) "
                "     VALUES (?, ?, 0, 0, 0, 0, '[\\\"tier-label\\\"]', 0, 0, 'redteam pending', 999999001, ?, ?, ?)\", "
                "    (login, db.no_key_sentinel(), 'xss-user', '<script>alert(1)</script><img src=x onerror=1>', db.utc_ms())"
                "  )",
                args.pending_login,
            )
            p_sid, p_csrf = _docker_create_test_session(args.docker_container, args.pending_login)
            p_cookie = f"__Host-drs_session={p_sid}"

            st_pme, _, body_pme = _raw_http(pub_base + "/auth/me", headers={"Cookie": p_cookie})
            pme = json.loads(body_pme.decode("utf-8"))
            record(
                "gh_proxy_cookie_and_pending_session",
                st_pme == 200 and pme.get("authenticated") is True and pme.get("status") == "pending",
                f"status={pme.get('status')} (proves Hostinger proxy forwards Cookie!)",
            )

            st_wait, _, body_wait = _raw_http(pub_base + "/waiting", headers={"Cookie": p_cookie})
            wait_txt = body_wait.decode("utf-8", errors="replace")
            record(
                "gh_waiting_page_xss_escaped",
                st_wait == 200 and "Waiting for approval" in wait_txt and "<script>alert(1)</script>" not in wait_txt and "&lt;script&gt;alert(1)&lt;/script&gt;" in wait_txt,
                f"HTTP {st_wait}",
            )

            st_pmint, _, _ = _raw_http(
                pub_base + "/api/keys",
                method="POST",
                headers={"Cookie": p_cookie, "X-Quant": "1", "X-CSRF-Token": p_csrf, "Content-Type": "application/json"},
                body=b"{}",
            )
            record("gh_pending_cannot_mint_key_403", st_pmint == 403, f"HTTP {st_pmint}")

            st_padm = _raw_http_status(pub_base + "/admin", {"Cookie": p_cookie})
            st_papi = _raw_http_status(pub_base + "/api/admin/access-requests", {"Cookie": p_cookie, "X-Quant": "1"})
            record("gh_pending_blocked_from_admin_403", st_padm == 403 and st_papi == 403, f"/admin={st_padm} api={st_papi}")

            # Deny via admin API -> kills pending session immediately!
            deny_res = adm.admin_post(f"/api/admin/access-requests/{args.pending_login}/deny", {"note": "redteam deny"})
            st_after_deny, _, body_after_deny = _raw_http(pub_base + "/auth/me", headers={"Cookie": p_cookie})
            record(
                "gh_admin_deny_kills_session",
                deny_res.get("denied") is True and st_after_deny == 200 and json.loads(body_after_deny.decode()).get("authenticated") is False,
                f"revoked_sessions={deny_res.get('revoked_sessions')}",
            )

            # 13b. Quarantined user (args.test_login currently at 3 strikes + quarantined=True!) cannot mint keys
            q_sid, q_csrf = _docker_create_test_session(args.docker_container, args.test_login)
            q_cookie = f"__Host-drs_session={q_sid}"
            st_qmint, _, _ = _raw_http(
                pub_base + "/api/keys",
                method="POST",
                headers={"Cookie": q_cookie, "X-Quant": "1", "X-CSRF-Token": q_csrf, "Content-Type": "application/json"},
                body=b"{}",
            )
            record("gh_quarantined_cannot_mint_key_403", st_qmint == 403, f"HTTP {st_qmint}")

            # 13c. Re-approve disposable test_login -> test CSRF, key create/cap/rotate/revoke, no key in HTML/audit, session rotation & logout!
            adm.admin_post("/api/admin/contributors", {"login": args.test_login, "approved": True, "denied": False, "quarantined": False, "strikes": 0})
            a_sid, a_csrf = _docker_create_test_session(args.docker_container, args.test_login)
            a_cookie = f"__Host-drs_session={a_sid}"

            # Missing CSRF -> 403
            st_nocsrf, _, _ = _raw_http(
                pub_base + "/api/keys",
                method="POST",
                headers={"Cookie": a_cookie, "X-Quant": "1", "Content-Type": "application/json"},
                body=b'{"label":"nocsrf"}',
            )
            record("gh_csrf_create_blocked_403", st_nocsrf == 403, f"HTTP {st_nocsrf}")

            # Revoke existing admin-issued key first via list, then mint 2 keys, verify 3rd -> 409
            st_kl, _, body_kl = _raw_http(pub_base + "/api/keys", headers={"Cookie": a_cookie, "X-Quant": "1"})
            kl = json.loads(body_kl.decode())
            for existing_k in kl.get("keys", []):
                _raw_http(
                    pub_base + f"/api/keys/{existing_k['key_id']}",
                    method="DELETE",
                    headers={"Cookie": a_cookie, "X-Quant": "1", "X-CSRF-Token": a_csrf},
                )

            st_k1, _, body_k1 = _raw_http(
                pub_base + "/api/keys",
                method="POST",
                headers={"Cookie": a_cookie, "X-Quant": "1", "X-CSRF-Token": a_csrf, "Content-Type": "application/json"},
                body=b'{"label":"rt-1"}',
            )
            k1_obj = json.loads(body_k1.decode())
            k1_raw = k1_obj.get("api_key_shown_once", "")
            k1_id = k1_obj.get("key_id", "")
            record("gh_key_create_via_live_proxy", st_k1 == 201 and k1_raw.startswith("drs_wrk_"), f"HTTP {st_k1}")

            # Key works immediately via /api/me
            st_k1_me = _raw_http_status(pub_base + "/api/me", {"X-Quant": "1", "Authorization": f"Bearer {k1_raw}"})
            record("gh_minted_key_authenticates", st_k1_me == 200, f"HTTP {st_k1_me}")

            # Key NEVER appears in HTML, GET /api/keys, or /api/audit
            _, _, body_kl2 = _raw_http(pub_base + "/api/keys", headers={"Cookie": a_cookie, "X-Quant": "1"})
            _, _, body_h2 = _raw_http(pub_base + "/", headers={"Cookie": a_cookie})
            _, _, body_au2 = _raw_http(pub_base + "/api/audit?limit=40", headers={"X-Quant": "1"})
            record(
                "gh_key_never_in_html_keys_or_audit",
                (k1_raw.encode() not in body_kl2 and k1_raw.encode() not in body_h2 and k1_raw.encode() not in body_au2),
                "verified absent from /, /api/keys, /api/audit",
            )

            # Create 2nd key -> 201; 3rd key -> 409
            st_k2, _, body_k2 = _raw_http(
                pub_base + "/api/keys",
                method="POST",
                headers={"Cookie": a_cookie, "X-Quant": "1", "X-CSRF-Token": a_csrf, "Content-Type": "application/json"},
                body=b'{"label":"rt-2"}',
            )
            k2_obj = json.loads(body_k2.decode())
            k2_id = k2_obj.get("key_id", "")
            st_k3, _, _ = _raw_http(
                pub_base + "/api/keys",
                method="POST",
                headers={"Cookie": a_cookie, "X-Quant": "1", "X-CSRF-Token": a_csrf, "Content-Type": "application/json"},
                body=b'{"label":"rt-3"}',
            )
            record("gh_key_cap_2_enforced", st_k2 == 201 and st_k3 == 409, f"k2={st_k2} k3={st_k3}")

            # CSRF enforced on rotate & revoke
            st_rot_nocsrf, _, _ = _raw_http(pub_base + f"/api/keys/{k1_id}/rotate", method="POST", headers={"Cookie": a_cookie, "X-Quant": "1"})
            st_rev_nocsrf, _, _ = _raw_http(pub_base + f"/api/keys/{k2_id}", method="DELETE", headers={"Cookie": a_cookie, "X-Quant": "1"})
            record("gh_csrf_rotate_revoke_blocked_403", st_rot_nocsrf == 403 and st_rev_nocsrf == 403, f"rot={st_rot_nocsrf} rev={st_rev_nocsrf}")

            # Rotate k1 -> old k1 immediately 401, new k1b works!
            st_rot, _, body_rot = _raw_http(
                pub_base + f"/api/keys/{k1_id}/rotate",
                method="POST",
                headers={"Cookie": a_cookie, "X-Quant": "1", "X-CSRF-Token": a_csrf, "Content-Type": "application/json"},
                body=b"{}",
            )
            k1b_obj = json.loads(body_rot.decode())
            k1b_raw = k1b_obj.get("api_key_shown_once", "")
            st_k1_old = _raw_http_status(pub_base + "/api/me", {"X-Quant": "1", "Authorization": f"Bearer {k1_raw}"})
            st_k1b_new = _raw_http_status(pub_base + "/api/me", {"X-Quant": "1", "Authorization": f"Bearer {k1b_raw}"})
            record("gh_key_rotate_revokes_old_immediately", st_rot == 200 and st_k1_old == 401 and st_k1b_new == 200, f"old={st_k1_old} new={st_k1b_new}")

            # Revoke k1b -> immediately 401
            st_rev, _, _ = _raw_http(
                pub_base + f"/api/keys/{k1b_obj['key_id']}",
                method="DELETE",
                headers={"Cookie": a_cookie, "X-Quant": "1", "X-CSRF-Token": a_csrf},
            )
            st_k1b_after = _raw_http_status(pub_base + "/api/me", {"X-Quant": "1", "Authorization": f"Bearer {k1b_raw}"})
            record("gh_key_revoke_immediate_401", st_rev == 200 and st_k1b_after == 401, f"rev={st_rev} after={st_k1b_after}")

            # Logout invalidates session server-side
            st_lo, _, _ = _raw_http(
                pub_base + "/auth/logout",
                method="POST",
                headers={"Cookie": a_cookie, "X-CSRF-Token": a_csrf},
            )
            st_after_lo, _, body_after_lo = _raw_http(pub_base + "/auth/me", headers={"Cookie": a_cookie})
            record(
                "gh_logout_invalidates_session",
                st_lo == 200 and st_after_lo == 200 and json.loads(body_after_lo.decode()).get("authenticated") is False,
                f"logout={st_lo}",
            )

    finally:
        for jb, sid in list(claimed_to_release):
            try:
                adm.release(jb, sid)
            except Exception:
                pass
        # Disable disposable test accounts cleanly & purge sessions/keys
        adm.admin_post("/api/admin/contributors", {"login": args.test_login, "approved": False, "denied": True, "quarantined": True, "note": "redteam test completed (disabled)"})
        if args.docker_container:
            try:
                _docker_exec_py(
                    args.docker_container,
                    "import sys; from server import db; "
                    "with db.tx(immediate=True) as conn: "
                    "  conn.execute('DELETE FROM contributors WHERE login=?', (sys.argv[1],))",
                    args.pending_login,
                )
            except Exception:
                pass

    print(f"\n=== ALL {len(results)} LIVE PUBLIC RED-TEAM CHECKS PASSED ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
