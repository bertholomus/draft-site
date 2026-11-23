"""Comprehensive tests for GitHub OAuth login, allow-list, sessions, CSRF, and self-service API keys."""
from __future__ import annotations

import http.server
import json
import sqlite3
import threading
import urllib.parse
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server import app as srv_app, config, db


def _extract_set_cookies(resp) -> list[str]:
    return resp.headers.get_list("set-cookie")


def _do_mock_login(
    browser: TestClient,
    origin_hdr: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    *,
    gh_id: int,
    gh_login: str,
    gh_name: str | None = None,
    next_url: str | None = None,
):
    """Run full /auth/login -> mocked GitHub code exchange -> /auth/callback flow."""
    captured: dict[str, Any] = {}

    def fake_exchange(*, code: str, client_id: str, client_secret: str, redirect_uri: str) -> dict[str, Any]:
        captured.update(
            {"code": code, "client_id": client_id, "client_secret": client_secret, "redirect_uri": redirect_uri}
        )
        return {"id": gh_id, "login": gh_login, "name": gh_name}

    monkeypatch.setattr(srv_app, "exchange_github_code", fake_exchange)

    login_path = "/auth/login"
    if next_url is not None:
        login_path += "?" + urllib.parse.urlencode({"next": next_url})
    r_start = browser.get(login_path, headers=origin_hdr, follow_redirects=False)
    assert r_start.status_code == 302, r_start.text
    loc = r_start.headers["location"]
    parsed = urllib.parse.urlsplit(loc)
    qs = urllib.parse.parse_qs(parsed.query)
    state = qs["state"][0]

    r_cb = browser.get(
        f"/auth/callback?code=mock_code_{gh_id}&state={urllib.parse.quote(state)}",
        headers=origin_hdr,
        follow_redirects=False,
    )
    return r_start, r_cb, state, captured


def test_oauth_authorize_url_cookies_and_state_validation(site_env, monkeypatch):
    origin = site_env["origin_hdr"]
    with site_env["new_browser"]() as b1:
        r = b1.get("/auth/login", headers=origin, follow_redirects=False)
        assert r.status_code == 302
        loc = r.headers["location"]
        parsed = urllib.parse.urlsplit(loc)
        qs = urllib.parse.parse_qs(parsed.query)
        assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == "https://github.com/login/oauth/authorize"
        assert qs["redirect_uri"] == ["https://draft.bertholomus.com/auth/callback"]
        assert qs["allow_signup"] == ["false"]
        assert "scope" not in qs  # read-only identity, no extra scopes
        assert len(qs["state"][0]) >= 32

        # __Host- cookie flags on state cookie
        sc = _extract_set_cookies(r)
        state_hdr = [c for c in sc if c.startswith(f"{config.OAUTH_STATE_COOKIE_NAME}=")]
        assert len(state_hdr) == 1
        low = state_hdr[0].lower()
        assert "secure" in low and "httponly" in low and "samesite=lax" in low and "path=/" in low
        assert "domain=" not in low
        state = qs["state"][0]

        # Missing code/state -> 400
        assert b1.get("/auth/callback", headers=origin, follow_redirects=False).status_code == 400
        assert b1.get(f"/auth/callback?state={state}", headers=origin, follow_redirects=False).status_code == 400

        # Bad state -> 400
        assert b1.get("/auth/callback?code=abc&state=totally_wrong", headers=origin, follow_redirects=False).status_code == 400

        # Another browser without the state cookie cannot use browser 1's state -> 400 (and burns state!)
        with site_env["new_browser"]() as b2:
            r_nocookie = b2.get(f"/auth/callback?code=abc&state={state}", headers=origin, follow_redirects=False)
            assert r_nocookie.status_code == 400

    # Replay check: valid login succeeds ONCE, replaying the exact same state+cookie fails with 400
    with site_env["new_browser"]() as b3:
        r_start, r_cb, state, captured = _do_mock_login(
            b3, origin, monkeypatch, gh_id=311723197, gh_login="bertholomus"
        )
        assert r_cb.status_code == 302
        assert captured["redirect_uri"] == "https://draft.bertholomus.com/auth/callback"
        assert captured["client_id"] == "Iv1.testclientid1234"
        # Grab original state cookie value and try to replay callback
        r_replay = b3.get(
            f"/auth/callback?code=mock_code_311723197&state={urllib.parse.quote(state)}",
            headers=origin,
            follow_redirects=False,
        )
        assert r_replay.status_code == 400


def test_real_http_code_exchange_against_mock_github_server(site_env, monkeypatch):
    """Exercise the actual urllib HTTP code-exchange & /user lookup against a local mock GitHub HTTP server."""
    seen: dict[str, Any] = {}

    class MockGitHubHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            seen["token_form"] = urllib.parse.parse_qs(raw)
            body = json.dumps({"access_token": "gho_mock_token_12345", "token_type": "bearer"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            seen["user_auth"] = self.headers.get("Authorization")
            body = json.dumps({"id": 512240, "login": "mmastrac", "name": "Matt Mastracci"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = http.server.HTTPServer(("127.0.0.1", 0), MockGitHubHandler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        monkeypatch.setattr(config, "GITHUB_TOKEN_URL", f"http://127.0.0.1:{port}/login/oauth/access_token")
        monkeypatch.setattr(config, "GITHUB_USER_API_URL", f"http://127.0.0.1:{port}/user")
        origin = site_env["origin_hdr"]
        with site_env["new_browser"]() as b:
            r1 = b.get("/auth/login", headers=origin, follow_redirects=False)
            assert r1.status_code == 302
            state = urllib.parse.parse_qs(urllib.parse.urlsplit(r1.headers["location"]).query)["state"][0]
            r2 = b.get(f"/auth/callback?code=real_http_code&state={state}", headers=origin, follow_redirects=False)
            assert r2.status_code == 302
            assert r2.headers["location"] == "https://draft.bertholomus.com/"
            assert seen["token_form"]["code"] == ["real_http_code"]
            assert seen["token_form"]["redirect_uri"] == ["https://draft.bertholomus.com/auth/callback"]
            assert seen["user_auth"] == "Bearer gho_mock_token_12345"
            me = b.get("/auth/me", headers=origin).json()
            assert me["authenticated"] is True
            assert me["login"] == "mmastrac"
            assert me["github_id"] == 512240
            assert me["approved"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_open_redirect_protection(site_env, monkeypatch):
    origin = site_env["origin_hdr"]
    bad_targets = [
        "https://evil.example/phish",
        "http://evil.example",
        "//evil.example/phish",
        "///evil.example",
        "/\\evil.example",
        "javascript:alert(1)",
        " /admin",
        "/auth/logout",
        "/\r\nSet-Cookie: hacked=1",
    ]
    for target in bad_targets:
        with site_env["new_browser"]() as b:
            _, r_cb, _, _ = _do_mock_login(
                b, origin, monkeypatch, gh_id=311723197, gh_login="bertholomus", next_url=target
            )
            assert r_cb.status_code == 302
            assert r_cb.headers["location"] == "https://draft.bertholomus.com/", f"target={target!r} -> {r_cb.headers['location']}"

    # Valid same-site relative paths ARE preserved
    with site_env["new_browser"]() as b:
        _, r_cb, _, _ = _do_mock_login(
            b, origin, monkeypatch, gh_id=311723197, gh_login="bertholomus", next_url="/#keys"
        )
        assert r_cb.status_code == 302
        assert r_cb.headers["location"] == "https://draft.bertholomus.com/#keys"


def test_preapproved_mappings_preserve_existing_keys_strikes_tiers_and_history(site_env, monkeypatch):
    client = site_env["client"]
    adm = site_env["adm_hdr"]
    origin = site_env["origin_hdr"]
    pub = site_env["pub_hdr"]

    # Seed existing contributor rows BEFORE OAuth login (simulating live DB pre-GHLOGIN state)
    k_matt, hdr_matt = site_env["mint_user"]("mmastrac", tiers=["tier-label"], is_owner=False)
    k_chuck, hdr_chuck = site_env["mint_user"]("ck2084", tiers=["tier-label"], is_owner=False)
    k_steve, hdr_steve = site_env["mint_user"]("coffeedev", tiers=["tier-label"], is_owner=False)

    # Give ck2084 1 strike and a verified slot history to confirm preservation
    assert client.post("/api/admin/contributors", headers=adm, json={"login": "ck2084", "strikes": 1}).status_code == 200
    assert client.post("/api/admin/runners", headers=adm, json={"commit_id": "drs-label-ext-v1.0.0", "kind": "drafter-label"}).status_code == 200
    assert client.post(
        "/api/admin/jobs",
        headers=adm,
        json={
            "job": "labels-prod",
            "title": "Labels",
            "job_kind": "drafter-label",
            "slots": [{"slot_id": "s0", "row_key": "b0", "col_key": "code", "profile_id": "bit280", "state": "open"}],
        },
    ).status_code == 200
    assert client.post(
        "/api/admin/jobs",
        headers=adm,
        json={
            "job": "train-r1",
            "title": "Train",
            "job_kind": "drafter-train",
            "slots": [{"slot_id": "t0", "row_key": "r1", "col_key": "bit280", "profile_id": "bit280", "state": "open"}],
        },
    ).status_code == 200

    # Re-run init_db() migration as happens at container deploy time
    db.init_db()

    # Existing keys still work!
    for hdr, expected_login in ((hdr_matt, "mmastrac"), (hdr_chuck, "ck2084"), (hdr_steve, "coffeedev")):
        me = client.get("/api/me", headers=hdr)
        assert me.status_code == 200 and me.json()["login"] == expected_login

    # Log in each of the 3 pre-approved GitHub accounts
    for gh_id, gh_login, site_login, old_hdr in (
        (512240, "mmastrac", "mmastrac", hdr_matt),
        (268695057, "chuck-ads", "ck2084", hdr_chuck),
        (137969418, "coffee-the-dev", "coffeedev", hdr_steve),
    ):
        with site_env["new_browser"]() as b:
            _, r_cb, _, _ = _do_mock_login(b, origin, monkeypatch, gh_id=gh_id, gh_login=gh_login)
            assert r_cb.status_code == 302 and r_cb.headers["location"] == "https://draft.bertholomus.com/"
            auth_me = b.get("/auth/me", headers=origin).json()
            assert auth_me["authenticated"] is True
            assert auth_me["login"] == site_login
            assert auth_me["display_login"] == gh_login
            assert auth_me["github_id"] == gh_id
            assert auth_me["approved"] is True
            assert auth_me["is_owner"] is False
            assert auth_me["allowed_kinds"] == ["drafter-label"]
            if site_login == "ck2084":
                assert auth_me["strikes"] == 1

            # Existing key appears in key panel and keeps working until revoked!
            keys_resp = b.get("/api/keys", headers=pub).json()
            assert len(keys_resp["keys"]) == 1
            legacy_key_id = keys_resp["keys"][0]["key_id"]
            assert client.get("/api/me", headers=old_hdr).status_code == 200

            # Revoking the existing key immediately invalidates it!
            r_rev = b.delete(
                f"/api/keys/{legacy_key_id}",
                headers={**pub, "X-CSRF-Token": auth_me["csrf_token"]},
            )
            assert r_rev.status_code == 200
            assert client.get("/api/me", headers=old_hdr).status_code == 401

            # Minting a fresh key works for drafter-label ONLY (403 on drafter-train!)
            r_new = b.post(
                "/api/keys",
                headers={**pub, "X-CSRF-Token": auth_me["csrf_token"]},
                json={"label": "self-service"},
            )
            assert r_new.status_code == 201
            new_key = r_new.json()["api_key_shown_once"]
            new_hdr = {**pub, "Authorization": f"Bearer {new_key}"}
            assert client.get("/api/me", headers=new_hdr).status_code == 200
            r_tr = client.post(
                "/api/claims",
                headers=new_hdr,
                json={"runner": "drs-label-ext-v1.0.0", "job": "train-r1", "slot_id": "t0", "request": f"req_{site_login}"},
            )
            assert r_tr.status_code == 403


def test_github_id_keying_survives_rename_and_blocks_username_hijack(site_env, monkeypatch):
    origin = site_env["origin_hdr"]
    pub = site_env["pub_hdr"]
    adm = site_env["adm_hdr"]
    client = site_env["client"]

    # Chuck (268695057) signs in, mints a key, then renames GitHub login to chuck-renamed
    with site_env["new_browser"]() as b:
        _do_mock_login(b, origin, monkeypatch, gh_id=268695057, gh_login="chuck-ads", gh_name="Chuck")
        me1 = b.get("/auth/me", headers=origin).json()
        assert me1["login"] == "ck2084" and me1["display_login"] == "chuck-ads"
        k1 = b.post("/api/keys", headers={**pub, "X-CSRF-Token": me1["csrf_token"]}, json={"label": "rig"}).json()["api_key_shown_once"]

        # Same numeric id, renamed login
        _do_mock_login(b, origin, monkeypatch, gh_id=268695057, gh_login="chuck-renamed", gh_name="Chuck R")
        me2 = b.get("/auth/me", headers=origin).json()
        assert me2["login"] == "ck2084"
        assert me2["display_login"] == "chuck-renamed"
        assert me2["github_login"] == "chuck-renamed"
        # Key still works and /api/me reflects new display login!
        api_me = client.get("/api/me", headers={**pub, "Authorization": f"Bearer {k1}"}).json()
        assert api_me["login"] == "ck2084"
        assert api_me["display_login"] == "chuck-renamed"

    # Now a DIFFERENT GitHub account grabs the abandoned username "chuck-ads" -> MUST be pending, NOT ck2084!
    with site_env["new_browser"]() as b_attacker:
        _, r_cb, _, _ = _do_mock_login(b_attacker, origin, monkeypatch, gh_id=999888777, gh_login="chuck-ads")
        assert r_cb.status_code == 302 and r_cb.headers["location"] == "https://draft.bertholomus.com/waiting"
        att_me = b_attacker.get("/auth/me", headers=origin).json()
        assert att_me["approved"] is False
        assert att_me["status"] == "pending"
        assert att_me["login"] != "ck2084"
        assert b_attacker.get("/api/keys", headers=pub).status_code == 403
        # Confirm ck2084 still belongs to 268695057
        reqs = client.get("/api/admin/access-requests", headers=adm).json()
        ck = next(r for r in reqs if r["login"] == "ck2084")
        assert ck["github_id"] == 268695057 and ck["github_login"] == "chuck-renamed"


def test_pending_approval_deny_xss_and_admin_view(site_env, monkeypatch):
    origin = site_env["origin_hdr"]
    pub = site_env["pub_hdr"]
    adm = site_env["adm_hdr"]
    client = site_env["client"]

    xss_name = '<script>alert("xss")</script><img src=x onerror=alert(1)>'
    with site_env["new_browser"]() as b_stranger:
        _, r_cb, _, _ = _do_mock_login(
            b_stranger, origin, monkeypatch, gh_id=424242, gh_login="new-contrib", gh_name=xss_name
        )
        assert r_cb.status_code == 302
        assert r_cb.headers["location"] == "https://draft.bertholomus.com/waiting"

        # Plain waiting page escapes XSS payload completely
        r_wait = b_stranger.get("/waiting", headers=origin)
        assert r_wait.status_code == 200
        assert "Waiting for approval" in r_wait.text
        assert "<script>alert" not in r_wait.text
        assert "&lt;script&gt;alert" in r_wait.text

        # Pending user has no keys, cannot mint keys, cannot access /admin or /api/admin/*
        me = b_stranger.get("/auth/me", headers=origin).json()
        assert me["status"] == "pending" and me["approved"] is False
        assert b_stranger.get("/api/keys", headers=pub).status_code == 403
        assert b_stranger.post("/api/keys", headers={**pub, "X-CSRF-Token": me["csrf_token"]}, json={}).status_code == 403
        assert b_stranger.get("/admin", headers=origin).status_code == 403
        assert b_stranger.get("/api/admin/access-requests", headers=pub).status_code == 403
        assert b_stranger.post("/api/admin/access-requests/new-contrib/approve", headers={**pub, "X-CSRF-Token": me["csrf_token"]}).status_code == 403

        # Pending user is NOT listed on public /api/contributors
        pub_contribs = client.get("/api/contributors", headers=pub).json()
        assert all(c["login"] != "new-contrib" for c in pub_contribs)

        # Visible to owner via admin API (existing admin bearer)
        pending = client.get("/api/admin/access-requests?status=pending", headers=adm).json()
        assert any(p["login"] == "new-contrib" and p["github_id"] == 424242 for p in pending)

        # Owner (bertholomus) signs in via GitHub -> CAN view /admin and approve via CSRF-protected session
        with site_env["new_browser"]() as b_owner:
            _do_mock_login(b_owner, origin, monkeypatch, gh_id=311723197, gh_login="bertholomus")
            own_me = b_owner.get("/auth/me", headers=origin).json()
            assert own_me["is_owner"] is True and own_me["can_admin_requests"] is True
            assert b_owner.get("/admin", headers=origin).status_code == 200
            assert b_owner.get("/api/admin/access-requests", headers=pub).status_code == 200

            # Owner worker API key minted in key panel MUST NOT have admin API access (S12 separation!)
            own_k = b_owner.post("/api/keys", headers={**pub, "X-CSRF-Token": own_me["csrf_token"]}, json={"label": "wrk"}).json()["api_key_shown_once"]
            assert client.get("/api/admin/access-requests", headers={**pub, "Authorization": f"Bearer {own_k}"}).status_code == 403
            assert client.post("/api/admin/runners", headers={**pub, "Authorization": f"Bearer {own_k}"}, json={"commit_id": "x"}).status_code == 403

            # CSRF required for owner session approval
            assert b_owner.post("/api/admin/access-requests/new-contrib/approve", headers=pub).status_code == 403
            r_app = b_owner.post(
                "/api/admin/access-requests/new-contrib/approve",
                headers={**pub, "X-CSRF-Token": own_me["csrf_token"]},
                json={},
            )
            assert r_app.status_code == 200 and r_app.json()["approved"] is True

        # Now stranger can mint a key!
        r_mk = b_stranger.post("/api/keys", headers={**pub, "X-CSRF-Token": me["csrf_token"]}, json={"label": "first"})
        assert r_mk.status_code == 201
        stranger_key = r_mk.json()["api_key_shown_once"]
        assert client.get("/api/me", headers={**pub, "Authorization": f"Bearer {stranger_key}"}).status_code == 200

        # Admin denies stranger -> immediately revokes keys AND kills session!
        r_deny = client.post("/api/admin/access-requests/new-contrib/deny", headers=adm, json={"note": "revoked"})
        assert r_deny.status_code == 200 and r_deny.json()["revoked_keys"] == 1 and r_deny.json()["revoked_sessions"] == 1
        assert client.get("/api/me", headers={**pub, "Authorization": f"Bearer {stranger_key}"}).status_code == 401
        assert b_stranger.get("/auth/me", headers=origin).json()["authenticated"] is False
        assert b_stranger.get("/api/keys", headers=pub).status_code == 401


def test_keys_panel_cap_rotate_revoke_csrf_and_no_secret_leaks(site_env, monkeypatch):
    origin = site_env["origin_hdr"]
    pub = site_env["pub_hdr"]
    client = site_env["client"]

    with site_env["new_browser"]() as b:
        _do_mock_login(b, origin, monkeypatch, gh_id=512240, gh_login="mmastrac")
        me = b.get("/auth/me", headers=origin).json()
        csrf = me["csrf_token"]
        csrf_hdr = {**pub, "X-CSRF-Token": csrf}

        # CSRF enforced on create, rotate, revoke, and cross-origin rejected
        assert b.post("/api/keys", headers=pub, json={"label": "a"}).status_code == 403
        assert b.post("/api/keys", headers={**pub, "X-CSRF-Token": "wrong"}, json={"label": "a"}).status_code == 403
        assert b.post("/api/keys", headers={**csrf_hdr, "Origin": "https://evil.example"}, json={"label": "a"}).status_code == 403

        # Create key 1 & key 2
        k1_resp = b.post("/api/keys", headers=csrf_hdr, json={"label": "rig-1"})
        assert k1_resp.status_code == 201
        k1 = k1_resp.json()["api_key_shown_once"]
        k1_id = k1_resp.json()["key_id"]
        assert k1.startswith("drs_wrk_")

        k2_resp = b.post("/api/keys", headers=csrf_hdr, json={"label": "rig-2"})
        assert k2_resp.status_code == 201
        k2 = k2_resp.json()["api_key_shown_once"]
        k2_id = k2_resp.json()["key_id"]

        # 3rd live key rejected with 409 (cap=2)
        assert b.post("/api/keys", headers=csrf_hdr, json={"label": "rig-3"}).status_code == 409

        # Using k1 updates last_used_at
        assert client.get("/api/me", headers={**pub, "Authorization": f"Bearer {k1}"}).status_code == 200
        listed = b.get("/api/keys", headers=pub).json()
        assert len(listed["keys"]) == 2
        k1_meta = next(k for k in listed["keys"] if k["key_id"] == k1_id)
        assert k1_meta["prefix"] == k1[:14]
        assert k1_meta["last_used_at"] is not None
        # Raw keys NEVER appear in GET /api/keys, HTML, or /api/audit
        assert k1 not in json.dumps(listed) and k2 not in json.dumps(listed)
        assert k1 not in b.get("/", headers=origin).text
        audit_dump = json.dumps(client.get("/api/audit?limit=50", headers=pub).json())
        assert k1 not in audit_dump and k2 not in audit_dump and k1[:14] not in audit_dump

        # CSRF enforced on rotate & revoke
        assert b.post(f"/api/keys/{k1_id}/rotate", headers=pub, json={}).status_code == 403
        assert b.delete(f"/api/keys/{k2_id}", headers=pub).status_code == 403

        # Rotate k1 while at cap=2 -> old k1 immediately 401, new k1b works!
        rot = b.post(f"/api/keys/{k1_id}/rotate", headers=csrf_hdr, json={"label": "rig-1b"})
        assert rot.status_code == 200
        k1b = rot.json()["api_key_shown_once"]
        assert client.get("/api/me", headers={**pub, "Authorization": f"Bearer {k1}"}).status_code == 401
        assert client.get("/api/me", headers={**pub, "Authorization": f"Bearer {k1b}"}).status_code == 200

        # Another user cannot rotate or revoke mmastrac's key (404)
        with site_env["new_browser"]() as b_other:
            _do_mock_login(b_other, origin, monkeypatch, gh_id=137969418, gh_login="coffee-the-dev")
            o_csrf = b_other.get("/auth/me", headers=origin).json()["csrf_token"]
            assert b_other.delete(f"/api/keys/{k2_id}", headers={**pub, "X-CSRF-Token": o_csrf}).status_code == 404
            assert b_other.post(f"/api/keys/{k2_id}/rotate", headers={**pub, "X-CSRF-Token": o_csrf}).status_code == 404

        # Revoke k2 -> stops working at once!
        assert b.delete(f"/api/keys/{k2_id}", headers=csrf_hdr).status_code == 200
        assert client.get("/api/me", headers={**pub, "Authorization": f"Bearer {k2}"}).status_code == 401
        # Now creating another key succeeds (back to 2 live keys)
        assert b.post("/api/keys", headers=csrf_hdr, json={"label": "rig-3"}).status_code == 201


def test_session_fixation_idle_expiry_logout_and_quarantine(site_env, monkeypatch):
    origin = site_env["origin_hdr"]
    pub = site_env["pub_hdr"]
    adm = site_env["adm_hdr"]
    client = site_env["client"]

    with site_env["new_browser"]() as b:
        # 1. Login 1 -> capture session cookie & verify __Host- attributes
        _, r_cb1, _, _ = _do_mock_login(b, origin, monkeypatch, gh_id=512240, gh_login="mmastrac")
        sc1 = [c for c in _extract_set_cookies(r_cb1) if c.startswith(f"{config.SESSION_COOKIE_NAME}=")]
        assert len(sc1) == 1
        low = sc1[0].lower()
        assert "secure" in low and "httponly" in low and "samesite=lax" in low and "path=/" in low
        assert "domain=" not in low
        sess1 = b.cookies.get(config.SESSION_COOKIE_NAME)
        assert sess1

        # 2. Session fixation defense: re-logging in rotates session id AND invalidates sess1!
        _, r_cb2, _, _ = _do_mock_login(b, origin, monkeypatch, gh_id=512240, gh_login="mmastrac")
        sess2 = b.cookies.get(config.SESSION_COOKIE_NAME)
        assert sess2 and sess2 != sess1
        with site_env["new_browser"]() as b_old:
            b_old.cookies.set(config.SESSION_COOKIE_NAME, sess1)
            assert b_old.get("/auth/me", headers=origin).json()["authenticated"] is False
            assert b_old.get("/api/keys", headers=pub).status_code == 401

        # 3. Quarantined user with active session CANNOT mint keys (403)
        me = b.get("/auth/me", headers=origin).json()
        assert client.post("/api/admin/contributors", headers=adm, json={"login": "mmastrac", "quarantined": True}).status_code == 200
        assert b.post("/api/keys", headers={**pub, "X-CSRF-Token": me["csrf_token"]}, json={}).status_code == 403
        assert b.get("/api/keys", headers=pub).status_code == 403
        # Un-quarantine for remaining checks
        assert client.post("/api/admin/contributors", headers=adm, json={"login": "mmastrac", "quarantined": False}).status_code == 200

        # 4. Idle expiry (30d): expire session in DB -> rejected immediately
        with db.tx(immediate=True) as conn:
            conn.execute("UPDATE sessions SET expires_at = ? WHERE contributor_login = 'mmastrac'", (db.utc_ms() - 1000,))
        assert b.get("/auth/me", headers=origin).json()["authenticated"] is False
        assert b.get("/api/keys", headers=pub).status_code == 401

        # 5. Fresh login + CSRF-checked logout invalidates session server-side
        _do_mock_login(b, origin, monkeypatch, gh_id=512240, gh_login="mmastrac")
        sess3 = b.cookies.get(config.SESSION_COOKIE_NAME)
        csrf3 = b.get("/auth/me", headers=origin).json()["csrf_token"]
        assert b.post("/auth/logout", headers=origin).status_code == 403  # missing CSRF
        r_lo = b.post("/auth/logout", headers={**origin, "X-CSRF-Token": csrf3})
        assert r_lo.status_code == 200
        with site_env["new_browser"]() as b_replay:
            b_replay.cookies.set(config.SESSION_COOKIE_NAME, sess3)
            assert b_replay.get("/auth/me", headers=origin).json()["authenticated"] is False


def test_origin_auth_s4_enforced_on_new_routes(site_env):
    client = site_env["client"]
    for path in ("/auth/login", "/auth/callback", "/auth/me", "/waiting", "/admin", "/api/keys", "/api/admin/access-requests"):
        r = client.get(path, headers={"X-Quant": "1"}, follow_redirects=False)
        assert r.status_code == 403, f"path {path} bypassed S4 origin auth: {r.status_code}"


def test_unconfigured_oauth_form_logout_and_callback_error_xss(site_env, monkeypatch):
    origin = site_env["origin_hdr"]
    pub = site_env["pub_hdr"]

    # 1. Callback error param escapes XSS payload
    with site_env["new_browser"]() as b:
        r_err = b.get(
            "/auth/callback?error=access_denied&error_description=%3Cscript%3Ealert(1)%3C/script%3E",
            headers=origin,
            follow_redirects=False,
        )
        assert r_err.status_code == 400
        assert "<script>alert(1)</script>" not in r_err.text
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in r_err.text

    # 2. Form-based POST /auth/logout (used on /waiting page without JS)
    with site_env["new_browser"]() as b:
        _do_mock_login(b, origin, monkeypatch, gh_id=555001, gh_login="form-logout-user")
        sess_tok = b.cookies.get(config.SESSION_COOKIE_NAME)
        me = b.get("/auth/me", headers=origin).json()
        assert me["authenticated"] is True and me["status"] == "pending"
        # Wrong form CSRF -> 403
        r_bad = b.post(
            "/auth/logout",
            headers={**origin, "Content-Type": "application/x-www-form-urlencoded"},
            content="csrf_token=wrong",
            follow_redirects=False,
        )
        assert r_bad.status_code == 403
        # Valid form CSRF -> 303 to https://draft.bertholomus.com/ and kills session
        r_ok = b.post(
            "/auth/logout",
            headers={**origin, "Content-Type": "application/x-www-form-urlencoded"},
            content=f"csrf_token={urllib.parse.quote(me['csrf_token'])}",
            follow_redirects=False,
        )
        assert r_ok.status_code == 303
        assert r_ok.headers["location"] == "https://draft.bertholomus.com/"
        with site_env["new_browser"]() as b_check:
            b_check.cookies.set(config.SESSION_COOKIE_NAME, sess_tok)
            assert b_check.get("/auth/me", headers=origin).json()["authenticated"] is False

    # 3. Missing github_oauth.json degrades cleanly (503 on /auth/login, oauth_configured=False on /auth/me)
    (site_env["secrets_dir"] / "github_oauth.json").unlink()
    with site_env["new_browser"]() as b:
        me = b.get("/auth/me", headers=origin).json()
        assert me["authenticated"] is False and me["oauth_configured"] is False
        r_503 = b.get("/auth/login", headers=origin, follow_redirects=False)
        assert r_503.status_code == 503
