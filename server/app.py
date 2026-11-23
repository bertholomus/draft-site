"""Hardened FastAPI coordinator for draft.bertholomus.com."""
from __future__ import annotations

import asyncio
import collections
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse

from server import config, db, ui

SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_READY_STATES = frozenset(("not_ready", "open", "disabled", "rejected"))
PUBLIC_AUDIT_REDACT_KEYS = frozenset(
    (
        "host_tag",
        "artifact_id",
        "quarantine_id",
        "quarantined_artifact",
        "claim_id",
        "reason",
        "gate_metrics",
        "expected",
        "actual",
        "key_id",
        "old_key_id",
        "new_key_id",
        "prefix",
        "api_key",
        "api_key_shown_once",
        "session_id",
        "csrf_token",
        "github_id",
        "state",
        "code",
        "access_token",
        "client_secret",
    )
)

CSP_STRICT = (
    "default-src 'self'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "script-src 'self'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)

_VERIFY_SEMAPHORE: asyncio.Semaphore | None = None
_CLAIM_RATE_BUCKETS: dict[str, collections.deque[float]] = {}


def _get_verify_semaphore() -> asyncio.Semaphore:
    global _VERIFY_SEMAPHORE
    if _VERIFY_SEMAPHORE is None:
        _VERIFY_SEMAPHORE = asyncio.Semaphore(max(1, config.MAX_CONCURRENT_VERIFIES))
    return _VERIFY_SEMAPHORE


async def _background_sweeper() -> None:
    while True:
        try:
            await asyncio.sleep(60)
            with db.tx(immediate=True) as conn:
                db.sweep_expired_claims(conn)
        except asyncio.CancelledError:
            break
        except Exception:
            pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    task = asyncio.create_task(_background_sweeper())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="draft.bertholomus.com", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


def _validate_id(val: str, name: str = "id") -> str:
    if not val or not SAFE_ID_RE.match(val):
        raise HTTPException(status_code=400, detail=f"invalid {name}")
    return val


def _extract_bearer(req: Request) -> str | None:
    auth = req.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        tok = auth[7:].strip()
        return tok if tok else None
    return None


def _require_x_quant(req: Request) -> None:
    if req.headers.get("x-quant") != "1":
        raise HTTPException(status_code=400, detail="missing required header X-Quant: 1")


def _is_owner(user: sqlite3.Row) -> bool:
    if bool(user["is_admin"]):
        return True
    try:
        if bool(user["is_owner"]):
            return True
    except (KeyError, IndexError):
        pass
    return str(user["login"]) in config.OWNER_LOGINS


def _user_status(user: sqlite3.Row) -> str:
    denied = False
    try:
        denied = bool(user["denied"])
    except (KeyError, IndexError):
        pass
    if denied:
        return "denied"
    if bool(user["quarantined"]):
        return "quarantined"
    if not _is_owner(user) and int(user["strikes"]) >= config.MAX_STRIKES_BEFORE_QUARANTINE:
        return "quarantined"
    if not bool(user["approved"]):
        return "pending"
    return "approved"


def _display_login(user: sqlite3.Row) -> str:
    try:
        gl = user["github_login"]
        if gl:
            return str(gl)
    except (KeyError, IndexError):
        pass
    return str(user["login"])


def _set_strict_cookie(response: Response, name: str, value: str, max_age: int) -> None:
    """Set a __Host- compliant cookie (Secure, HttpOnly, SameSite=Lax, Path=/, no Domain)."""
    response.set_cookie(
        key=name,
        value=value,
        max_age=max_age,
        path="/",
        secure=True,
        httponly=True,
        samesite="lax",
    )


def _clear_strict_cookie(response: Response, name: str) -> None:
    response.set_cookie(
        key=name,
        value="",
        max_age=0,
        expires=0,
        path="/",
        secure=True,
        httponly=True,
        samesite="lax",
    )


def _sanitize_return_to(raw: str | None) -> str:
    """Allow ONLY same-site relative paths (blocks protocol-relative, backslash, scheme, control chars)."""
    if not raw or not isinstance(raw, str) or len(raw) > 256:
        return "/"
    if raw != raw.strip():
        return "/"
    if "\\" in raw or any(ord(c) < 0x20 or ord(c) == 0x7F for c in raw):
        return "/"
    if not raw.startswith("/") or raw.startswith("//"):
        return "/"
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme or parsed.netloc:
        return "/"
    if not parsed.path.startswith("/") or parsed.path.startswith("//"):
        return "/"
    if parsed.path.startswith("/auth/"):
        return "/"
    return raw


def _public_redirect(path: str) -> str:
    safe_path = _sanitize_return_to(path)
    return f"{config.PUBLIC_URL}{safe_path}"


def _check_origin_header(req: Request) -> None:
    """If Origin header is present on state-changing browser requests, require same-site match."""
    origin = (req.headers.get("origin") or "").strip().rstrip("/")
    if not origin:
        return
    allowed = {config.PUBLIC_URL.rstrip("/")}
    if not req.headers.get("x-forwarded-host"):
        host_hdr = (req.headers.get("host") or "").strip()
        hostname = host_hdr.split(":")[0].lower()
        if hostname in ("127.0.0.1", "localhost", "testserver"):
            allowed.add(f"{req.url.scheme}://{host_hdr}".rstrip("/"))
    if origin not in allowed:
        raise HTTPException(status_code=403, detail="cross-origin request rejected")


def _require_auth(
    conn: sqlite3.Connection,
    req: Request,
    *,
    admin: bool = False,
    touch_last_seen: bool = False,
) -> sqlite3.Row:
    tok = _extract_bearer(req)
    user = db.lookup_auth(conn, tok, touch_last_seen=touch_last_seen)
    if user is None:
        raise HTTPException(status_code=401, detail="valid Authorization: Bearer <key> required")
    if admin:
        if not user["is_admin"]:
            raise HTTPException(status_code=403, detail="admin credential required")
        if config.ADMIN_LOCAL_ONLY and req.headers.get("x-forwarded-for"):
            raise HTTPException(status_code=403, detail="admin API disabled over public proxy")
        return user
    status = _user_status(user)
    if status != "approved":
        raise HTTPException(status_code=403, detail=f"contributor account is not approved ({status})")
    return user


def _require_session(
    conn: sqlite3.Connection,
    req: Request,
    *,
    require_approved: bool = True,
    require_csrf: bool = False,
    require_owner_admin: bool = False,
    csrf_override: str | None = None,
    touch: bool = False,
) -> sqlite3.Row:
    cookie_val = req.cookies.get(config.SESSION_COOKIE_NAME)
    sess = db.lookup_session(conn, cookie_val, touch=touch)
    if sess is None:
        raise HTTPException(status_code=401, detail="active GitHub session required")
    if require_csrf:
        _check_origin_header(req)
        provided = (csrf_override if csrf_override is not None else req.headers.get("x-csrf-token", "")).strip()
        expected = str(sess["csrf_token"] or "")
        if not provided or not expected or not hmac.compare_digest(provided, expected):
            raise HTTPException(status_code=403, detail="missing or invalid CSRF token")
    if require_approved and _user_status(sess) != "approved":
        raise HTTPException(status_code=403, detail=f"account is {_user_status(sess)}; approval required")
    if require_owner_admin:
        if not _is_owner(sess) or _user_status(sess) != "approved" or sess["github_id"] is None:
            raise HTTPException(status_code=403, detail="owner session required")
    return sess


def _require_admin_or_owner_session(
    conn: sqlite3.Connection,
    req: Request,
    *,
    require_csrf_for_session: bool = False,
    touch_last_seen: bool = False,
) -> sqlite3.Row:
    """Admin Bearer (is_admin=1) OR signed-in owner GitHub browser session (+ CSRF on writes)."""
    if _extract_bearer(req) is not None:
        return _require_auth(conn, req, admin=True, touch_last_seen=touch_last_seen)
    if req.cookies.get(config.SESSION_COOKIE_NAME):
        return _require_session(
            conn,
            req,
            require_approved=True,
            require_csrf=require_csrf_for_session,
            require_owner_admin=True,
            touch=touch_last_seen,
        )
    raise HTTPException(status_code=401, detail="admin credential or owner session required")


def exchange_github_code(
    *,
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> dict[str, Any]:
    """Server-side GitHub OAuth code exchange + read-only /user lookup (no secrets logged)."""
    form_data = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
        }
    ).encode("utf-8")
    tok_req = urllib.request.Request(
        config.GITHUB_TOKEN_URL,
        data=form_data,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "draft-site-oauth",
        },
    )
    with urllib.request.urlopen(tok_req, timeout=15) as resp:
        tok_payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(tok_payload, dict) or not tok_payload.get("access_token"):
        err = tok_payload.get("error_description") or tok_payload.get("error") if isinstance(tok_payload, dict) else "invalid token response"
        raise RuntimeError(f"GitHub token exchange failed: {err}")
    access_token = str(tok_payload["access_token"])

    user_req = urllib.request.Request(
        config.GITHUB_USER_API_URL,
        method="GET",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "draft-site-oauth",
        },
    )
    with urllib.request.urlopen(user_req, timeout=15) as resp:
        u = json.loads(resp.read().decode("utf-8"))
    if not isinstance(u, dict):
        raise RuntimeError("invalid GitHub user payload")
    gh_id = u.get("id")
    gh_login = str(u.get("login") or "").strip()
    gh_name = str(u.get("name") or "").strip() or None
    if not isinstance(gh_id, int) or isinstance(gh_id, bool) or gh_id <= 0 or not gh_login:
        raise RuntimeError("GitHub user response missing valid numeric id or login")
    return {"id": gh_id, "login": gh_login, "name": gh_name}


def _safe_under(base: Path, target: Path) -> Path:
    res = target.resolve()
    if not res.is_relative_to(base.resolve()):
        raise HTTPException(status_code=400, detail="invalid storage path")
    return res


def _check_claim_rate_limit(login: str) -> None:
    limit = config.CLAIM_RATE_LIMIT_PER_MIN
    if limit <= 0:
        return
    now = time.monotonic()
    q = _CLAIM_RATE_BUCKETS.setdefault(login, collections.deque())
    cutoff = now - 60.0
    while q and q[0] < cutoff:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(status_code=429, detail="claim rate limit exceeded; retry shortly")
    q.append(now)


def _sanitize_audit_detail(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _sanitize_audit_detail(v)
            for k, v in obj.items()
            if k not in PUBLIC_AUDIT_REDACT_KEYS
        }
    if isinstance(obj, list):
        return [_sanitize_audit_detail(v) for v in obj]
    return obj


async def _read_body_capped(request: Request, max_bytes: int, err_msg: str) -> bytes:
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > max_bytes:
                raise HTTPException(status_code=413, detail=err_msg)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid Content-Length header")
    buf = bytearray()
    async for chunk in request.stream():
        if not chunk:
            continue
        if len(buf) + len(chunk) > max_bytes:
            raise HTTPException(status_code=413, detail=err_msg)
        buf.extend(chunk)
    return bytes(buf)


def _allowed_kinds_for_user(user: sqlite3.Row) -> set[str]:
    if user["is_admin"]:
        return set(config.KIND_TO_TIER.keys())
    tiers = set(json.loads(user["tiers_json"] or "[]"))
    permitted = {k for k, t in config.KIND_TO_TIER.items() if t in tiers}
    if not _is_owner(user):
        permitted &= set(config.NON_OWNER_ALLOWED_KINDS)
    return permitted


@app.middleware("http")
async def security_and_origin_middleware(request: Request, call_next):
    # S4: Default-deny origin auth on ALL non-/healthz paths regardless of Host header.
    if config.REQUIRE_ORIGIN_AUTH and request.url.path != "/healthz":
        peer = request.client.host if request.client else ""
        is_loopback = peer in ("127.0.0.1", "::1")
        if not (is_loopback and config.ALLOW_LOOPBACK_NO_ORIGIN_AUTH):
            expected = config.load_origin_auth_secret()
            provided = request.headers.get("x-origin-auth", "")
            if not expected or not hmac.compare_digest(provided, expected):
                return JSONResponse(
                    status_code=403,
                    content={"error": "direct origin access forbidden; use https://draft.bertholomus.com"},
                )

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    p = request.url.path
    if p.startswith("/api/") or p.startswith("/auth/") or p in ("/", "/waiting", "/admin"):
        response.headers["Cache-Control"] = "no-store"
    return response


# ---------------------------------------------------------------------------
# Health, Static Assets, Legal & Landing UI
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "service": "draft-site", "utc": db.utc_iso()}


@app.get("/NOTICE", response_class=PlainTextResponse)
def notice_file() -> str:
    for p in (Path("/app/NOTICE"), Path(__file__).resolve().parents[1] / "NOTICE"):
        if p.is_file():
            return p.read_text(encoding="utf-8")
    return "Apache-2.0 NOTICE — see repository NOTICE file."


@app.get("/LICENSE", response_class=PlainTextResponse)
def license_file() -> str:
    for p in (Path("/app/LICENSE"), Path(__file__).resolve().parents[1] / "LICENSE"):
        if p.is_file():
            return p.read_text(encoding="utf-8")
    return "Apache-2.0 LICENSE — see repository LICENSE file."


@app.get("/static/app.js")
def static_app_js() -> FileResponse:
    js_path = Path(__file__).resolve().parent / "static" / "app.js"
    if not js_path.is_file():
        raise HTTPException(status_code=404, detail="asset not found")
    return FileResponse(
        path=str(js_path),
        media_type="application/javascript; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/", response_class=HTMLResponse)
def index_page() -> HTMLResponse:
    return HTMLResponse(
        content=ui.render_index_html(),
        headers={"Cache-Control": "no-store", "Content-Security-Policy": CSP_STRICT},
    )


@app.get("/waiting", response_class=HTMLResponse)
def waiting_page(request: Request) -> Response:
    with db.ro() as conn:
        sess = db.lookup_session(conn, request.cookies.get(config.SESSION_COOKIE_NAME), touch=False)
        if sess is None:
            return RedirectResponse(url=_public_redirect("/"), status_code=302)
        status = _user_status(sess)
        if status == "approved":
            return RedirectResponse(url=_public_redirect("/"), status_code=302)
        html_body = ui.render_waiting_html(
            github_login=_display_login(sess),
            github_name=sess["github_name"],
            status=status,
            csrf_token=str(sess["csrf_token"]),
        )
    return HTMLResponse(
        content=html_body,
        headers={"Cache-Control": "no-store", "Content-Security-Policy": CSP_STRICT},
    )


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request) -> Response:
    with db.ro() as conn:
        sess = db.lookup_session(conn, request.cookies.get(config.SESSION_COOKIE_NAME), touch=False)
        if sess is None:
            return RedirectResponse(url=f"{config.PUBLIC_URL}/auth/login?next=/admin", status_code=302)
        if not _is_owner(sess) or _user_status(sess) != "approved" or sess["github_id"] is None:
            return HTMLResponse(
                content=ui.render_auth_error_html("Forbidden", "Owner GitHub account required to view admin panel."),
                status_code=403,
                headers={"Cache-Control": "no-store", "Content-Security-Policy": CSP_STRICT},
            )
    return HTMLResponse(
        content=ui.render_admin_html(),
        headers={"Cache-Control": "no-store", "Content-Security-Policy": CSP_STRICT},
    )


# ---------------------------------------------------------------------------
# GitHub OAuth Web Flow & Browser Sessions
# ---------------------------------------------------------------------------

@app.get("/auth/login")
def auth_login(request: Request) -> Response:
    oauth_cfg = config.load_github_oauth_config()
    if oauth_cfg is None:
        return HTMLResponse(
            content=ui.render_auth_error_html(
                "GitHub login unavailable",
                "GitHub OAuth is not configured on the server yet. Please try again shortly.",
            ),
            status_code=503,
            headers={"Cache-Control": "no-store", "Content-Security-Policy": CSP_STRICT},
        )
    client_id, _ = oauth_cfg
    raw_next = request.query_params.get("next") or request.query_params.get("return_to")
    return_to = _sanitize_return_to(raw_next)
    state = secrets.token_urlsafe(32)
    browser_nonce = secrets.token_urlsafe(32)
    with db.tx(immediate=True) as conn:
        db.create_oauth_state(conn, state, browser_nonce, return_to)

    redirect_uri = f"{config.PUBLIC_URL}/auth/callback"
    params = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "allow_signup": "false",
        }
    )
    authorize_url = f"{config.GITHUB_AUTHORIZE_URL}?{params}"
    resp = RedirectResponse(url=authorize_url, status_code=302, headers={"Cache-Control": "no-store"})
    _set_strict_cookie(resp, config.OAUTH_STATE_COOKIE_NAME, browser_nonce, config.OAUTH_STATE_TTL_SEC)
    return resp


@app.get("/auth/callback")
def auth_callback(request: Request) -> Response:
    oauth_cfg = config.load_github_oauth_config()
    if oauth_cfg is None:
        resp = HTMLResponse(
            content=ui.render_auth_error_html("GitHub login unavailable", "GitHub OAuth is not configured on the server."),
            status_code=503,
            headers={"Cache-Control": "no-store", "Content-Security-Policy": CSP_STRICT},
        )
        _clear_strict_cookie(resp, config.OAUTH_STATE_COOKIE_NAME)
        return resp

    err = (request.query_params.get("error") or "").strip()
    if err:
        err_desc = (request.query_params.get("error_description") or err)[:200]
        resp = HTMLResponse(
            content=ui.render_auth_error_html("GitHub login cancelled", f"GitHub returned an error: {err_desc}"),
            status_code=400,
            headers={"Cache-Control": "no-store", "Content-Security-Policy": CSP_STRICT},
        )
        _clear_strict_cookie(resp, config.OAUTH_STATE_COOKIE_NAME)
        return resp

    code = (request.query_params.get("code") or "").strip()
    state = (request.query_params.get("state") or "").strip()
    browser_nonce = (request.cookies.get(config.OAUTH_STATE_COOKIE_NAME) or "").strip()
    if not code or not state or not browser_nonce:
        resp = HTMLResponse(
            content=ui.render_auth_error_html("Invalid login state", "Missing OAuth code, state, or browser state cookie."),
            status_code=400,
            headers={"Cache-Control": "no-store", "Content-Security-Policy": CSP_STRICT},
        )
        _clear_strict_cookie(resp, config.OAUTH_STATE_COOKIE_NAME)
        return resp

    with db.tx(immediate=True) as conn:
        return_to = db.consume_oauth_state(conn, state, browser_nonce)
    if return_to is None:
        resp = HTMLResponse(
            content=ui.render_auth_error_html("Invalid login state", "OAuth state is invalid, expired, or already used."),
            status_code=400,
            headers={"Cache-Control": "no-store", "Content-Security-Policy": CSP_STRICT},
        )
        _clear_strict_cookie(resp, config.OAUTH_STATE_COOKIE_NAME)
        return resp

    client_id, client_secret = oauth_cfg
    redirect_uri = f"{config.PUBLIC_URL}/auth/callback"
    try:
        gh_user = exchange_github_code(
            code=code,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
        )
    except Exception:
        resp = HTMLResponse(
            content=ui.render_auth_error_html("GitHub sign-in failed", "Could not verify your GitHub login. Please try again."),
            status_code=502,
            headers={"Cache-Control": "no-store", "Content-Security-Policy": CSP_STRICT},
        )
        _clear_strict_cookie(resp, config.OAUTH_STATE_COOKIE_NAME)
        return resp

    old_session = request.cookies.get(config.SESSION_COOKIE_NAME)
    with db.tx(immediate=True) as conn:
        user, created_pending = db.resolve_or_create_github_contributor(
            conn,
            github_id=int(gh_user["id"]),
            github_login=str(gh_user["login"]),
            github_name=gh_user.get("name"),
        )
        # Rotate session id at login (invalidating any pre-existing session cookie -> defeats fixation)
        new_session_id, _csrf = db.create_session(conn, str(user["login"]), old_session_token=old_session)
        status = _user_status(user)
        action = "github_access_requested" if created_pending else "github_login"
        db.log_audit(conn, str(user["login"]), action, status, detail={})

    dest = _public_redirect(return_to) if status == "approved" else _public_redirect("/waiting")
    resp = RedirectResponse(url=dest, status_code=302, headers={"Cache-Control": "no-store"})
    _clear_strict_cookie(resp, config.OAUTH_STATE_COOKIE_NAME)
    _set_strict_cookie(resp, config.SESSION_COOKIE_NAME, new_session_id, config.SESSION_IDLE_TTL_SEC)
    return resp


@app.get("/auth/me")
def auth_me(request: Request) -> dict[str, Any]:
    oauth_configured = config.load_github_oauth_config() is not None
    touch_info: tuple[str, int] | None = None
    with db.ro() as conn:
        sess = db.lookup_session(conn, request.cookies.get(config.SESSION_COOKIE_NAME), touch=False)
        if sess is None:
            return {"authenticated": False, "oauth_configured": oauth_configured}
        touch_info = (str(sess["session_hash"]), int(sess["session_last_seen_at"]))
        status = _user_status(sess)
        is_own = _is_owner(sess)
        vcount = int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM slots WHERE uploaded_by=? AND state='verified'",
                (sess["login"],),
            ).fetchone()["n"]
        )
        resp = {
            "authenticated": True,
            "oauth_configured": oauth_configured,
            "status": status,
            "login": str(sess["login"]),
            "display_login": _display_login(sess),
            "github_id": sess["github_id"],
            "github_login": sess["github_login"],
            "github_name": sess["github_name"],
            "approved": bool(sess["approved"]),
            "denied": bool(sess["denied"]),
            "quarantined": bool(sess["quarantined"]),
            "is_owner": is_own,
            "can_admin_requests": bool(is_own and status == "approved" and sess["github_id"] is not None),
            "tiers": json.loads(sess["tiers_json"] or "[]"),
            "allowed_kinds": sorted(_allowed_kinds_for_user(sess)) if status == "approved" else [],
            "strikes": int(sess["strikes"]),
            "verified_slots": vcount,
            "max_live_keys": config.MAX_LIVE_KEYS_PER_ACCOUNT,
            "csrf_token": str(sess["csrf_token"]),
        }
    if touch_info is not None:
        db.touch_session_if_stale(*touch_info)
    return resp


@app.post("/auth/logout")
async def auth_logout(request: Request) -> Response:
    _check_origin_header(request)
    ctype = (request.headers.get("content-type") or "").lower()
    csrf_from_body: str | None = None
    is_form = "application/x-www-form-urlencoded" in ctype
    if is_form:
        raw = await _read_body_capped(request, 8192, "request body too large")
        parsed = urllib.parse.parse_qs(raw.decode("utf-8", errors="replace"), keep_blank_values=True)
        vals = parsed.get("csrf_token") or []
        csrf_from_body = vals[0] if vals else ""
    elif "application/json" in ctype:
        raw = await _read_body_capped(request, 8192, "request body too large")
        if raw.strip():
            try:
                obj = json.loads(raw.decode("utf-8"))
                if isinstance(obj, dict) and "csrf_token" in obj:
                    csrf_from_body = str(obj["csrf_token"] or "")
            except Exception:
                pass

    cookie_val = request.cookies.get(config.SESSION_COOKIE_NAME)
    if cookie_val:
        with db.tx(immediate=True) as conn:
            sess = db.lookup_session(conn, cookie_val, touch=False)
            if sess is not None:
                provided = (request.headers.get("x-csrf-token") or csrf_from_body or "").strip()
                expected = str(sess["csrf_token"] or "")
                if not provided or not expected or not hmac.compare_digest(provided, expected):
                    raise HTTPException(status_code=403, detail="missing or invalid CSRF token")
                db.delete_session(conn, cookie_val)
                db.log_audit(conn, str(sess["login"]), "github_logout", "ok", detail={})

    if is_form:
        resp: Response = RedirectResponse(url=_public_redirect("/"), status_code=303, headers={"Cache-Control": "no-store"})
    else:
        resp = JSONResponse(content={"ok": True}, headers={"Cache-Control": "no-store"})
    _clear_strict_cookie(resp, config.SESSION_COOKIE_NAME)
    return resp


# ---------------------------------------------------------------------------
# Self-Service Worker API Keys (Session + CSRF + X-Quant: 1)
# ---------------------------------------------------------------------------

@app.get("/api/keys")
def list_my_keys(request: Request) -> dict[str, Any]:
    _require_x_quant(request)
    with db.ro() as conn:
        sess = _require_session(conn, request, require_approved=True, require_csrf=False, touch=False)
        rows = db.list_active_api_keys(conn, str(sess["login"]))
        tiers = json.loads(sess["tiers_json"] or "[]")
        allowed_kinds = sorted(_allowed_kinds_for_user(sess))
        return {
            "login": str(sess["login"]),
            "display_login": _display_login(sess),
            "max_live_keys": config.MAX_LIVE_KEYS_PER_ACCOUNT,
            "tiers": tiers,
            "allowed_kinds": allowed_kinds,
            "keys": [
                {
                    "key_id": r["key_id"],
                    "prefix": r["key_prefix"],
                    "label": r["label"],
                    "tiers": tiers,
                    "allowed_kinds": allowed_kinds,
                    "created_at": r["created_at"],
                    "created_utc": db.utc_iso(r["created_at"]),
                    "last_used_at": r["last_used_at"],
                    "last_used_utc": db.utc_iso(r["last_used_at"]) if r["last_used_at"] is not None else None,
                }
                for r in rows
            ],
        }


@app.post("/api/keys", status_code=201)
async def create_my_key(request: Request) -> JSONResponse:
    _require_x_quant(request)
    label = ""
    raw = await _read_body_capped(request, 8192, "request body too large")
    if raw.strip():
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            raise HTTPException(status_code=400, detail="valid JSON body required")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON object required")
        label = str(body.get("label") or "").strip()[:64]

    with db.tx(immediate=True) as conn:
        sess = _require_session(conn, request, require_approved=True, require_csrf=True, touch=True)
        try:
            meta, raw_key = db.create_api_key(conn, str(sess["login"]), label=label)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
        tiers = json.loads(sess["tiers_json"] or "[]")
        allowed_kinds = sorted(_allowed_kinds_for_user(sess))
        db.log_audit(conn, str(sess["login"]), "api_key_created", "ok", detail={"key_id": meta["key_id"], "prefix": meta["prefix"]})
        return JSONResponse(
            status_code=201,
            content={
                **meta,
                "tiers": tiers,
                "allowed_kinds": allowed_kinds,
                "api_key_shown_once": raw_key,
            },
            headers={"Cache-Control": "no-store"},
        )


@app.post("/api/keys/{key_id}/rotate")
async def rotate_my_key(request: Request, key_id: str) -> JSONResponse:
    _require_x_quant(request)
    _validate_id(key_id, "key_id")
    label: str | None = None
    raw = await _read_body_capped(request, 8192, "request body too large")
    if raw.strip():
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            raise HTTPException(status_code=400, detail="valid JSON body required")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON object required")
        if "label" in body and body["label"] is not None:
            label = str(body["label"]).strip()[:64]

    with db.tx(immediate=True) as conn:
        sess = _require_session(conn, request, require_approved=True, require_csrf=True, touch=True)
        res = db.rotate_api_key(conn, str(sess["login"]), key_id, label=label)
        if res is None:
            raise HTTPException(status_code=404, detail="active API key not found")
        meta, raw_key = res
        tiers = json.loads(sess["tiers_json"] or "[]")
        allowed_kinds = sorted(_allowed_kinds_for_user(sess))
        db.log_audit(
            conn,
            str(sess["login"]),
            "api_key_rotated",
            "ok",
            detail={"old_key_id": key_id, "new_key_id": meta["key_id"], "prefix": meta["prefix"]},
        )
        return JSONResponse(
            status_code=200,
            content={
                **meta,
                "tiers": tiers,
                "allowed_kinds": allowed_kinds,
                "api_key_shown_once": raw_key,
            },
            headers={"Cache-Control": "no-store"},
        )


@app.delete("/api/keys/{key_id}")
def revoke_my_key(request: Request, key_id: str) -> dict[str, Any]:
    _require_x_quant(request)
    _validate_id(key_id, "key_id")
    with db.tx(immediate=True) as conn:
        sess = _require_session(conn, request, require_approved=True, require_csrf=True, touch=True)
        revoked = db.revoke_api_key(conn, str(sess["login"]), key_id)
        if revoked is None:
            raise HTTPException(status_code=404, detail="active API key not found")
        db.log_audit(
            conn,
            str(sess["login"]),
            "api_key_revoked",
            "ok",
            detail={"key_id": key_id, "prefix": revoked["prefix"]},
        )
        return {"ok": True, "revoked": True, "key_id": key_id}


# ---------------------------------------------------------------------------
# Public Read API (X-Quant: 1) — strictly read-only (zero SQLite write locks)
# ---------------------------------------------------------------------------

@app.get("/api/encoders")
@app.get("/api/runners")
def list_runners(request: Request) -> list[dict[str, Any]]:
    _require_x_quant(request)
    with db.ro() as conn:
        rows = conn.execute(
            "SELECT commit_id, kind, image_digest, note, added_at FROM runners WHERE active=1 ORDER BY added_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]


@app.get("/api/jobs")
def list_jobs(request: Request) -> list[dict[str, Any]]:
    _require_x_quant(request)
    with db.ro() as conn:
        jobs = conn.execute("SELECT * FROM jobs ORDER BY created_at ASC").fetchall()
        agg_rows = conn.execute(
            """SELECT job, state, COUNT(*) AS n, SUM(harvested) AS harv, MAX(uploaded_at) AS last_up
               FROM slots GROUP BY job, state"""
        ).fetchall()
        contrib_rows = conn.execute(
            """SELECT job, COUNT(DISTINCT uploaded_by) AS c
               FROM slots WHERE state='verified' AND uploaded_by IS NOT NULL
               GROUP BY job"""
        ).fetchall()
        by_job_state: dict[str, dict[str, int]] = {}
        by_job_harv: dict[str, int] = {}
        by_job_last: dict[str, int | None] = {}
        for r in agg_rows:
            jb = r["job"]
            by_job_state.setdefault(jb, {})[r["state"]] = int(r["n"])
            by_job_harv[jb] = by_job_harv.get(jb, 0) + int(r["harv"] or 0)
            lu = r["last_up"]
            if lu is not None and (by_job_last.get(jb) is None or lu > by_job_last[jb]):
                by_job_last[jb] = lu
        by_job_contribs = {r["job"]: int(r["c"]) for r in contrib_rows}

        out: list[dict[str, Any]] = []
        for j in jobs:
            jb = j["job"]
            counts = by_job_state.get(jb, {})
            total_slots = sum(counts.values())
            cfg = json.loads(j["config_json"] or "{}")
            out.append(
                {
                    "job": jb,
                    "title": j["title"],
                    "job_kind": j["job_kind"],
                    "model": j["model"],
                    "version": j["version"],
                    "active_round": j["active_round"],
                    "slots": total_slots,
                    "uploaded": counts.get("uploaded", 0) + counts.get("verified", 0),
                    "verified": counts.get("verified", 0),
                    "pending_verify": counts.get("uploaded", 0),
                    "claimed": counts.get("claimed", 0),
                    "open": counts.get("open", 0),
                    "not_ready": counts.get("not_ready", 0),
                    "rejected": counts.get("rejected", 0),
                    "disabled": counts.get("disabled", 0),
                    "harvested": by_job_harv.get(jb, 0),
                    "contributors": by_job_contribs.get(jb, 0),
                    "last_upload": by_job_last.get(jb),
                    "weights": {
                        "hf": j["release_repo"],
                        "revision": j["release_revision"],
                    },
                    "config": cfg,
                }
            )
        return out


@app.get("/api/grid")
def get_grid(request: Request, job: str, offset: int = 0, limit: int = 500) -> dict[str, Any]:
    _require_x_quant(request)
    _validate_id(job, "job")
    off = max(0, int(offset))
    lim = max(1, min(1000, int(limit)))
    with db.ro() as conn:
        me = db.lookup_auth(conn, _extract_bearer(request), touch_last_seen=False)
        if me is None:
            me = db.lookup_session(conn, request.cookies.get(config.SESSION_COOKIE_NAME), touch=False)
        my_login = me["login"] if me else None
        j = conn.execute("SELECT * FROM jobs WHERE job=?", (job,)).fetchone()
        if j is None:
            raise HTTPException(status_code=404, detail=f"unknown job '{job}'")
        all_jobs = [r["job"] for r in conn.execute("SELECT job FROM jobs ORDER BY created_at ASC").fetchall()]
        display_map = {
            r["login"]: (r["github_login"] or r["login"])
            for r in conn.execute("SELECT login, github_login FROM contributors").fetchall()
        }
        total_slots = int(conn.execute("SELECT COUNT(*) AS n FROM slots WHERE job=?", (job,)).fetchone()["n"])
        slot_rows = conn.execute(
            """SELECT * FROM slots WHERE job=?
               ORDER BY round_num ASC, row_key ASC, col_key ASC, slot_id ASC
               LIMIT ? OFFSET ?""",
            (job, lim, off),
        ).fetchall()
        rows_axis: list[str] = []
        cols_axis: list[str] = []
        ready_rows: set[str] = set()
        slots_out: list[dict[str, Any]] = []
        eval_matrix: dict[str, dict[str, Any]] = {}

        for s in slot_rows:
            rk, ck = s["row_key"], s["col_key"]
            if rk not in rows_axis:
                rows_axis.append(rk)
            if ck not in cols_axis:
                cols_axis.append(ck)
            if s["state"] != "not_ready":
                ready_rows.add(rk)
            holder = s["uploaded_by"] if s["state"] in ("uploaded", "verified", "rejected") else s["claimed_by"]
            holder_disp = display_map.get(holder, holder) if holder else None
            mine = bool(my_login and holder == my_login)
            metrics = json.loads(s["metrics_json"] or "{}")
            plan = json.loads(s["plan_json"] or "{}")
            slots_out.append(
                {
                    "slot_id": s["slot_id"],
                    "row": rk,
                    "col": ck,
                    "round": s["round_num"],
                    "profile": s["profile_id"],
                    "state": s["state"],
                    "contributor_login": holder,
                    "display_login": holder_disp,
                    "mine": mine,
                    "depends_on": f"{s['depends_on_job']}/{s['depends_on_slot']}" if s["depends_on_job"] else None,
                    "upload_bytes": s["upload_bytes"],
                    "upload_sha256": s["upload_sha256"],
                    "verify_status": s["verify_status"],
                    "verified_at": s["verified_at"],
                    "metrics": {k: v for k, v in metrics.items() if k not in ("host_tag", "engines", "st_meta")},
                    "plan_summary": {
                        k: plan[k]
                        for k in ("codec", "profile_id", "checkpoint_rev", "local_steps", "n_prompts", "tap_layers")
                        if k in plan
                    },
                }
            )
            if j["job_kind"] == "drafter-eval" and s["state"] == "verified" and metrics:
                ckpt_rev = str(metrics.get("checkpoint_rev") or rk)
                prof = str(metrics.get("profile_id") or ck)
                eval_matrix.setdefault(ckpt_rev, {})[prof] = {
                    "b8_ref_tau": metrics.get("b8_ref_tau"),
                    "b16_ref_tau": metrics.get("b16_ref_tau"),
                    "roll16_tau": metrics.get("roll16_tau"),
                    "ce8": metrics.get("ce8"),
                    "top1": metrics.get("top1"),
                    "by_cat": metrics.get("by_cat"),
                    "pred_sha256": metrics.get("pred_sha256"),
                    "contributor_login": holder_disp or holder,
                    "quorum": metrics.get("quorum"),
                }

        rounds = [
            {
                "round": r["round_num"],
                "base_ckpt_name": r["base_ckpt_name"],
                "base_ckpt_sha256": r["base_ckpt_sha256"],
                "quorum": r["quorum"],
                "total_slots": r["total_slots"],
                "status": r["status"],
                "agg_method": r["agg_method"],
                "agg_ckpt_name": r["agg_ckpt_name"],
                "agg_ckpt_sha256": r["agg_ckpt_sha256"],
                "agg_metrics": json.loads(r["agg_metrics_json"] or "{}"),
                "completed_at": r["completed_at"],
            }
            for r in conn.execute("SELECT * FROM rounds WHERE job=? ORDER BY round_num ASC", (job,)).fetchall()
        ]

        return {
            "job": j["job"],
            "jobs": all_jobs,
            "job_kind": j["job_kind"],
            "model": j["model"],
            "release": j["release_repo"],
            "revision": j["release_revision"],
            "title": j["title"],
            "active_round": j["active_round"],
            "axes": {"rows": rows_axis, "cols": cols_axis},
            "ready": sorted(ready_rows),
            "config": json.loads(j["config_json"] or "{}"),
            "rounds": rounds,
            "eval_matrix": eval_matrix,
            "total_slots": total_slots,
            "offset": off,
            "limit": lim,
            "slots": slots_out,
        }


@app.get("/api/contributors")
def list_contributors(request: Request) -> list[dict[str, Any]]:
    """Public contributor leaderboard (S9: omits internal status, strikes, tiers, last_seen_at, pending requests)."""
    _require_x_quant(request)
    with db.ro() as conn:
        contribs = conn.execute(
            """SELECT login, github_login FROM contributors
               WHERE is_admin=0
                 AND (approved=1 OR EXISTS (SELECT 1 FROM slots WHERE uploaded_by=contributors.login AND state='verified'))
               ORDER BY login ASC"""
        ).fetchall()
        out: list[dict[str, Any]] = []
        for c in contribs:
            rows = conn.execute(
                """SELECT j.job_kind, s.state, s.upload_bytes
                   FROM slots s JOIN jobs j ON s.job = j.job
                   WHERE s.uploaded_by=?""",
                (c["login"],),
            ).fetchall()
            verified_total = 0
            rejected_total = 0
            by_kind: dict[str, int] = {}
            total_bytes = 0
            for r in rows:
                if r["state"] == "verified":
                    verified_total += 1
                    by_kind[r["job_kind"]] = by_kind.get(r["job_kind"], 0) + 1
                    total_bytes += int(r["upload_bytes"] or 0)
                elif r["state"] == "rejected":
                    rejected_total += 1
            audit_rej = conn.execute(
                "SELECT COUNT(*) AS n FROM audit_events WHERE actor=? AND action IN ('upload_rejected','verification_rejected')",
                (c["login"],),
            ).fetchone()["n"]
            out.append(
                {
                    "login": c["login"],
                    "display_login": c["github_login"] or c["login"],
                    "verified_slots": verified_total,
                    "rejected_attempts": max(rejected_total, int(audit_rej)),
                    "by_kind": by_kind,
                    "total_bytes_uploaded": total_bytes,
                }
            )
        out.sort(key=lambda x: (-x["verified_slots"], x["login"]))
        return out


@app.get("/api/audit")
def list_recent_audit(request: Request, limit: int = 50) -> list[dict[str, Any]]:
    """Public audit log (S5 & S9: strips artifact IDs, claim IDs, host tags, key/session metadata, and threshold oracles)."""
    _require_x_quant(request)
    lim = max(1, min(200, int(limit)))
    with db.ro() as conn:
        rows = conn.execute(
            "SELECT id, ts, actor, action, job, slot_id, status, detail_json FROM audit_events ORDER BY id DESC LIMIT ?",
            (lim,),
        ).fetchall()
        out = []
        for r in rows:
            det = _sanitize_audit_detail(json.loads(r["detail_json"] or "{}"))
            out.append(
                {
                    "id": r["id"],
                    "ts": r["ts"],
                    "utc": db.utc_iso(r["ts"]),
                    "actor": r["actor"],
                    "action": r["action"],
                    "job": r["job"],
                    "slot_id": r["slot_id"],
                    "status": r["status"],
                    "detail": det,
                }
            )
        return out


# ---------------------------------------------------------------------------
# Authenticated Contributor API (X-Quant: 1 + Authorization: Bearer <key>)
# ---------------------------------------------------------------------------

@app.get("/api/me")
def get_me(request: Request) -> dict[str, Any]:
    _require_x_quant(request)
    touch_args: tuple[str | None, str, int | None] | None = None
    with db.ro() as conn:
        tok = _extract_bearer(request)
        user = db.lookup_auth(conn, tok, touch_last_seen=False)
        if user is None:
            raise HTTPException(status_code=401, detail="valid Authorization: Bearer <key> required")
        touch_args = (user["auth_key_id"], str(user["login"]), user["auth_key_last_used_at"])
        vcount = conn.execute(
            "SELECT COUNT(*) AS n FROM slots WHERE uploaded_by=? AND state='verified'", (user["login"],)
        ).fetchone()["n"]
        active_claims = conn.execute(
            "SELECT COUNT(*) AS n FROM slots WHERE claimed_by=? AND state='claimed'", (user["login"],)
        ).fetchone()["n"]
        is_own = _is_owner(user)
        resp = {
            "login": user["login"],
            "display_login": _display_login(user),
            "github_id": user["github_id"],
            "github_login": user["github_login"],
            "approved": bool(user["approved"]),
            "denied": bool(user["denied"]),
            "admin": bool(user["is_admin"]),
            "owner": is_own,
            "quarantined": bool(user["quarantined"]),
            "tiers": json.loads(user["tiers_json"] or "[]"),
            "allowed_kinds": sorted(_allowed_kinds_for_user(user)),
            "claim_cap": config.MAX_ACTIVE_CLAIMS_OWNER if is_own else config.MAX_ACTIVE_CLAIMS_PER_LOGIN,
            "active_claims": int(active_claims),
            "strikes": int(user["strikes"]),
            "verified_slots": int(vcount),
        }
    if touch_args is not None:
        db.touch_key_usage_if_stale(*touch_args)
    return resp


def _base_url_for_request(req: Request) -> str:
    if not req.headers.get("x-forwarded-host"):
        host_hdr = (req.headers.get("host") or "").strip()
        hostname = host_hdr.split(":")[0].lower()
        if hostname in ("127.0.0.1", "localhost", "testserver"):
            return f"{req.url.scheme}://{host_hdr}"
    return config.PUBLIC_URL


def _format_claim_response(slot: sqlite3.Row, job_row: sqlite3.Row, *, base_url: str | None = None) -> dict[str, Any]:
    burl = (base_url or config.PUBLIC_URL).rstrip("/")
    inputs_raw = json.loads(slot["inputs_json"] or "[]")
    inputs_out = []
    for item in inputs_raw:
        entry = dict(item)
        if "artifact_id" in entry:
            entry["url"] = f"{burl}/api/artifacts/{entry['artifact_id']}"
        inputs_out.append(entry)
    max_bytes = config.MAX_UPLOAD_BYTES_BY_KIND.get(job_row["job_kind"], config.MAX_DIRECT_UPLOAD_BYTES)
    return {
        "claim_id": slot["claim_id"],
        "job": slot["job"],
        "job_kind": job_row["job_kind"],
        "slot_id": slot["slot_id"],
        "row_key": slot["row_key"],
        "col_key": slot["col_key"],
        "round": slot["round_num"],
        "profile_id": slot["profile_id"],
        "expires_at": slot["expires_at"],
        "expires_utc": db.utc_iso(slot["expires_at"]),
        "release": {"hf": job_row["release_repo"], "revision": job_row["release_revision"]},
        "inputs": inputs_out,
        "plan": json.loads(slot["plan_json"] or "{}"),
        "upload": {
            "direct_url": f"{burl}/api/uploads/direct/{slot['claim_id']}",
            "init_url": f"{burl}/api/uploads/init",
            "max_bytes": max_bytes,
            "max_direct_bytes": config.MAX_DIRECT_UPLOAD_BYTES,
            "min_chunk_bytes": config.MIN_CHUNK_BYTES,
            "chunk_bytes": config.MAX_CHUNK_BYTES,
        },
    }


@app.get("/api/claims")
def list_my_claims(request: Request) -> list[dict[str, Any]]:
    _require_x_quant(request)
    burl = _base_url_for_request(request)
    with db.tx(immediate=True) as conn:
        db.sweep_expired_claims(conn)
        user = _require_auth(conn, request, touch_last_seen=True)
        rows = conn.execute(
            "SELECT * FROM slots WHERE state='claimed' AND claimed_by=? ORDER BY claimed_at ASC",
            (user["login"],),
        ).fetchall()
        out = []
        for s in rows:
            j = conn.execute("SELECT * FROM jobs WHERE job=?", (s["job"],)).fetchone()
            out.append(_format_claim_response(s, j, base_url=burl))
        return out


@app.post("/api/claims", status_code=201)
async def create_claim(request: Request) -> JSONResponse:
    _require_x_quant(request)
    with db.ro() as conn:
        pre_user = _require_auth(conn, request)
        pre_login = str(pre_user["login"])
    _check_claim_rate_limit(pre_login)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="valid JSON body required")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body required")

    runner_commit = str(body.get("runner") or body.get("encoder") or "").strip()
    if not runner_commit:
        raise HTTPException(status_code=400, detail="runner (or encoder) commit_id required")
    req_id = str(body.get("request") or "").strip()
    if not req_id:
        raise HTTPException(status_code=400, detail="fresh request id required")
    req_job = body.get("job")
    req_slot = body.get("slot_id")
    req_profile = body.get("profile")
    req_round = body.get("round")
    host_tag = str(body.get("host_tag") or "").strip()[:64]
    capabilities = body.get("capabilities") or body.get("codebooks") or []

    if req_job is not None:
        _validate_id(str(req_job), "job")
    if req_slot is not None:
        _validate_id(str(req_slot), "slot_id")

    with db.tx(immediate=True) as conn:
        db.sweep_expired_claims(conn)
        user = _require_auth(conn, request, touch_last_seen=True)
        is_own = _is_owner(user)
        allowed_kinds = _allowed_kinds_for_user(user)

        runner = conn.execute("SELECT * FROM runners WHERE commit_id=? AND active=1", (runner_commit,)).fetchone()
        if runner is None:
            raise HTTPException(status_code=400, detail=f"runner commit '{runner_commit}' not in /api/runners")

        prev = conn.execute(
            "SELECT * FROM slots WHERE request_id=? AND claimed_by=? AND state='claimed'",
            (req_id, user["login"]),
        ).fetchone()
        if prev is not None:
            j = conn.execute("SELECT * FROM jobs WHERE job=?", (prev["job"],)).fetchone()
            return JSONResponse(status_code=201, content=_format_claim_response(prev, j, base_url=_base_url_for_request(request)))

        if req_job:
            jcheck = conn.execute("SELECT * FROM jobs WHERE job=?", (req_job,)).fetchone()
            if jcheck is None:
                raise HTTPException(status_code=404, detail=f"job '{req_job}' not found")
            jkind = str(jcheck["job_kind"])
            if not is_own and jkind not in config.NON_OWNER_ALLOWED_KINDS:
                raise HTTPException(status_code=403, detail=f"non-owner contributors may only claim {', '.join(sorted(config.NON_OWNER_ALLOWED_KINDS))} slots")
            if jkind not in allowed_kinds:
                raise HTTPException(status_code=403, detail=f"contributor tier does not permit '{jkind}'")

        if not allowed_kinds:
            raise HTTPException(status_code=403, detail="contributor has no permitted job kinds")

        active_claims = int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM slots WHERE state='claimed' AND claimed_by=?",
                (user["login"],),
            ).fetchone()["n"]
        )
        cap = config.MAX_ACTIVE_CLAIMS_OWNER if is_own else config.MAX_ACTIVE_CLAIMS_PER_LOGIN
        if active_claims >= cap:
            raise HTTPException(status_code=429, detail=f"active claim cap reached ({active_claims}/{cap})")

        if req_job and req_slot:
            slot = conn.execute("SELECT * FROM slots WHERE job=? AND slot_id=?", (req_job, req_slot)).fetchone()
            if slot is None:
                raise HTTPException(status_code=404, detail=f"slot {req_job}/{req_slot} not found")
            if slot["state"] != "open":
                raise HTTPException(status_code=409, detail=f"slot {req_slot} is {slot['state']} (taken or not ready)")
        else:
            sorted_kinds = sorted(allowed_kinds)
            if capabilities:
                kind_caps = [c for c in capabilities if str(c).startswith("drafter-")]
                if kind_caps:
                    filtered = [k for k in sorted_kinds if k in kind_caps]
                    if not filtered:
                        raise HTTPException(status_code=403, detail="requested capability kinds not permitted for account")
                    sorted_kinds = filtered
            placeholders = ",".join("?" for _ in sorted_kinds)
            q = f"SELECT s.* FROM slots s JOIN jobs j ON s.job=j.job WHERE s.state='open' AND j.job_kind IN ({placeholders})"
            params: list[Any] = list(sorted_kinds)
            if req_job:
                q += " AND s.job=?"
                params.append(req_job)
            if req_profile:
                q += " AND s.profile_id=?"
                params.append(str(req_profile))
            if req_round is not None:
                q += " AND s.round_num=?"
                params.append(int(req_round))
            q += " ORDER BY s.round_num ASC, s.row_key ASC, s.col_key ASC, s.slot_id ASC LIMIT 1"
            slot = conn.execute(q, params).fetchone()
            if slot is None:
                raise HTTPException(status_code=409, detail="no open slots matching filter")

        job_row = conn.execute("SELECT * FROM jobs WHERE job=?", (slot["job"],)).fetchone()
        jkind = str(job_row["job_kind"])
        if not is_own and jkind not in config.NON_OWNER_ALLOWED_KINDS:
            raise HTTPException(status_code=403, detail=f"non-owner contributors may only claim {', '.join(sorted(config.NON_OWNER_ALLOWED_KINDS))} slots")
        if jkind not in allowed_kinds:
            raise HTTPException(status_code=403, detail=f"contributor tier does not permit '{jkind}'")
        if runner["kind"] not in ("all", jkind):
            raise HTTPException(status_code=400, detail=f"runner kind '{runner['kind']}' cannot claim '{jkind}'")

        if jkind in ("drafter-eval", "drafter-pack"):
            dup = conn.execute(
                """SELECT 1 FROM slots
                   WHERE job=? AND row_key=? AND col_key=? AND slot_id != ?
                     AND state='verified' AND uploaded_by=? LIMIT 1""",
                (slot["job"], slot["row_key"], slot["col_key"], slot["slot_id"], user["login"]),
            ).fetchone()
            if dup is not None:
                raise HTTPException(status_code=409, detail="cannot claim duplicate quorum slot for cell already verified by your login")

        now = db.utc_ms()
        claim_id = "clm_" + secrets.token_hex(12)
        expires_at = now + config.DEFAULT_CLAIM_TTL_SEC * 1000
        cur = conn.execute(
            """UPDATE slots
               SET state='claimed', claim_id=?, claimed_by=?, claimed_at=?, expires_at=?, host_tag=?, request_id=?
               WHERE job=? AND slot_id=? AND state='open'""",
            (claim_id, user["login"], now, expires_at, host_tag or None, req_id, slot["job"], slot["slot_id"]),
        )
        if cur.rowcount != 1:
            raise HTTPException(status_code=409, detail="slot was claimed concurrently")
        updated = conn.execute("SELECT * FROM slots WHERE job=? AND slot_id=?", (slot["job"], slot["slot_id"])).fetchone()
        db.log_audit(
            conn,
            user["login"],
            "claim_created",
            "ok",
            slot["job"],
            slot["slot_id"],
            {"runner": runner_commit, "host_tag": host_tag},
        )
        return JSONResponse(status_code=201, content=_format_claim_response(updated, job_row, base_url=_base_url_for_request(request)))


@app.delete("/api/claims/{job}/{slot_id}")
def release_claim(request: Request, job: str, slot_id: str) -> dict[str, Any]:
    _require_x_quant(request)
    _validate_id(job, "job")
    _validate_id(slot_id, "slot_id")
    with db.tx(immediate=True) as conn:
        user = _require_auth(conn, request, touch_last_seen=True)
        slot = conn.execute("SELECT * FROM slots WHERE job=? AND slot_id=?", (job, slot_id)).fetchone()
        if slot is None:
            raise HTTPException(status_code=404, detail="slot not found")
        if slot["state"] != "claimed":
            raise HTTPException(status_code=409, detail=f"slot is {slot['state']}, not claimed")
        if slot["claimed_by"] != user["login"] and not user["is_admin"]:
            raise HTTPException(status_code=403, detail="you do not hold this claim")
        conn.execute(
            """UPDATE slots
               SET state='open', claim_id=NULL, claimed_by=NULL, claimed_at=NULL, expires_at=NULL, host_tag=NULL, request_id=NULL
               WHERE job=? AND slot_id=?""",
            (job, slot_id),
        )
        db.sweep_expired_uploads(conn)
        db.log_audit(conn, user["login"], "claim_released", "ok", job, slot_id, {})
        return {"released": True, "job": job, "slot_id": slot_id}


# ---------------------------------------------------------------------------
# Authenticated Artifact Download (S5: bound strictly to caller's active claim)
# ---------------------------------------------------------------------------

@app.get("/api/artifacts/{artifact_id}")
def download_artifact(request: Request, artifact_id: str) -> FileResponse:
    _require_x_quant(request)
    _validate_id(artifact_id, "artifact_id")
    with db.ro() as conn:
        user = _require_auth(conn, request)
        art = conn.execute("SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
        if art is None:
            raise HTTPException(status_code=404, detail="artifact not found")
        if not user["is_admin"]:
            if art["kind"] == "quarantine":
                raise HTTPException(status_code=403, detail="artifact is not downloadable")
            claim_rows = conn.execute(
                "SELECT inputs_json FROM slots WHERE state='claimed' AND claimed_by=?",
                (user["login"],),
            ).fetchall()
            allowed_ids: set[str] = set()
            for cr in claim_rows:
                for inp in json.loads(cr["inputs_json"] or "[]"):
                    aid = inp.get("artifact_id")
                    if aid:
                        allowed_ids.add(str(aid))
            if artifact_id not in allowed_ids:
                raise HTTPException(status_code=403, detail="artifact is not bound to an active claim held by your account")
        fpath = _safe_under(config.DATA_DIR, config.DATA_DIR / art["rel_path"])
        if not fpath.is_file():
            raise HTTPException(status_code=404, detail="artifact file missing on disk")
        return FileResponse(
            path=str(fpath),
            media_type=art["content_type"] or "application/octet-stream",
            filename=art["filename"],
            headers={
                "X-SHA256": art["sha256"],
                "ETag": f'"{art["sha256"]}"',
                "Cache-Control": "no-store",
            },
        )


# ---------------------------------------------------------------------------
# Upload Processing & Isolated Subprocess Verification (S6)
# ---------------------------------------------------------------------------

def _gather_verify_context(
    conn: sqlite3.Connection,
    user: sqlite3.Row,
    slot: sqlite3.Row,
    job_row: sqlite3.Row,
) -> dict[str, Any]:
    kind = str(job_row["job_kind"])
    slot_plan = json.loads(slot["plan_json"] or "{}")
    inputs = json.loads(slot["inputs_json"] or "[]")
    prompt_path: str | None = None
    peer_metrics: list[dict[str, Any]] = []

    if kind == "drafter-label":
        for inp in inputs:
            if inp.get("name") in ("prompt_shard", "prompts") and inp.get("artifact_id"):
                art_row = conn.execute(
                    "SELECT rel_path FROM artifacts WHERE artifact_id=?", (inp["artifact_id"],)
                ).fetchone()
                if art_row:
                    prompt_path = str(_safe_under(config.DATA_DIR, config.DATA_DIR / art_row["rel_path"]))
                    break
    elif kind == "drafter-eval":
        peers = conn.execute(
            """SELECT metrics_json FROM slots
               WHERE job=? AND state='verified' AND slot_id != ?
                 AND profile_id=? AND row_key=? AND COALESCE(uploaded_by, '') != ?""",
            (slot["job"], slot["slot_id"], slot["profile_id"], slot["row_key"], user["login"]),
        ).fetchall()
        peer_metrics = [json.loads(p["metrics_json"] or "{}") for p in peers]
    elif kind == "drafter-pack":
        peers = conn.execute(
            """SELECT metrics_json FROM slots
               WHERE job=? AND state='verified' AND slot_id != ?
                 AND row_key=? AND col_key=? AND COALESCE(uploaded_by, '') != ?""",
            (slot["job"], slot["slot_id"], slot["row_key"], slot["col_key"], user["login"]),
        ).fetchall()
        peer_metrics = [json.loads(p["metrics_json"] or "{}") for p in peers]

    require_owner_gate = str(user["login"]) not in config.LABEL_GATE_TRUSTED_LOGINS
    return {
        "kind": kind,
        "slot_plan": slot_plan,
        "prompt_path": prompt_path,
        "peer_metrics": peer_metrics,
        "require_owner_gate": require_owner_gate,
        "max_bytes": config.MAX_UPLOAD_BYTES_BY_KIND.get(kind, config.MAX_DIRECT_UPLOAD_BYTES),
    }


async def _run_verify_subprocess(payload: dict[str, Any]) -> dict[str, Any]:
    """Run Phase-A assembly/hashing/verification in an isolated bounded worker subprocess."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "DRAFT_DATA_DIR": str(config.DATA_DIR),
        "DRAFT_SECRETS_DIR": str(config.SECRETS_DIR),
        "DRAFT_VERIFY_MEM_BYTES": str(config.VERIFY_SUBPROCESS_MEM_LIMIT_BYTES),
    }
    sem = _get_verify_semaphore()
    async with sem:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "server.verify_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(json.dumps(payload).encode("utf-8")),
                timeout=config.VERIFY_SUBPROCESS_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            await proc.wait()
            return {
                "ok": False,
                "error_status": 422,
                "error_code": "failed_verify_timeout",
                "error_detail": "verification timed out in worker subprocess",
            }
        if proc.returncode != 0:
            err_tail = stderr.decode("utf-8", errors="replace")[-300:]
            return {
                "ok": False,
                "error_status": 422,
                "error_code": "failed_verify_worker",
                "error_detail": f"verification worker exited with rc={proc.returncode}: {err_tail}",
            }
        try:
            return json.loads(stdout.decode("utf-8"))
        except Exception as e:
            return {
                "ok": False,
                "error_status": 422,
                "error_code": "failed_verify_worker",
                "error_detail": f"invalid verifier output: {e}",
            }


def _apply_strike_and_quarantine(conn: sqlite3.Connection, login: str, *, is_owner: bool) -> bool:
    """Increment strikes and auto-quarantine non-owner contributors at MAX_STRIKES_BEFORE_QUARANTINE."""
    if is_owner:
        conn.execute("UPDATE contributors SET strikes = strikes + 1 WHERE login=?", (login,))
        return False
    conn.execute(
        """UPDATE contributors
           SET strikes = strikes + 1,
               quarantined = CASE WHEN is_admin = 0 AND strikes + 1 >= ? THEN 1 ELSE quarantined END
           WHERE login=?""",
        (config.MAX_STRIKES_BEFORE_QUARANTINE, login),
    )
    row = conn.execute("SELECT quarantined FROM contributors WHERE login=?", (login,)).fetchone()
    return bool(row and row["quarantined"])


def _commit_upload_verdict(
    conn: sqlite3.Connection,
    user: sqlite3.Row,
    slot: sqlite3.Row,
    staged_file: Path,
    filename: str,
    worker_res: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Commit verification outcome in a short BEGIN IMMEDIATE transaction."""
    now = db.utc_ms()
    art_id = "art_" + secrets.token_hex(12)
    safe_fn = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)[:96] or "artifact.bin"
    is_own = _is_owner(user)

    if not worker_res.get("ok"):
        err_code = str(worker_res.get("error_code") or "failed_verify")
        err_detail = str(worker_res.get("error_detail") or "verification failed")
        err_status = int(worker_res.get("error_status") or 422)
        actual_sha = str(worker_res.get("actual_sha256") or "")
        actual_bytes = int(worker_res.get("actual_bytes") or (staged_file.stat().st_size if staged_file.is_file() else 0))

        if err_code in ("failed_verify_worker", "failed_verify_timeout") and staged_file.is_file():
            q_path = _safe_under(config.QUARANTINE_DIR, config.QUARANTINE_DIR / f"{art_id}_{safe_fn}")
            shutil.move(str(staged_file), str(q_path))
            conn.execute(
                """UPDATE slots
                   SET state='open', claim_id=NULL, claimed_by=NULL, claimed_at=NULL, expires_at=NULL,
                       request_id=NULL, verify_status=?, verify_reason=?
                   WHERE job=? AND slot_id=?""",
                (err_code, err_detail, slot["job"], slot["slot_id"]),
            )
            _apply_strike_and_quarantine(conn, user["login"], is_owner=is_own)
        else:
            staged_file.unlink(missing_ok=True)
        db.log_audit(
            conn,
            user["login"],
            "upload_rejected",
            err_code,
            slot["job"],
            slot["slot_id"],
            {"sha256": actual_sha[:16], "bytes": actual_bytes},
        )
        return err_status, {
            "accepted": False,
            "state": "rejected",
            "verify_status": err_code,
            "detail": err_detail,
        }

    actual_bytes = int(worker_res["actual_bytes"])
    actual_sha = str(worker_res["actual_sha256"])
    vr = worker_res["verify_result"]
    accepted = bool(vr["accepted"])
    final_state = str(vr["final_state"])
    verify_status = str(vr["verify_status"])
    reason = str(vr["reason"])
    metrics = vr.get("metrics") or {}

    if not accepted:
        q_name = f"{art_id}_{safe_fn}"
        q_path = _safe_under(config.QUARANTINE_DIR, config.QUARANTINE_DIR / q_name)
        shutil.move(str(staged_file), str(q_path))
        (config.QUARANTINE_DIR / f"{art_id}.verdict.json").write_text(
            json.dumps(
                {
                    "job": slot["job"],
                    "slot_id": slot["slot_id"],
                    "actor": user["login"],
                    "sha256": actual_sha,
                    "bytes": actual_bytes,
                    "verify_status": verify_status,
                    "reason": reason,
                    "metrics": metrics,
                    "utc": db.utc_iso(now),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        conn.execute(
            """UPDATE slots
               SET state='open', claim_id=NULL, claimed_by=NULL, claimed_at=NULL, expires_at=NULL,
                   request_id=NULL, verify_status=?, verify_reason=?
               WHERE job=? AND slot_id=?""",
            (verify_status, reason, slot["job"], slot["slot_id"]),
        )
        quarantined_now = _apply_strike_and_quarantine(conn, user["login"], is_owner=is_own)
        db.log_audit(
            conn,
            user["login"],
            "upload_rejected",
            verify_status,
            slot["job"],
            slot["slot_id"],
            {"sha256": actual_sha[:16], "bytes": actual_bytes, "auto_quarantined": quarantined_now},
        )
        return 422, {
            "accepted": False,
            "state": "rejected_and_reopened",
            "verify_status": verify_status,
            "reason": f"rejected ({verify_status})",
            "sha256": actual_sha,
            "bytes": actual_bytes,
            "quarantined": quarantined_now,
        }

    v_name = f"{art_id}_{safe_fn}"
    v_path = _safe_under(config.VERIFIED_DIR, config.VERIFIED_DIR / v_name)
    shutil.move(str(staged_file), str(v_path))
    rel_path = str(v_path.relative_to(config.DATA_DIR))
    ctype = "application/json" if safe_fn.endswith(".json") else "application/octet-stream"
    conn.execute(
        """INSERT INTO artifacts (artifact_id, kind, filename, rel_path, sha256, bytes, content_type, job, slot_id, owner_login, created_at)
           VALUES (?, 'upload', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (art_id, safe_fn, rel_path, actual_sha, actual_bytes, ctype, slot["job"], slot["slot_id"], user["login"], now),
    )

    verified_at = now if final_state == "verified" else None
    conn.execute(
        """UPDATE slots
           SET state=?, upload_artifact_id=?, uploaded_by=?, uploaded_at=?, upload_bytes=?, upload_sha256=?,
               metrics_json=?, verify_status=?, verify_reason=?, verified_at=?,
               claim_id=NULL, expires_at=NULL
           WHERE job=? AND slot_id=?""",
        (
            final_state,
            art_id,
            user["login"],
            now,
            actual_bytes,
            actual_sha,
            json.dumps(metrics),
            verify_status,
            reason,
            verified_at,
            slot["job"],
            slot["slot_id"],
        ),
    )
    opened = []
    if final_state == "verified":
        opened = db.trigger_downstream_readiness(conn, slot["job"], slot["slot_id"], art_id)

    db.log_audit(
        conn,
        user["login"],
        "upload_accepted" if final_state == "verified" else "upload_queued_for_gate",
        verify_status,
        slot["job"],
        slot["slot_id"],
        {
            "sha256": actual_sha[:16],
            "bytes": actual_bytes,
            "opened_downstream": opened,
        },
    )
    return 200, {
        "accepted": True,
        "state": final_state,
        "sha256": actual_sha,
        "bytes": actual_bytes,
        "verify_status": verify_status,
        "reason": reason,
        "metrics": metrics,
        "opened_downstream": opened,
    }


@app.post("/api/uploads/direct/{claim_id}")
async def direct_upload(request: Request, claim_id: str) -> JSONResponse:
    """Single-request upload for artifacts <= 12 MiB; verification runs in isolated subprocess outside DB lock."""
    _require_x_quant(request)
    _validate_id(claim_id, "claim_id")

    expected_sha = (request.headers.get("x-content-sha256") or "").strip().lower()
    if not expected_sha or not HEX64_RE.match(expected_sha):
        raise HTTPException(status_code=400, detail="X-Content-SHA256 header (64-char hex) is required")

    with db.ro() as conn:
        user = _require_auth(conn, request)
        slot = conn.execute("SELECT * FROM slots WHERE claim_id=? AND state='claimed'", (claim_id,)).fetchone()
        if slot is None:
            raise HTTPException(status_code=404, detail="active claim not found or expired")
        if slot["claimed_by"] != user["login"]:
            raise HTTPException(status_code=403, detail="claim belongs to another contributor")
        job_row = conn.execute("SELECT * FROM jobs WHERE job=?", (slot["job"],)).fetchone()
        vctx = _gather_verify_context(conn, user, slot, job_row)

    raw = await _read_body_capped(
        request,
        min(config.MAX_DIRECT_UPLOAD_BYTES, int(vctx["max_bytes"])),
        "payload exceeds direct upload limit; use chunked upload",
    )
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="empty upload body")

    filename = (request.query_params.get("filename") or "upload.bin").strip()
    manifest_raw = request.headers.get("x-manifest-json", "{}")
    try:
        manifest = json.loads(manifest_raw) if manifest_raw else {}
        if not isinstance(manifest, dict):
            manifest = {}
    except Exception:
        manifest = {}

    staged = config.STAGING_DIR / f"direct_{secrets.token_hex(12)}.tmp"
    try:
        staged.write_bytes(raw)
        worker_res = await _run_verify_subprocess(
            {
                "mode": "direct",
                "kind": vctx["kind"],
                "staged_path": str(staged),
                "expected_sha256": expected_sha,
                "max_bytes": vctx["max_bytes"],
                "slot_plan": vctx["slot_plan"],
                "prompt_path": vctx["prompt_path"],
                "manifest": manifest,
                "peer_metrics": vctx["peer_metrics"],
                "require_owner_gate": vctx["require_owner_gate"],
            }
        )
        with db.tx(immediate=True) as conn:
            db.sweep_expired_claims(conn)
            user = _require_auth(conn, request, touch_last_seen=True)
            slot = conn.execute("SELECT * FROM slots WHERE claim_id=? AND state='claimed'", (claim_id,)).fetchone()
            if slot is None:
                raise HTTPException(status_code=404, detail="active claim not found or expired")
            if slot["claimed_by"] != user["login"]:
                raise HTTPException(status_code=403, detail="claim belongs to another contributor")
            status_code, content = _commit_upload_verdict(conn, user, slot, staged, filename, worker_res)
            return JSONResponse(status_code=status_code, content=content)
    finally:
        staged.unlink(missing_ok=True)


@app.post("/api/uploads/init", status_code=201)
async def init_chunked_upload(request: Request) -> JSONResponse:
    _require_x_quant(request)
    with db.ro() as conn:
        _require_auth(conn, request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="valid JSON body required")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body required")

    claim_id = _validate_id(str(body.get("claim_id") or ""), "claim_id")
    filename = str(body.get("filename") or "artifact.safetensors").strip()
    try:
        total_bytes = int(body.get("total_bytes") or 0)
        chunk_bytes = int(body.get("chunk_bytes") or config.MAX_CHUNK_BYTES)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="total_bytes and chunk_bytes must be integers")
    total_sha256 = str(body.get("total_sha256") or "").strip().lower()
    manifest = body.get("manifest") or {}
    if not isinstance(manifest, dict):
        raise HTTPException(status_code=400, detail="manifest must be a JSON object")

    if not HEX64_RE.match(total_sha256):
        raise HTTPException(status_code=400, detail="total_sha256 must be 64-char lowercase hex")
    if total_bytes <= 0:
        raise HTTPException(status_code=400, detail="total_bytes must be positive")

    min_allowed = config.MIN_CHUNK_BYTES if total_bytes > config.MIN_CHUNK_BYTES else total_bytes
    if chunk_bytes < min_allowed or chunk_bytes > config.MAX_CHUNK_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"chunk_bytes must be in {min_allowed}..{config.MAX_CHUNK_BYTES}",
        )

    total_chunks = (total_bytes + chunk_bytes - 1) // chunk_bytes
    if total_chunks > config.MAX_CHUNKS_PER_SESSION:
        raise HTTPException(
            status_code=400,
            detail=f"total_chunks {total_chunks} exceeds max {config.MAX_CHUNKS_PER_SESSION}",
        )

    with db.tx(immediate=True) as conn:
        db.sweep_expired_claims(conn)
        user = _require_auth(conn, request, touch_last_seen=True)
        slot = conn.execute("SELECT * FROM slots WHERE claim_id=? AND state='claimed'", (claim_id,)).fetchone()
        if slot is None:
            raise HTTPException(status_code=404, detail="active claim not found")
        if slot["claimed_by"] != user["login"]:
            raise HTTPException(status_code=403, detail="claim belongs to another contributor")
        job_row = conn.execute("SELECT * FROM jobs WHERE job=?", (slot["job"],)).fetchone()
        max_bytes = config.MAX_UPLOAD_BYTES_BY_KIND.get(job_row["job_kind"], config.MAX_DIRECT_UPLOAD_BYTES)
        if total_bytes > max_bytes:
            raise HTTPException(status_code=413, detail=f"total_bytes {total_bytes} exceeds cap {max_bytes}")

        open_sessions = int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM upload_sessions WHERE contributor_login=? AND state IN ('open', 'verifying')",
                (user["login"],),
            ).fetchone()["n"]
        )
        if open_sessions >= config.MAX_OPEN_UPLOAD_SESSIONS_PER_LOGIN:
            raise HTTPException(
                status_code=429,
                detail=f"open upload session cap reached ({open_sessions}/{config.MAX_OPEN_UPLOAD_SESSIONS_PER_LOGIN})",
            )

        upload_id = "upl_" + secrets.token_hex(12)
        sdir = _safe_under(config.STAGING_DIR, config.STAGING_DIR / upload_id)
        sdir.mkdir(parents=True, exist_ok=True)
        now = db.utc_ms()
        conn.execute(
            """INSERT INTO upload_sessions
               (upload_id, claim_id, job, slot_id, contributor_login, filename, total_bytes, total_sha256,
                chunk_bytes, total_chunks, received_chunks_json, manifest_json, state, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, 'open', ?, ?)""",
            (
                upload_id,
                claim_id,
                slot["job"],
                slot["slot_id"],
                user["login"],
                filename,
                total_bytes,
                total_sha256,
                chunk_bytes,
                total_chunks,
                json.dumps(manifest),
                now,
                slot["expires_at"] or (now + config.DEFAULT_CLAIM_TTL_SEC * 1000),
            ),
        )
        burl = _base_url_for_request(request)
        return JSONResponse(
            status_code=201,
            content={
                "upload_id": upload_id,
                "total_chunks": total_chunks,
                "chunk_bytes": chunk_bytes,
                "chunk_url_template": f"{burl}/api/uploads/{upload_id}/chunk/{{index}}",
                "complete_url": f"{burl}/api/uploads/{upload_id}/complete",
            },
        )


@app.put("/api/uploads/{upload_id}/chunk/{index}")
async def put_upload_chunk(request: Request, upload_id: str, index: int) -> dict[str, Any]:
    _require_x_quant(request)
    _validate_id(upload_id, "upload_id")

    hdr_sha = (request.headers.get("x-chunk-sha256") or "").strip().lower()
    if not hdr_sha or not HEX64_RE.match(hdr_sha):
        raise HTTPException(status_code=400, detail="X-Chunk-SHA256 header (64-char hex) is required")

    with db.ro() as conn:
        user = _require_auth(conn, request)
        sess = conn.execute("SELECT * FROM upload_sessions WHERE upload_id=? AND state='open'", (upload_id,)).fetchone()
        if sess is None:
            raise HTTPException(status_code=404, detail="open upload session not found")
        if sess["contributor_login"] != user["login"]:
            raise HTTPException(status_code=403, detail="upload session belongs to another contributor")
        total_chunks = int(sess["total_chunks"])
        if index < 0 or index >= total_chunks:
            raise HTTPException(status_code=400, detail=f"chunk index {index} out of range 0..{total_chunks - 1}")
        chunk_bytes = int(sess["chunk_bytes"])
        total_bytes = int(sess["total_bytes"])
        expected_len = chunk_bytes if index < total_chunks - 1 else (total_bytes - (total_chunks - 1) * chunk_bytes)

    raw = await _read_body_capped(
        request,
        config.MAX_CHUNK_BYTES,
        f"chunk size exceeds max {config.MAX_CHUNK_BYTES}",
    )
    if len(raw) != expected_len:
        raise HTTPException(
            status_code=422,
            detail=f"chunk {index} size {len(raw)} != expected {expected_len}",
        )

    chunk_sha = hashlib.sha256(raw).hexdigest()
    if hdr_sha != chunk_sha:
        raise HTTPException(status_code=422, detail="X-Chunk-SHA256 mismatch")

    part_path = _safe_under(config.STAGING_DIR, config.STAGING_DIR / upload_id / f"part_{index:06d}")
    part_path.parent.mkdir(parents=True, exist_ok=True)
    part_tmp = _safe_under(config.STAGING_DIR, config.STAGING_DIR / upload_id / f"part_{index:06d}.{secrets.token_hex(4)}.tmp")
    part_tmp.write_bytes(raw)
    os.replace(part_tmp, part_path)

    with db.tx(immediate=True) as conn:
        user = _require_auth(conn, request, touch_last_seen=True)
        sess = conn.execute("SELECT * FROM upload_sessions WHERE upload_id=? AND state='open'", (upload_id,)).fetchone()
        if sess is None:
            part_path.unlink(missing_ok=True)
            raise HTTPException(status_code=409, detail="upload session no longer open")
        if sess["contributor_login"] != user["login"]:
            raise HTTPException(status_code=403, detail="upload session belongs to another contributor")
        rec = json.loads(sess["received_chunks_json"] or "{}")
        rec[str(index)] = {"bytes": len(raw), "sha256": chunk_sha}
        conn.execute("UPDATE upload_sessions SET received_chunks_json=? WHERE upload_id=?", (json.dumps(rec), upload_id))
        return {
            "upload_id": upload_id,
            "index": index,
            "bytes": len(raw),
            "sha256": chunk_sha,
            "received": len(rec),
            "total_chunks": sess["total_chunks"],
        }


@app.post("/api/uploads/{upload_id}/complete")
async def complete_chunked_upload(request: Request, upload_id: str) -> JSONResponse:
    """Assemble and verify chunked upload in isolated worker subprocess outside SQLite write lock."""
    _require_x_quant(request)
    _validate_id(upload_id, "upload_id")

    with db.tx(immediate=True) as conn:
        db.sweep_expired_claims(conn)
        user = _require_auth(conn, request, touch_last_seen=True)
        sess = conn.execute("SELECT * FROM upload_sessions WHERE upload_id=? AND state='open'", (upload_id,)).fetchone()
        if sess is None:
            raise HTTPException(status_code=404, detail="open upload session not found")
        if sess["contributor_login"] != user["login"]:
            raise HTTPException(status_code=403, detail="upload session belongs to another contributor")
        slot = conn.execute("SELECT * FROM slots WHERE claim_id=? AND state='claimed'", (sess["claim_id"],)).fetchone()
        if slot is None:
            raise HTTPException(status_code=409, detail="associated claim expired or closed")
        job_row = conn.execute("SELECT * FROM jobs WHERE job=?", (slot["job"],)).fetchone()

        total_chunks = int(sess["total_chunks"])
        rec = json.loads(sess["received_chunks_json"] or "{}")
        missing = [i for i in range(total_chunks) if str(i) not in rec]
        if missing:
            raise HTTPException(status_code=400, detail=f"missing {len(missing)} chunks (first: {missing[:5]})")

        conn.execute("UPDATE upload_sessions SET state='verifying' WHERE upload_id=? AND state='open'", (upload_id,))
        vctx = _gather_verify_context(conn, user, slot, job_row)
        claim_id = str(sess["claim_id"])
        declared_bytes = int(sess["total_bytes"])
        total_sha256 = str(sess["total_sha256"])
        filename = str(sess["filename"])
        manifest = json.loads(sess["manifest_json"] or "{}")

    sdir = _safe_under(config.STAGING_DIR, config.STAGING_DIR / upload_id)
    assembled = _safe_under(config.STAGING_DIR, config.STAGING_DIR / f"{upload_id}_assembled.tmp")
    try:
        worker_res = await _run_verify_subprocess(
            {
                "mode": "chunked",
                "kind": vctx["kind"],
                "staging_dir": str(sdir),
                "assembled_path": str(assembled),
                "total_chunks": total_chunks,
                "declared_bytes": declared_bytes,
                "expected_sha256": total_sha256,
                "max_bytes": vctx["max_bytes"],
                "slot_plan": vctx["slot_plan"],
                "prompt_path": vctx["prompt_path"],
                "manifest": manifest,
                "peer_metrics": vctx["peer_metrics"],
                "require_owner_gate": vctx["require_owner_gate"],
            }
        )
        shutil.rmtree(sdir, ignore_errors=True)
        with db.tx(immediate=True) as conn:
            user = _require_auth(conn, request, touch_last_seen=True)
            slot = conn.execute("SELECT * FROM slots WHERE claim_id=? AND state='claimed'", (claim_id,)).fetchone()
            if slot is None:
                conn.execute("UPDATE upload_sessions SET state='failed' WHERE upload_id=?", (upload_id,))
                raise HTTPException(status_code=409, detail="associated claim expired or closed during verification")
            status_code, content = _commit_upload_verdict(conn, user, slot, assembled, filename, worker_res)
            sess_state = "done" if content.get("accepted") else "rejected"
            conn.execute("UPDATE upload_sessions SET state=? WHERE upload_id=?", (sess_state, upload_id))
            return JSONResponse(status_code=status_code, content=content)
    finally:
        shutil.rmtree(sdir, ignore_errors=True)
        assembled.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Admin API (X-Quant: 1 + separate Admin Bearer credential; access-requests also supports owner GitHub session + CSRF)
# ---------------------------------------------------------------------------

@app.get("/api/admin/access-requests")
def admin_list_access_requests(request: Request, status: str | None = None) -> list[dict[str, Any]]:
    """List pending, approved, denied, and quarantined GitHub accounts for owner review."""
    _require_x_quant(request)
    with db.ro() as conn:
        _require_admin_or_owner_session(conn, request, require_csrf_for_session=False)
        rows = conn.execute(
            """SELECT c.*,
                      (SELECT COUNT(*) FROM api_keys k WHERE k.contributor_login = c.login AND k.revoked_at IS NULL) AS live_keys
               FROM contributors c
               WHERE c.is_admin = 0 AND (c.github_id IS NOT NULL OR c.approved = 0 OR c.denied = 1)
               ORDER BY c.approved ASC, c.denied ASC, c.created_at DESC"""
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            st = _user_status(r)
            if status and st != status:
                continue
            out.append(
                {
                    "login": r["login"],
                    "display_login": _display_login(r),
                    "github_id": r["github_id"],
                    "github_login": r["github_login"],
                    "github_name": r["github_name"],
                    "status": st,
                    "is_owner": bool(r["is_owner"]),
                    "approved": bool(r["approved"]),
                    "denied": bool(r["denied"]),
                    "quarantined": bool(r["quarantined"]),
                    "tiers": json.loads(r["tiers_json"] or "[]"),
                    "strikes": int(r["strikes"]),
                    "live_keys": int(r["live_keys"] or 0),
                    "note": r["note"],
                    "created_at": r["created_at"],
                    "created_utc": db.utc_iso(r["created_at"]),
                    "last_seen_at": r["last_seen_at"],
                }
            )
        return out


@app.post("/api/admin/access-requests/{login}/approve")
async def admin_approve_access_request(request: Request, login: str) -> dict[str, Any]:
    _require_x_quant(request)
    _validate_id(login, "login")
    body: dict[str, Any] = {}
    raw = await _read_body_capped(request, 8192, "request body too large")
    if raw.strip():
        try:
            parsed = json.loads(raw.decode("utf-8"))
            if isinstance(parsed, dict):
                body = parsed
        except Exception:
            raise HTTPException(status_code=400, detail="valid JSON body required")

    with db.tx(immediate=True) as conn:
        actor = _require_admin_or_owner_session(conn, request, require_csrf_for_session=True, touch_last_seen=True)
        existing = conn.execute("SELECT * FROM contributors WHERE login=? AND is_admin=0", (login,)).fetchone()
        if existing is None:
            raise HTTPException(status_code=404, detail="contributor not found")
        tiers = json.dumps(body["tiers"]) if "tiers" in body else (str(existing["tiers_json"] or "[]") if json.loads(existing["tiers_json"] or "[]") else json.dumps(["tier-label"]))
        is_owner = (1 if body["is_owner"] else 0) if "is_owner" in body else int(existing["is_owner"])
        quarantined = (1 if body["quarantined"] else 0) if "quarantined" in body else 0
        note = body["note"] if "note" in body else (existing["note"] if existing["note"] != "Pending GitHub access request" else f"Approved by {actor['login']}")
        conn.execute(
            """UPDATE contributors
               SET approved=1, denied=0, quarantined=?, is_owner=?, tiers_json=?, note=?
               WHERE login=?""",
            (quarantined, is_owner, tiers, note, login),
        )
        db.log_audit(conn, str(actor["login"]), "access_request_approved", "ok", detail={"login": login})
        updated = conn.execute("SELECT * FROM contributors WHERE login=?", (login,)).fetchone()
        return {
            "ok": True,
            "login": updated["login"],
            "display_login": _display_login(updated),
            "status": _user_status(updated),
            "approved": bool(updated["approved"]),
            "denied": bool(updated["denied"]),
            "quarantined": bool(updated["quarantined"]),
            "is_owner": bool(updated["is_owner"]),
            "tiers": json.loads(updated["tiers_json"] or "[]"),
        }


@app.post("/api/admin/access-requests/{login}/deny")
async def admin_deny_access_request(request: Request, login: str) -> dict[str, Any]:
    _require_x_quant(request)
    _validate_id(login, "login")
    body: dict[str, Any] = {}
    raw = await _read_body_capped(request, 8192, "request body too large")
    if raw.strip():
        try:
            parsed = json.loads(raw.decode("utf-8"))
            if isinstance(parsed, dict):
                body = parsed
        except Exception:
            raise HTTPException(status_code=400, detail="valid JSON body required")

    with db.tx(immediate=True) as conn:
        actor = _require_admin_or_owner_session(conn, request, require_csrf_for_session=True, touch_last_seen=True)
        existing = conn.execute("SELECT * FROM contributors WHERE login=? AND is_admin=0", (login,)).fetchone()
        if existing is None:
            raise HTTPException(status_code=404, detail="contributor not found")
        if str(existing["login"]) == str(actor["login"]) or bool(existing["is_owner"]):
            raise HTTPException(status_code=400, detail="cannot deny an owner account")
        note = body["note"] if "note" in body else f"Denied by {actor['login']}"
        conn.execute(
            "UPDATE contributors SET approved=0, denied=1, note=? WHERE login=?",
            (note, login),
        )
        revoked_keys = db.revoke_all_api_keys(conn, login)
        revoked_sessions = db.delete_user_sessions(conn, login)
        db.log_audit(conn, str(actor["login"]), "access_request_denied", "ok", detail={"login": login})
        updated = conn.execute("SELECT * FROM contributors WHERE login=?", (login,)).fetchone()
        return {
            "ok": True,
            "login": updated["login"],
            "display_login": _display_login(updated),
            "status": _user_status(updated),
            "approved": False,
            "denied": True,
            "revoked_keys": revoked_keys,
            "revoked_sessions": revoked_sessions,
        }


@app.post("/api/admin/runners")
async def admin_add_runner(request: Request) -> dict[str, Any]:
    _require_x_quant(request)
    with db.ro() as conn:
        _require_auth(conn, request, admin=True)
    body = await request.json()
    commit_id = _validate_id(str(body.get("commit_id") or ""), "commit_id")
    kind = str(body.get("kind") or "all")
    if kind != "all" and kind not in config.KIND_TO_TIER:
        raise HTTPException(status_code=400, detail="invalid runner kind")
    image_digest = body.get("image_digest")
    note = body.get("note")
    with db.tx(immediate=True) as conn:
        admin = _require_auth(conn, request, admin=True, touch_last_seen=True)
        conn.execute(
            """INSERT INTO runners (commit_id, kind, image_digest, note, active, added_at)
               VALUES (?, ?, ?, ?, 1, ?)
               ON CONFLICT(commit_id) DO UPDATE SET kind=excluded.kind, image_digest=excluded.image_digest, note=excluded.note, active=1""",
            (commit_id, kind, image_digest, note, db.utc_ms()),
        )
        db.log_audit(conn, admin["login"], "runner_registered", "ok", detail={"commit_id": commit_id, "kind": kind})
        return {"ok": True, "commit_id": commit_id}


@app.post("/api/admin/contributors")
async def admin_upsert_contributor(request: Request) -> dict[str, Any]:
    """Create or update contributor (S10: omitting fields on update preserves existing values)."""
    _require_x_quant(request)
    with db.ro() as conn:
        _require_auth(conn, request, admin=True)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body required")
    login = _validate_id(str(body.get("login") or ""), "login")
    raw_key = None
    with db.tx(immediate=True) as conn:
        admin = _require_auth(conn, request, admin=True, touch_last_seen=True)
        existing = conn.execute("SELECT * FROM contributors WHERE login=?", (login,)).fetchone()
        now = db.utc_ms()
        if existing is None:
            approved = 1 if body.get("approved", True) else 0
            denied = 1 if body.get("denied", False) else 0
            quarantined = 1 if body.get("quarantined", False) else 0
            is_owner = 1 if body.get("is_owner", False) else 0
            tiers = json.dumps(body["tiers"] if "tiers" in body else ["tier-label"])
            note = body.get("note")
            strikes = int(body.get("strikes", 0))
            github_id = int(body["github_id"]) if body.get("github_id") is not None else None
            github_login = str(body["github_login"]).strip()[:100] if body.get("github_login") else None
            raw_key = "drs_wrk_" + secrets.token_urlsafe(32)
            kh = db.hash_api_key(raw_key)
            conn.execute(
                """INSERT INTO contributors
                   (login, key_hash, is_admin, is_owner, approved, denied, tiers_json, strikes, quarantined, note, github_id, github_login, created_at)
                   VALUES (?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (login, kh, is_owner, approved, denied, tiers, strikes, quarantined, note, github_id, github_login, now),
            )
            kid = "key_" + secrets.token_hex(8)
            conn.execute(
                """INSERT INTO api_keys (key_id, contributor_login, key_hash, key_prefix, label, created_at, last_used_at, revoked_at)
                   VALUES (?, ?, ?, ?, 'Admin issued', ?, NULL, NULL)""",
                (kid, login, kh, db.key_prefix(raw_key), now),
            )
        else:
            # S10: Preserve every field omitted from the update payload!
            approved = (1 if body["approved"] else 0) if "approved" in body else int(existing["approved"])
            denied = (1 if body["denied"] else 0) if "denied" in body else int(existing["denied"])
            quarantined = (1 if body["quarantined"] else 0) if "quarantined" in body else int(existing["quarantined"])
            is_owner = (1 if body["is_owner"] else 0) if "is_owner" in body else int(existing["is_owner"])
            tiers = json.dumps(body["tiers"]) if "tiers" in body else str(existing["tiers_json"])
            note = body["note"] if "note" in body else existing["note"]
            strikes = int(body["strikes"]) if "strikes" in body else int(existing["strikes"])
            github_id = (int(body["github_id"]) if body["github_id"] is not None else None) if "github_id" in body else existing["github_id"]
            github_login = (str(body["github_login"]).strip()[:100] if body["github_login"] else None) if "github_login" in body else existing["github_login"]
            if body.get("rotate_key"):
                raw_key = "drs_wrk_" + secrets.token_urlsafe(32)
                kh = db.hash_api_key(raw_key)
                conn.execute(
                    "UPDATE api_keys SET revoked_at=? WHERE contributor_login=? AND revoked_at IS NULL",
                    (now, login),
                )
                kid = "key_" + secrets.token_hex(8)
                conn.execute(
                    """INSERT INTO api_keys (key_id, contributor_login, key_hash, key_prefix, label, created_at, last_used_at, revoked_at)
                       VALUES (?, ?, ?, ?, 'Admin rotated', ?, NULL, NULL)""",
                    (kid, login, kh, db.key_prefix(raw_key), now),
                )
                conn.execute(
                    """UPDATE contributors
                       SET key_hash=?, is_owner=?, approved=?, denied=?, tiers_json=?, strikes=?, quarantined=?, note=?, github_id=?, github_login=?
                       WHERE login=?""",
                    (kh, is_owner, approved, denied, tiers, strikes, quarantined, note, github_id, github_login, login),
                )
            else:
                conn.execute(
                    """UPDATE contributors
                       SET is_owner=?, approved=?, denied=?, tiers_json=?, strikes=?, quarantined=?, note=?, github_id=?, github_login=?
                       WHERE login=?""",
                    (is_owner, approved, denied, tiers, strikes, quarantined, note, github_id, github_login, login),
                )
            if "denied" in body and denied == 1:
                db.revoke_all_api_keys(conn, login)
                db.delete_user_sessions(conn, login)

        db.log_audit(
            conn,
            admin["login"],
            "contributor_updated",
            "ok",
            detail={"login": login, "approved": bool(approved), "denied": bool(denied), "quarantined": bool(quarantined), "is_owner": bool(is_owner)},
        )
        resp: dict[str, Any] = {
            "ok": True,
            "login": login,
            "approved": bool(approved),
            "denied": bool(denied),
            "quarantined": bool(quarantined),
            "is_owner": bool(is_owner),
            "tiers": json.loads(tiers),
            "strikes": strikes,
        }
        if raw_key:
            resp["api_key_shown_once"] = raw_key
        return resp


@app.post("/api/admin/artifacts/register")
async def admin_register_local_artifact(request: Request) -> dict[str, Any]:
    """Register an immutable input file already staged under /data/inputs/."""
    _require_x_quant(request)
    with db.ro() as conn:
        _require_auth(conn, request, admin=True)
    body = await request.json()
    rel_path = str(body.get("rel_path") or "").lstrip("/")
    artifact_id = _validate_id(str(body.get("artifact_id") or ("inp_" + secrets.token_hex(8))), "artifact_id")
    kind = str(body.get("kind") or "input")
    content_type = str(body.get("content_type") or "application/octet-stream")
    fpath = _safe_under(config.INPUTS_DIR, config.DATA_DIR / rel_path)
    if not fpath.is_file():
        raise HTTPException(status_code=404, detail=f"file not found under inputs: {rel_path}")
    from server import verify

    sha = verify.sha256_file(fpath)
    nbytes = fpath.stat().st_size
    with db.tx(immediate=True) as conn:
        admin = _require_auth(conn, request, admin=True, touch_last_seen=True)
        conn.execute(
            """INSERT INTO artifacts (artifact_id, kind, filename, rel_path, sha256, bytes, content_type, job, slot_id, owner_login, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(artifact_id) DO UPDATE SET rel_path=excluded.rel_path, sha256=excluded.sha256, bytes=excluded.bytes""",
            (artifact_id, kind, fpath.name, rel_path, sha, nbytes, content_type, body.get("job"), body.get("slot_id"), admin["login"], db.utc_ms()),
        )
        return {"ok": True, "artifact_id": artifact_id, "sha256": sha, "bytes": nbytes, "url": f"{config.PUBLIC_URL}/api/artifacts/{artifact_id}"}


@app.post("/api/admin/jobs")
async def admin_upsert_job(request: Request) -> dict[str, Any]:
    _require_x_quant(request)
    with db.ro() as conn:
        _require_auth(conn, request, admin=True)
    body = await request.json()
    job = _validate_id(str(body.get("job") or ""), "job")
    title = str(body.get("title") or job)
    job_kind = str(body.get("job_kind") or "")
    if job_kind not in config.KIND_TO_TIER:
        raise HTTPException(status_code=400, detail="invalid job_kind")
    model = str(body.get("model") or "GLM-5.3")
    release_repo = str(body.get("release_repo") or "kindlingai/glm-5.3-exlr8-k2-k3-k4")
    release_revision = str(body.get("release_revision") or "9e25c28a2e8808d8b2599d780521fd6af24e7558")
    active_round = int(body.get("active_round") or 1)
    cfg = json.dumps(body.get("config") or {})
    slots = body.get("slots") or []
    rounds = body.get("rounds") or []

    with db.tx(immediate=True) as conn:
        admin = _require_auth(conn, request, admin=True, touch_last_seen=True)
        conn.execute(
            """INSERT INTO jobs (job, title, job_kind, model, release_repo, release_revision, version, active_round, config_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
               ON CONFLICT(job) DO UPDATE SET title=excluded.title, active_round=excluded.active_round, config_json=excluded.config_json""",
            (job, title, job_kind, model, release_repo, release_revision, active_round, cfg, db.utc_ms()),
        )
        for rd in rounds:
            conn.execute(
                """INSERT INTO rounds (job, round_num, base_ckpt_name, base_ckpt_sha256, quorum, total_slots, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(job, round_num) DO UPDATE SET quorum=excluded.quorum, total_slots=excluded.total_slots""",
                (
                    job,
                    int(rd["round_num"]),
                    str(rd["base_ckpt_name"]),
                    rd.get("base_ckpt_sha256"),
                    int(rd.get("quorum", 4)),
                    int(rd.get("total_slots", 4)),
                    str(rd.get("status", "open")),
                ),
            )
        inserted = 0
        for s in slots:
            sid = _validate_id(str(s["slot_id"]), "slot_id")
            cur = conn.execute(
                """INSERT INTO slots (job, slot_id, row_key, col_key, round_num, profile_id, state,
                                      depends_on_job, depends_on_slot, inputs_json, plan_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(job, slot_id) DO NOTHING""",
                (
                    job,
                    sid,
                    str(s["row_key"]),
                    str(s["col_key"]),
                    int(s.get("round_num", 1)),
                    str(s.get("profile_id", "")),
                    str(s.get("state", "not_ready")),
                    s.get("depends_on_job"),
                    s.get("depends_on_slot"),
                    json.dumps(s.get("inputs") or []),
                    json.dumps(s.get("plan") or {}),
                ),
            )
            inserted += int(cur.rowcount or 0)
        db.log_audit(conn, admin["login"], "job_upserted", "ok", job, detail={"job_kind": job_kind, "slots_inserted": inserted})
        return {"ok": True, "job": job, "slots_processed": inserted}


@app.post("/api/admin/slots/{job}/{slot_id}/ready")
async def admin_mark_slot_ready(request: Request, job: str, slot_id: str) -> dict[str, Any]:
    _require_x_quant(request)
    _validate_id(job, "job")
    _validate_id(slot_id, "slot_id")
    body = await request.json()
    state = str(body.get("state") or "open")
    if state not in ALLOWED_READY_STATES:
        raise HTTPException(status_code=400, detail=f"invalid slot state '{state}'")
    with db.tx(immediate=True) as conn:
        admin = _require_auth(conn, request, admin=True, touch_last_seen=True)
        slot = conn.execute("SELECT * FROM slots WHERE job=? AND slot_id=?", (job, slot_id)).fetchone()
        if slot is None:
            raise HTTPException(status_code=404, detail="slot not found")
        inputs = body.get("inputs")
        plan = body.get("plan")
        conn.execute(
            """UPDATE slots SET state=?,
               inputs_json=COALESCE(?, inputs_json),
               plan_json=COALESCE(?, plan_json)
               WHERE job=? AND slot_id=?""",
            (state, json.dumps(inputs) if inputs is not None else None, json.dumps(plan) if plan is not None else None, job, slot_id),
        )
        db.log_audit(conn, admin["login"], "slot_readiness_set", state, job, slot_id)
        return {"ok": True, "job": job, "slot_id": slot_id, "state": state}


@app.post("/api/admin/verifications/{job}/{slot_id}")
async def admin_record_verification(request: Request, job: str, slot_id: str) -> dict[str, Any]:
    """Record owner-side verification verdict (label spot-rescoring gate or train replay gate)."""
    _require_x_quant(request)
    _validate_id(job, "job")
    _validate_id(slot_id, "slot_id")
    body = await request.json()
    passed = bool(body.get("passed"))
    default_vstatus = "verified_replay_fltrust_gate" if passed else "rejected_replay_fltrust_gate"
    verify_status = _validate_id(str(body.get("verify_status") or default_vstatus), "verify_status")
    reason = str(body.get("reason") or ("passed owner verification gate" if passed else "failed owner verification gate"))
    gate_metrics = body.get("gate_metrics") or {}
    reopen_on_reject = bool(body.get("reopen_on_reject", True))

    with db.tx(immediate=True) as conn:
        _require_auth(conn, request, admin=True, touch_last_seen=True)
        slot = conn.execute("SELECT * FROM slots WHERE job=? AND slot_id=?", (job, slot_id)).fetchone()
        if slot is None:
            raise HTTPException(status_code=404, detail="slot not found")
        if slot["state"] not in ("uploaded", "verified"):
            raise HTTPException(status_code=409, detail=f"slot is '{slot['state']}', expected 'uploaded'")

        now = db.utc_ms()
        merged_metrics = json.loads(slot["metrics_json"] or "{}")
        merged_metrics["owner_gate"] = gate_metrics
        actor = slot["uploaded_by"] or "unknown"

        if passed:
            conn.execute(
                """UPDATE slots
                   SET state='verified', verify_status=?, verify_reason=?,
                       verified_at=?, metrics_json=?
                   WHERE job=? AND slot_id=?""",
                (verify_status, reason, now, json.dumps(merged_metrics), job, slot_id),
            )
            opened = []
            if slot["upload_artifact_id"]:
                opened = db.trigger_downstream_readiness(conn, job, slot_id, slot["upload_artifact_id"])
            db.log_audit(
                conn,
                actor,
                "verification_passed",
                verify_status,
                job,
                slot_id,
                {"opened_downstream": opened},
            )
            return {"ok": True, "state": "verified", "verify_status": verify_status, "opened_downstream": opened}

        art_id = slot["upload_artifact_id"]
        if art_id:
            art = conn.execute("SELECT * FROM artifacts WHERE artifact_id=?", (art_id,)).fetchone()
            if art:
                src = _safe_under(config.DATA_DIR, config.DATA_DIR / art["rel_path"])
                if src.is_file():
                    dst = _safe_under(config.QUARANTINE_DIR, config.QUARANTINE_DIR / src.name)
                    shutil.move(str(src), str(dst))
                    conn.execute(
                        "UPDATE artifacts SET kind='quarantine', rel_path=? WHERE artifact_id=?",
                        (str(dst.relative_to(config.DATA_DIR)), art_id),
                    )
                (config.QUARANTINE_DIR / f"{art_id}.gate_verdict.json").write_text(
                    json.dumps(
                        {
                            "job": job,
                            "slot_id": slot_id,
                            "actor": actor,
                            "passed": False,
                            "verify_status": verify_status,
                            "reason": reason,
                            "gate_metrics": gate_metrics,
                            "utc": db.utc_iso(now),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

        next_state = "open" if reopen_on_reject else "rejected"
        if reopen_on_reject:
            conn.execute(
                """UPDATE slots
                   SET state='open', upload_artifact_id=NULL, uploaded_by=NULL, uploaded_at=NULL,
                       upload_bytes=0, upload_sha256=NULL, verify_status=?,
                       verify_reason=?, metrics_json=?, claim_id=NULL, claimed_by=NULL, claimed_at=NULL, expires_at=NULL, host_tag=NULL, request_id=NULL
                   WHERE job=? AND slot_id=?""",
                (verify_status, reason, json.dumps(merged_metrics), job, slot_id),
            )
        else:
            conn.execute(
                """UPDATE slots
                   SET state='rejected', verify_status=?, verify_reason=?, metrics_json=?
                   WHERE job=? AND slot_id=?""",
                (verify_status, reason, json.dumps(merged_metrics), job, slot_id),
            )
        quarantined_now = False
        if actor != "unknown":
            actor_row = conn.execute("SELECT * FROM contributors WHERE login=?", (actor,)).fetchone()
            if actor_row is not None:
                quarantined_now = _apply_strike_and_quarantine(conn, actor, is_owner=_is_owner(actor_row))
        db.log_audit(
            conn,
            actor,
            "verification_rejected",
            verify_status,
            job,
            slot_id,
            {"next_state": next_state, "auto_quarantined": quarantined_now},
        )
        return {
            "ok": True,
            "state": next_state,
            "verify_status": verify_status,
            "quarantined_actor": quarantined_now,
        }


@app.post("/api/admin/rounds/{job}/{round_num}/aggregate")
async def admin_record_round_aggregate(request: Request, job: str, round_num: int) -> dict[str, Any]:
    _require_x_quant(request)
    _validate_id(job, "job")
    body = await request.json()
    with db.tx(immediate=True) as conn:
        admin = _require_auth(conn, request, admin=True, touch_last_seen=True)
        now = db.utc_ms()
        conn.execute(
            """UPDATE rounds
               SET status='completed', agg_method=?, agg_ckpt_name=?, agg_ckpt_sha256=?, agg_metrics_json=?, completed_at=?
               WHERE job=? AND round_num=?""",
            (
                str(body.get("agg_method") or "trimmed_mean_soup"),
                str(body.get("agg_ckpt_name") or ""),
                str(body.get("agg_ckpt_sha256") or ""),
                json.dumps(body.get("agg_metrics") or {}),
                now,
                job,
                round_num,
            ),
        )
        conn.execute(
            "UPDATE slots SET harvested=1 WHERE job=? AND round_num=? AND state='verified'",
            (job, round_num),
        )
        db.log_audit(
            conn,
            admin["login"],
            "round_aggregated",
            "ok",
            job,
            f"round_{round_num}",
            {"agg_method": body.get("agg_method"), "agg_ckpt_name": body.get("agg_ckpt_name")},
        )
        return {"ok": True, "job": job, "round_num": round_num}


@app.get("/api/admin/host-map")
def admin_host_map(request: Request) -> list[dict[str, Any]]:
    """Private admin-only mapping of neutral contributor logins to internal host tags, tiers, strikes, and notes."""
    _require_x_quant(request)
    with db.ro() as conn:
        _require_auth(conn, request, admin=True)
        contribs = conn.execute(
            "SELECT login, github_id, github_login, is_owner, approved, denied, quarantined, tiers_json, strikes, note, last_seen_at FROM contributors WHERE is_admin=0 ORDER BY login ASC"
        ).fetchall()
        out = []
        for c in contribs:
            host_rows = conn.execute(
                "SELECT DISTINCT host_tag FROM slots WHERE (uploaded_by=? OR claimed_by=?) AND host_tag IS NOT NULL",
                (c["login"], c["login"]),
            ).fetchall()
            audit_rows = conn.execute(
                "SELECT detail_json FROM audit_events WHERE actor=? AND action='claim_created'",
                (c["login"],),
            ).fetchall()
            hosts = {r["host_tag"] for r in host_rows if r["host_tag"]}
            for ar in audit_rows:
                ht = json.loads(ar["detail_json"] or "{}").get("host_tag")
                if ht:
                    hosts.add(ht)
            out.append(
                {
                    "login": c["login"],
                    "github_id": c["github_id"],
                    "github_login": c["github_login"],
                    "is_owner": bool(c["is_owner"]),
                    "approved": bool(c["approved"]),
                    "denied": bool(c["denied"]),
                    "quarantined": bool(c["quarantined"]),
                    "note": c["note"],
                    "tiers": json.loads(c["tiers_json"] or "[]"),
                    "strikes": c["strikes"],
                    "observed_host_tags": sorted(hosts),
                    "last_seen_at": c["last_seen_at"],
                }
            )
        return out


@app.get("/api/admin/pending-verifications")
def admin_list_pending(request: Request) -> list[dict[str, Any]]:
    """List uploaded slots awaiting owner-side verification (label spot gate or train replay gate)."""
    _require_x_quant(request)
    with db.ro() as conn:
        _require_auth(conn, request, admin=True)
        rows = conn.execute(
            """SELECT s.job, j.job_kind, s.slot_id, s.round_num, s.profile_id, s.uploaded_by, s.uploaded_at,
                      s.upload_artifact_id, s.upload_bytes, s.upload_sha256, s.verify_status,
                      s.metrics_json, s.plan_json, s.inputs_json, a.rel_path
               FROM slots s
               JOIN jobs j ON s.job = j.job
               LEFT JOIN artifacts a ON s.upload_artifact_id = a.artifact_id
               WHERE s.state='uploaded'
               ORDER BY s.uploaded_at ASC"""
        ).fetchall()
        return [
            {
                "job": r["job"],
                "job_kind": r["job_kind"],
                "slot_id": r["slot_id"],
                "round_num": r["round_num"],
                "profile_id": r["profile_id"],
                "uploaded_by": r["uploaded_by"],
                "uploaded_at": r["uploaded_at"],
                "artifact_id": r["upload_artifact_id"],
                "rel_path": r["rel_path"],
                "bytes": r["upload_bytes"],
                "sha256": r["upload_sha256"],
                "verify_status": r["verify_status"],
                "metrics": json.loads(r["metrics_json"] or "{}"),
                "plan": json.loads(r["plan_json"] or "{}"),
                "inputs": json.loads(r["inputs_json"] or "[]"),
            }
            for r in rows
        ]
