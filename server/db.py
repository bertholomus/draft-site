"""SQLite state store for jobs, slots, claims, uploads, verifications, rounds, sessions, API keys, and audit logs.

Uses journal_mode=DELETE (low-QPS coordinator; avoids -wal/-shm sidecar UID issues) and
separates read-only queries (db.ro()) from immediate write transactions (db.tx(immediate=True)).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import shutil
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from server import config

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_LOGIN_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
RESERVED_LOGINS = frozenset(("admin", "system", "owner", "root", "anonymous", "unknown"))


def utc_ms() -> int:
    return int(time.time() * 1000)


def utc_iso(ts_ms: int | None = None) -> str:
    t = (ts_ms / 1000.0) if ts_ms is not None else time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def hash_api_key(raw_key: str) -> str:
    """Deterministic SHA-256 with domain separation (keys are 256-bit random tokens)."""
    return hashlib.sha256(("drs-v1:" + raw_key).encode("utf-8")).hexdigest()


def hmac_token(purpose: str, value: str) -> str:
    """Keyed HMAC-SHA256 for storing session IDs and OAuth state/nonces at rest."""
    key = config.load_session_secret()
    return hmac.new(key, f"drs-v1:{purpose}:{value}".encode("utf-8"), hashlib.sha256).hexdigest()


def key_prefix(raw_key: str) -> str:
    """Return safe display prefix (e.g. drs_wrk_Ab12Cd) exposing at most 6 random chars."""
    return raw_key[:14]


def no_key_sentinel() -> str:
    """Non-matching unique placeholder for contributors.key_hash when account has no legacy key."""
    return "none:" + secrets.token_hex(16)


def is_real_key_hash(val: str | None) -> bool:
    return bool(val and HEX64_RE.match(val))


def ensure_dirs() -> None:
    for d in (
        config.DATA_DIR,
        config.DB_PATH.parent,
        config.INPUTS_DIR,
        config.STAGING_DIR,
        config.VERIFIED_DIR,
        config.QUARANTINE_DIR,
        config.EVALS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def connect(*, set_journal: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.DB_PATH), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    if set_journal:
        conn.execute("PRAGMA journal_mode=DELETE;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=15000;")
    return conn


@contextmanager
def ro() -> Iterator[sqlite3.Connection]:
    """Read-only SQLite connection (never takes the write lock)."""
    conn = connect()
    try:
        conn.execute("BEGIN;")
        yield conn
    finally:
        try:
            conn.execute("ROLLBACK;")
        except Exception:
            pass
        conn.close()


@contextmanager
def tx(immediate: bool = False) -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE;" if immediate else "BEGIN;")
        yield conn
        conn.execute("COMMIT;")
    except Exception:
        try:
            conn.execute("ROLLBACK;")
        except Exception:
            pass
        raise
    finally:
        conn.close()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS contributors (
    login TEXT PRIMARY KEY,
    key_hash TEXT NOT NULL UNIQUE,
    is_admin INTEGER NOT NULL DEFAULT 0,
    is_owner INTEGER NOT NULL DEFAULT 0,
    approved INTEGER NOT NULL DEFAULT 1,
    denied INTEGER NOT NULL DEFAULT 0,
    tiers_json TEXT NOT NULL DEFAULT '[]',
    strikes INTEGER NOT NULL DEFAULT 0,
    quarantined INTEGER NOT NULL DEFAULT 0,
    note TEXT,
    github_id INTEGER,
    github_login TEXT,
    github_name TEXT,
    created_at INTEGER NOT NULL,
    last_seen_at INTEGER
);

CREATE TABLE IF NOT EXISTS api_keys (
    key_id TEXT PRIMARY KEY,
    contributor_login TEXT NOT NULL REFERENCES contributors(login) ON DELETE CASCADE,
    key_hash TEXT NOT NULL UNIQUE,
    key_prefix TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    last_used_at INTEGER,
    revoked_at INTEGER
);

CREATE INDEX IF NOT EXISTS idx_api_keys_login ON api_keys(contributor_login, revoked_at);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);

CREATE TABLE IF NOT EXISTS sessions (
    session_hash TEXT PRIMARY KEY,
    contributor_login TEXT NOT NULL REFERENCES contributors(login) ON DELETE CASCADE,
    csrf_token TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    last_seen_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_login ON sessions(contributor_login);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);

CREATE TABLE IF NOT EXISTS oauth_states (
    state_hash TEXT PRIMARY KEY,
    nonce_hash TEXT NOT NULL,
    return_to TEXT NOT NULL DEFAULT '/',
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_oauth_states_expires ON oauth_states(expires_at);

CREATE TABLE IF NOT EXISTS runners (
    commit_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'all',
    image_digest TEXT,
    note TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    added_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    job_kind TEXT NOT NULL,
    model TEXT NOT NULL,
    release_repo TEXT NOT NULL,
    release_revision TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    active_round INTEGER NOT NULL DEFAULT 1,
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS slots (
    job TEXT NOT NULL REFERENCES jobs(job) ON DELETE CASCADE,
    slot_id TEXT NOT NULL,
    row_key TEXT NOT NULL,
    col_key TEXT NOT NULL,
    round_num INTEGER NOT NULL DEFAULT 1,
    profile_id TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT 'not_ready',
    depends_on_job TEXT,
    depends_on_slot TEXT,
    inputs_json TEXT NOT NULL DEFAULT '[]',
    plan_json TEXT NOT NULL DEFAULT '{}',
    claim_id TEXT,
    claimed_by TEXT REFERENCES contributors(login),
    claimed_at INTEGER,
    expires_at INTEGER,
    host_tag TEXT,
    request_id TEXT,
    upload_artifact_id TEXT,
    uploaded_by TEXT REFERENCES contributors(login),
    uploaded_at INTEGER,
    upload_bytes INTEGER DEFAULT 0,
    upload_sha256 TEXT,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    verify_status TEXT,
    verify_reason TEXT,
    verified_at INTEGER,
    harvested INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (job, slot_id)
);

CREATE INDEX IF NOT EXISTS idx_slots_state ON slots(job, state);
CREATE INDEX IF NOT EXISTS idx_slots_claim ON slots(claim_id);
CREATE INDEX IF NOT EXISTS idx_slots_deps ON slots(depends_on_job, depends_on_slot);
CREATE INDEX IF NOT EXISTS idx_slots_claimed_by ON slots(claimed_by, state);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    filename TEXT NOT NULL,
    rel_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'application/octet-stream',
    job TEXT,
    slot_id TEXT,
    owner_login TEXT,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS upload_sessions (
    upload_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    job TEXT NOT NULL,
    slot_id TEXT NOT NULL,
    contributor_login TEXT NOT NULL,
    filename TEXT NOT NULL,
    total_bytes INTEGER NOT NULL,
    total_sha256 TEXT NOT NULL,
    chunk_bytes INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    received_chunks_json TEXT NOT NULL DEFAULT '{}',
    manifest_json TEXT NOT NULL DEFAULT '{}',
    state TEXT NOT NULL DEFAULT 'open',
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_upload_sessions_state ON upload_sessions(state, expires_at);

CREATE TABLE IF NOT EXISTS rounds (
    job TEXT NOT NULL REFERENCES jobs(job) ON DELETE CASCADE,
    round_num INTEGER NOT NULL,
    base_ckpt_name TEXT NOT NULL,
    base_ckpt_sha256 TEXT,
    quorum INTEGER NOT NULL DEFAULT 4,
    total_slots INTEGER NOT NULL DEFAULT 4,
    status TEXT NOT NULL DEFAULT 'open',
    agg_method TEXT,
    agg_ckpt_name TEXT,
    agg_ckpt_sha256 TEXT,
    agg_metrics_json TEXT NOT NULL DEFAULT '{}',
    completed_at INTEGER,
    PRIMARY KEY (job, round_num)
);

CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    job TEXT,
    slot_id TEXT,
    status TEXT NOT NULL,
    detail_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_events(ts DESC);
"""


def init_db() -> None:
    ensure_dirs()
    c0 = connect(set_journal=True)
    try:
        c0.executescript(SCHEMA_SQL)
        # Idempotent additive migrations for existing DBs
        cols = {r["name"] for r in c0.execute("PRAGMA table_info(contributors)").fetchall()}
        if "is_owner" not in cols:
            c0.execute("ALTER TABLE contributors ADD COLUMN is_owner INTEGER NOT NULL DEFAULT 0;")
        if "denied" not in cols:
            c0.execute("ALTER TABLE contributors ADD COLUMN denied INTEGER NOT NULL DEFAULT 0;")
        if "github_id" not in cols:
            c0.execute("ALTER TABLE contributors ADD COLUMN github_id INTEGER;")
        if "github_login" not in cols:
            c0.execute("ALTER TABLE contributors ADD COLUMN github_login TEXT;")
        if "github_name" not in cols:
            c0.execute("ALTER TABLE contributors ADD COLUMN github_name TEXT;")
        c0.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_contributors_github_id "
            "ON contributors(github_id) WHERE github_id IS NOT NULL;"
        )
    finally:
        c0.close()

    with tx(immediate=True) as conn:
        now = utc_ms()
        # Bootstrap admin credential from mounted read-only secret file (separate from contributors)
        admin_key = config.load_bootstrap_admin_key()
        if admin_key:
            kh = hash_api_key(admin_key)
            conn.execute(
                """INSERT INTO contributors (login, key_hash, is_admin, is_owner, approved, denied, tiers_json, note, created_at)
                   VALUES ('admin', ?, 1, 1, 1, 0, '["admin"]', 'Bootstrap site operator', ?)
                   ON CONFLICT(login) DO UPDATE SET key_hash=excluded.key_hash, is_admin=1, is_owner=1, approved=1, denied=0""",
                (kh, now),
            )
        # Ensure existing owner logins and admins carry is_owner=1
        conn.execute("UPDATE contributors SET is_owner=1 WHERE is_admin=1 AND is_owner=0")
        if config.OWNER_LOGINS:
            placeholders = ",".join("?" for _ in config.OWNER_LOGINS)
            conn.execute(
                f"UPDATE contributors SET is_owner=1 WHERE login IN ({placeholders}) AND is_owner=0",
                tuple(sorted(config.OWNER_LOGINS)),
            )
        # Optional legacy bootstrap (off by default; never overwrites existing rows)
        for login, raw_key in config.load_bootstrap_worker_keys().items():
            if not raw_key:
                continue
            kh = hash_api_key(raw_key)
            is_own = 1 if login in config.OWNER_LOGINS else 0
            tiers = json.dumps(["tier-label"])
            conn.execute(
                """INSERT INTO contributors (login, key_hash, is_admin, is_owner, approved, denied, tiers_json, note, created_at)
                   VALUES (?, ?, 0, ?, 1, 0, ?, 'Bootstrapped worker', ?)
                   ON CONFLICT(login) DO NOTHING""",
                (login, kh, is_own, tiers, now),
            )

        # Idempotent migration of existing contributor key_hash rows into api_keys (preserves all existing keys!)
        existing_contribs = conn.execute(
            "SELECT login, key_hash, created_at, last_seen_at FROM contributors WHERE is_admin=0"
        ).fetchall()
        for row in existing_contribs:
            kh = str(row["key_hash"] or "")
            if not is_real_key_hash(kh):
                continue
            already = conn.execute("SELECT 1 FROM api_keys WHERE key_hash=?", (kh,)).fetchone()
            if already is None:
                kid = "key_" + secrets.token_hex(8)
                conn.execute(
                    """INSERT INTO api_keys (key_id, contributor_login, key_hash, key_prefix, label, created_at, last_used_at, revoked_at)
                       VALUES (?, ?, ?, 'drs_wrk_legacy', 'Initial key', ?, ?, NULL)""",
                    (kid, row["login"], kh, int(row["created_at"] or now), row["last_seen_at"]),
                )

        # Seed/bind pre-approved GitHub accounts (owner bertholomus + existing outside rows)
        for gh_login, spec in config.PREAPPROVED_GITHUB_ACCOUNTS.items():
            site_login = str(spec["site_login"])
            gh_id = spec.get("github_id")
            if gh_id is not None:
                bound = conn.execute("SELECT login FROM contributors WHERE github_id=?", (int(gh_id),)).fetchone()
                if bound is not None:
                    continue
            existing = conn.execute("SELECT * FROM contributors WHERE login=?", (site_login,)).fetchone()
            if existing is not None:
                if existing["github_id"] is None and gh_id is not None:
                    conn.execute(
                        "UPDATE contributors SET github_id=?, github_login=? WHERE login=? AND github_id IS NULL",
                        (int(gh_id), gh_login, site_login),
                    )
                if spec.get("is_owner") and not existing["is_owner"]:
                    conn.execute("UPDATE contributors SET is_owner=1 WHERE login=?", (site_login,))
            elif spec.get("is_owner"):
                # Seed the owner GitHub account (no worker key minted automatically)
                conn.execute(
                    """INSERT INTO contributors
                       (login, key_hash, is_admin, is_owner, approved, denied, tiers_json, strikes, quarantined, note, github_id, github_login, created_at)
                       VALUES (?, ?, 0, 1, 1, 0, ?, 0, 0, ?, ?, ?, ?)""",
                    (
                        site_login,
                        no_key_sentinel(),
                        json.dumps(spec.get("tiers") or ["tier-label"]),
                        spec.get("note") or "Owner account (GitHub)",
                        int(gh_id) if gh_id is not None else None,
                        gh_login,
                        now,
                    ),
                )

        sweep_expired_claims(conn)


def log_audit(
    conn: sqlite3.Connection,
    actor: str,
    action: str,
    status: str,
    job: str | None = None,
    slot_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        """INSERT INTO audit_events (ts, actor, action, job, slot_id, status, detail_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (utc_ms(), actor, action, job, slot_id, status, json.dumps(detail or {}, sort_keys=True)),
    )


def sweep_expired_sessions(conn: sqlite3.Connection) -> int:
    now = utc_ms()
    c1 = conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
    c2 = conn.execute("DELETE FROM oauth_states WHERE expires_at < ?", (now,))
    return int((c1.rowcount or 0) + (c2.rowcount or 0))


def sweep_expired_uploads(conn: sqlite3.Connection) -> int:
    """Expire stale upload sessions and clean abandoned staging dirs/files."""
    now = utc_ms()
    verify_stale_cutoff = now - max(config.VERIFY_SUBPROCESS_TIMEOUT_SEC * 2, 300) * 1000
    rows = conn.execute(
        """SELECT u.upload_id, u.claim_id, u.state, u.expires_at, u.created_at, s.state AS slot_state, s.claim_id AS slot_claim
           FROM upload_sessions u
           LEFT JOIN slots s ON u.job = s.job AND u.slot_id = s.slot_id
           WHERE u.state IN ('open', 'verifying')"""
    ).fetchall()

    expired = 0
    active_ids: set[str] = set()
    for r in rows:
        uid = r["upload_id"]
        claim_dead = (r["slot_state"] != "claimed") or (r["slot_claim"] != r["claim_id"])
        is_expired = (int(r["expires_at"]) < now) or claim_dead
        if r["state"] == "verifying":
            is_expired = is_expired and (int(r["created_at"]) < verify_stale_cutoff)
        if is_expired:
            conn.execute("UPDATE upload_sessions SET state='expired' WHERE upload_id=?", (uid,))
            sdir = config.STAGING_DIR / uid
            if sdir.exists():
                shutil.rmtree(sdir, ignore_errors=True)
            assembled = config.STAGING_DIR / f"{uid}_assembled.tmp"
            assembled.unlink(missing_ok=True)
            expired += 1
        else:
            active_ids.add(uid)

    if config.STAGING_DIR.is_dir():
        cutoff_sec = time.time() - config.STAGING_ORPHAN_TTL_SEC
        for child in list(config.STAGING_DIR.iterdir()):
            name = child.name
            if name in active_ids or any(name.startswith(aid + "_") for aid in active_ids):
                continue
            try:
                if child.stat().st_mtime < cutoff_sec:
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
                    expired += 1
            except FileNotFoundError:
                pass
    return expired


def sweep_expired_claims(conn: sqlite3.Connection) -> int:
    now = utc_ms()
    rows = conn.execute(
        "SELECT job, slot_id, claimed_by, claim_id FROM slots WHERE state='claimed' AND expires_at IS NOT NULL AND expires_at < ?",
        (now,),
    ).fetchall()
    for r in rows:
        conn.execute(
            """UPDATE slots
               SET state='open', claim_id=NULL, claimed_by=NULL, claimed_at=NULL, expires_at=NULL, host_tag=NULL, request_id=NULL
               WHERE job=? AND slot_id=? AND state='claimed'""",
            (r["job"], r["slot_id"]),
        )
        log_audit(conn, "system", "claim_expired", "ok", r["job"], r["slot_id"], {"prev_holder": r["claimed_by"]})
    sweep_expired_uploads(conn)
    sweep_expired_sessions(conn)
    return len(rows)


def lookup_auth(
    conn: sqlite3.Connection,
    raw_token: str | None,
    *,
    touch_last_seen: bool = False,
) -> sqlite3.Row | None:
    if not raw_token:
        return None
    kh = hash_api_key(raw_token)
    # 1. Bootstrap admin credential (strictly separate from contributor api_keys)
    admin_row = conn.execute(
        "SELECT *, NULL AS auth_key_id, NULL AS auth_key_last_used_at FROM contributors WHERE is_admin=1 AND key_hash=?",
        (kh,),
    ).fetchone()
    if admin_row is not None and hmac.compare_digest(admin_row["key_hash"], kh):
        if touch_last_seen:
            now = utc_ms()
            prev = admin_row["last_seen_at"]
            if prev is None or (now - int(prev)) >= 60_000:
                conn.execute("UPDATE contributors SET last_seen_at=? WHERE login=?", (now, admin_row["login"]))
        return admin_row

    # 2. Contributor worker key in api_keys (revoked keys rejected immediately)
    row = conn.execute(
        """SELECT c.*, k.key_id AS auth_key_id, k.key_hash AS auth_key_hash, k.last_used_at AS auth_key_last_used_at
           FROM api_keys k
           JOIN contributors c ON k.contributor_login = c.login
           WHERE k.key_hash=? AND k.revoked_at IS NULL""",
        (kh,),
    ).fetchone()
    if row is not None and hmac.compare_digest(row["auth_key_hash"], kh):
        if touch_last_seen:
            now = utc_ms()
            prev_u = row["last_seen_at"]
            if prev_u is None or (now - int(prev_u)) >= 60_000:
                conn.execute("UPDATE contributors SET last_seen_at=? WHERE login=?", (now, row["login"]))
            prev_k = row["auth_key_last_used_at"]
            if prev_k is None or (now - int(prev_k)) >= 60_000:
                conn.execute("UPDATE api_keys SET last_used_at=? WHERE key_id=?", (now, row["auth_key_id"]))
        return row
    return None


def touch_key_usage_if_stale(key_id: str | None, login: str, last_used_at: int | None) -> None:
    """Update api_keys.last_used_at and contributors.last_seen_at when null or >= 60s old."""
    if not key_id:
        return
    now = utc_ms()
    if last_used_at is not None and (now - int(last_used_at)) < 60_000:
        return
    try:
        with tx(immediate=True) as conn:
            conn.execute("UPDATE api_keys SET last_used_at=? WHERE key_id=? AND revoked_at IS NULL", (now, key_id))
            conn.execute("UPDATE contributors SET last_seen_at=? WHERE login=?", (now, login))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# OAuth State & Server-Side Session Store
# ---------------------------------------------------------------------------

def create_oauth_state(conn: sqlite3.Connection, state: str, browser_nonce: str, return_to: str) -> None:
    now = utc_ms()
    sweep_expired_sessions(conn)
    conn.execute(
        """INSERT INTO oauth_states (state_hash, nonce_hash, return_to, created_at, expires_at)
           VALUES (?, ?, ?, ?, ?)""",
        (
            hmac_token("oauth_state", state),
            hmac_token("oauth_nonce", browser_nonce),
            return_to,
            now,
            now + config.OAUTH_STATE_TTL_SEC * 1000,
        ),
    )


def consume_oauth_state(conn: sqlite3.Connection, state: str, browser_nonce: str) -> str | None:
    """Atomically verify and delete a single-use OAuth state bound to the browser nonce."""
    if not state or not browser_nonce:
        return None
    now = utc_ms()
    sweep_expired_sessions(conn)
    sh = hmac_token("oauth_state", state)
    row = conn.execute("SELECT * FROM oauth_states WHERE state_hash=?", (sh,)).fetchone()
    if row is None:
        return None
    # Burn immediately so state can NEVER be replayed even if nonce check fails
    conn.execute("DELETE FROM oauth_states WHERE state_hash=?", (sh,))
    if int(row["expires_at"]) < now:
        return None
    expected_nh = hmac_token("oauth_nonce", browser_nonce)
    if not hmac.compare_digest(str(row["nonce_hash"]), expected_nh):
        return None
    return str(row["return_to"] or "/")


def create_session(
    conn: sqlite3.Connection,
    contributor_login: str,
    *,
    old_session_token: str | None = None,
) -> tuple[str, str]:
    """Create a fresh rotated server-side session (deleting any prior session to block fixation)."""
    if old_session_token:
        old_hash = hmac_token("session", old_session_token)
        conn.execute("DELETE FROM sessions WHERE session_hash=?", (old_hash,))
    now = utc_ms()
    session_id = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(32)
    sh = hmac_token("session", session_id)
    expires_at = now + config.SESSION_IDLE_TTL_SEC * 1000
    conn.execute(
        """INSERT INTO sessions (session_hash, contributor_login, csrf_token, created_at, last_seen_at, expires_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (sh, contributor_login, csrf_token, now, now, expires_at),
    )
    return session_id, csrf_token


def lookup_session(
    conn: sqlite3.Connection,
    session_token: str | None,
    *,
    touch: bool = False,
) -> sqlite3.Row | None:
    if not session_token:
        return None
    sh = hmac_token("session", session_token)
    row = conn.execute(
        """SELECT s.session_hash, s.csrf_token, s.created_at AS session_created_at,
                  s.last_seen_at AS session_last_seen_at, s.expires_at AS session_expires_at,
                  c.*
           FROM sessions s
           JOIN contributors c ON s.contributor_login = c.login
           WHERE s.session_hash=?""",
        (sh,),
    ).fetchone()
    if row is None or not hmac.compare_digest(str(row["session_hash"]), sh):
        return None
    now = utc_ms()
    if int(row["session_expires_at"]) < now:
        return None
    if touch:
        prev = int(row["session_last_seen_at"])
        if (now - prev) >= 300_000:
            conn.execute(
                "UPDATE sessions SET last_seen_at=?, expires_at=? WHERE session_hash=?",
                (now, now + config.SESSION_IDLE_TTL_SEC * 1000, sh),
            )
    return row


def touch_session_if_stale(session_hash: str, session_last_seen_at: int) -> None:
    now = utc_ms()
    if (now - int(session_last_seen_at)) < 300_000:
        return
    try:
        with tx(immediate=True) as conn:
            conn.execute(
                "UPDATE sessions SET last_seen_at=?, expires_at=? WHERE session_hash=?",
                (now, now + config.SESSION_IDLE_TTL_SEC * 1000, session_hash),
            )
    except Exception:
        pass


def delete_session(conn: sqlite3.Connection, session_token: str | None) -> bool:
    if not session_token:
        return False
    sh = hmac_token("session", session_token)
    cur = conn.execute("DELETE FROM sessions WHERE session_hash=?", (sh,))
    return bool(cur.rowcount)


def delete_user_sessions(conn: sqlite3.Connection, contributor_login: str) -> int:
    cur = conn.execute("DELETE FROM sessions WHERE contributor_login=?", (contributor_login,))
    return int(cur.rowcount or 0)


# ---------------------------------------------------------------------------
# GitHub Account Resolution & Self-Service API Keys
# ---------------------------------------------------------------------------

def resolve_or_create_github_contributor(
    conn: sqlite3.Connection,
    *,
    github_id: int,
    github_login: str,
    github_name: str | None = None,
) -> tuple[sqlite3.Row, bool]:
    """Key account strictly on GitHub numeric user id; update display login; map pre-approved rows."""
    now = utc_ms()
    gh_login = github_login.strip()[:100]
    gh_name = (github_name or "").strip()[:128] or None

    # 1. Existing contributor keyed by GitHub numeric user id (survives GitHub username renames!)
    existing = conn.execute("SELECT * FROM contributors WHERE github_id=?", (github_id,)).fetchone()
    if existing is not None:
        conn.execute(
            "UPDATE contributors SET github_login=?, github_name=?, last_seen_at=? WHERE login=?",
            (gh_login, gh_name, now, existing["login"]),
        )
        return conn.execute("SELECT * FROM contributors WHERE login=?", (existing["login"],)).fetchone(), False

    # 2. Match pre-approved mapping (strictly by numeric github_id if configured; login match ONLY if id is unset)
    matched_spec: dict[str, Any] | None = None
    for spec_login, spec in config.PREAPPROVED_GITHUB_ACCOUNTS.items():
        spec_gid = spec.get("github_id")
        if spec_gid is not None:
            if int(spec_gid) == github_id:
                matched_spec = spec
                break
        elif spec_login.lower() == gh_login.lower():
            matched_spec = spec
            break

    if matched_spec is not None:
        site_login = str(matched_spec["site_login"])
        row_by_login = conn.execute("SELECT * FROM contributors WHERE login=?", (site_login,)).fetchone()
        if row_by_login is not None and row_by_login["github_id"] is None:
            conn.execute(
                "UPDATE contributors SET github_id=?, github_login=?, github_name=?, last_seen_at=? WHERE login=?",
                (github_id, gh_login, gh_name, now, site_login),
            )
            return conn.execute("SELECT * FROM contributors WHERE login=?", (site_login,)).fetchone(), False
        if row_by_login is None:
            is_own = 1 if matched_spec.get("is_owner") else 0
            tiers = json.dumps(matched_spec.get("tiers") or ["tier-label"])
            conn.execute(
                """INSERT INTO contributors
                   (login, key_hash, is_admin, is_owner, approved, denied, tiers_json, strikes, quarantined, note, github_id, github_login, github_name, created_at, last_seen_at)
                   VALUES (?, ?, 0, ?, 1, 0, ?, 0, 0, ?, ?, ?, ?, ?, ?)""",
                (
                    site_login,
                    no_key_sentinel(),
                    is_own,
                    tiers,
                    matched_spec.get("note") or "Pre-approved GitHub contributor",
                    github_id,
                    gh_login,
                    gh_name,
                    now,
                    now,
                ),
            )
            return conn.execute("SELECT * FROM contributors WHERE login=?", (site_login,)).fetchone(), False

    # 3. Unknown GitHub user -> pending request (approved=0, denied=0, no key, no claims)
    candidate = gh_login if (SAFE_LOGIN_RE.match(gh_login) and gh_login.lower() not in RESERVED_LOGINS) else f"gh_{github_id}"
    if conn.execute("SELECT 1 FROM contributors WHERE login=?", (candidate,)).fetchone() is not None:
        candidate = f"gh_{github_id}"
    conn.execute(
        """INSERT INTO contributors
           (login, key_hash, is_admin, is_owner, approved, denied, tiers_json, strikes, quarantined, note, github_id, github_login, github_name, created_at, last_seen_at)
           VALUES (?, ?, 0, 0, 0, 0, '["tier-label"]', 0, 0, 'Pending GitHub access request', ?, ?, ?, ?, ?)""",
        (candidate, no_key_sentinel(), github_id, gh_login, gh_name, now, now),
    )
    return conn.execute("SELECT * FROM contributors WHERE login=?", (candidate,)).fetchone(), True


def list_active_api_keys(conn: sqlite3.Connection, contributor_login: str) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT key_id, contributor_login, key_prefix, label, created_at, last_used_at, revoked_at
           FROM api_keys
           WHERE contributor_login=? AND revoked_at IS NULL
           ORDER BY created_at ASC""",
        (contributor_login,),
    ).fetchall()


def _sync_contributor_primary_key_hash(conn: sqlite3.Connection, contributor_login: str, revoked_hash: str | None = None) -> None:
    """Keep contributors.key_hash clear of revoked hashes (and pointing at an active key if one exists)."""
    row = conn.execute("SELECT key_hash FROM contributors WHERE login=?", (contributor_login,)).fetchone()
    if row is None:
        return
    cur_hash = str(row["key_hash"] or "")
    if revoked_hash is not None and cur_hash == revoked_hash:
        other = conn.execute(
            "SELECT key_hash FROM api_keys WHERE contributor_login=? AND revoked_at IS NULL ORDER BY created_at ASC LIMIT 1",
            (contributor_login,),
        ).fetchone()
        new_hash = str(other["key_hash"]) if other is not None else no_key_sentinel()
        conn.execute("UPDATE contributors SET key_hash=? WHERE login=?", (new_hash, contributor_login))
    elif not is_real_key_hash(cur_hash):
        first = conn.execute(
            "SELECT key_hash FROM api_keys WHERE contributor_login=? AND revoked_at IS NULL ORDER BY created_at ASC LIMIT 1",
            (contributor_login,),
        ).fetchone()
        if first is not None:
            conn.execute("UPDATE contributors SET key_hash=? WHERE login=?", (str(first["key_hash"]), contributor_login))


def create_api_key(
    conn: sqlite3.Connection,
    contributor_login: str,
    *,
    label: str = "",
    max_live: int | None = None,
) -> tuple[dict[str, Any], str]:
    limit = config.MAX_LIVE_KEYS_PER_ACCOUNT if max_live is None else max_live
    live = int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM api_keys WHERE contributor_login=? AND revoked_at IS NULL",
            (contributor_login,),
        ).fetchone()["n"]
    )
    if live >= limit:
        raise ValueError(f"maximum {limit} live API keys per account")
    raw_key = "drs_wrk_" + secrets.token_urlsafe(32)
    kh = hash_api_key(raw_key)
    pref = key_prefix(raw_key)
    kid = "key_" + secrets.token_hex(8)
    clean_label = (label or "").strip()[:64]
    now = utc_ms()
    conn.execute(
        """INSERT INTO api_keys (key_id, contributor_login, key_hash, key_prefix, label, created_at, last_used_at, revoked_at)
           VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)""",
        (kid, contributor_login, kh, pref, clean_label, now),
    )
    _sync_contributor_primary_key_hash(conn, contributor_login)
    return {
        "key_id": kid,
        "prefix": pref,
        "label": clean_label,
        "created_at": now,
        "created_utc": utc_iso(now),
        "last_used_at": None,
        "last_used_utc": None,
    }, raw_key


def rotate_api_key(
    conn: sqlite3.Connection,
    contributor_login: str,
    key_id: str,
    *,
    label: str | None = None,
) -> tuple[dict[str, Any], str] | None:
    old = conn.execute(
        "SELECT * FROM api_keys WHERE key_id=? AND contributor_login=? AND revoked_at IS NULL",
        (key_id, contributor_login),
    ).fetchone()
    if old is None:
        return None
    now = utc_ms()
    conn.execute("UPDATE api_keys SET revoked_at=? WHERE key_id=?", (now, key_id))
    _sync_contributor_primary_key_hash(conn, contributor_login, revoked_hash=str(old["key_hash"]))

    new_label = (label if label is not None else str(old["label"] or "")).strip()[:64]
    raw_key = "drs_wrk_" + secrets.token_urlsafe(32)
    kh = hash_api_key(raw_key)
    pref = key_prefix(raw_key)
    new_kid = "key_" + secrets.token_hex(8)
    conn.execute(
        """INSERT INTO api_keys (key_id, contributor_login, key_hash, key_prefix, label, created_at, last_used_at, revoked_at)
           VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)""",
        (new_kid, contributor_login, kh, pref, new_label, now),
    )
    _sync_contributor_primary_key_hash(conn, contributor_login)
    return {
        "key_id": new_kid,
        "revoked_key_id": key_id,
        "prefix": pref,
        "label": new_label,
        "created_at": now,
        "created_utc": utc_iso(now),
        "last_used_at": None,
        "last_used_utc": None,
    }, raw_key


def revoke_api_key(conn: sqlite3.Connection, contributor_login: str, key_id: str) -> dict[str, Any] | None:
    old = conn.execute(
        "SELECT * FROM api_keys WHERE key_id=? AND contributor_login=? AND revoked_at IS NULL",
        (key_id, contributor_login),
    ).fetchone()
    if old is None:
        return None
    now = utc_ms()
    conn.execute("UPDATE api_keys SET revoked_at=? WHERE key_id=?", (now, key_id))
    _sync_contributor_primary_key_hash(conn, contributor_login, revoked_hash=str(old["key_hash"]))
    return {"key_id": key_id, "prefix": str(old["key_prefix"]), "revoked_at": now}


def revoke_all_api_keys(conn: sqlite3.Connection, contributor_login: str) -> int:
    now = utc_ms()
    cur = conn.execute(
        "UPDATE api_keys SET revoked_at=? WHERE contributor_login=? AND revoked_at IS NULL",
        (now, contributor_login),
    )
    conn.execute(
        "UPDATE contributors SET key_hash=? WHERE login=? AND is_admin=0",
        (no_key_sentinel(), contributor_login),
    )
    return int(cur.rowcount or 0)


def trigger_downstream_readiness(
    conn: sqlite3.Connection,
    upstream_job: str,
    upstream_slot: str,
    verified_artifact_id: str,
) -> list[dict[str, str]]:
    """Flip any not_ready downstream slots depending on (upstream_job, upstream_slot) to open."""
    art = conn.execute("SELECT * FROM artifacts WHERE artifact_id=?", (verified_artifact_id,)).fetchone()
    if art is None:
        return []
    deps = conn.execute(
        """SELECT job, slot_id, inputs_json FROM slots
           WHERE state='not_ready' AND depends_on_job=? AND depends_on_slot=?""",
        (upstream_job, upstream_slot),
    ).fetchall()
    opened: list[dict[str, str]] = []
    for d in deps:
        inputs = json.loads(d["inputs_json"] or "[]")
        inputs.append(
            {
                "name": "verified_upstream_labels",
                "artifact_id": art["artifact_id"],
                "filename": art["filename"],
                "sha256": art["sha256"],
                "bytes": art["bytes"],
                "source_slot": f"{upstream_job}/{upstream_slot}",
            }
        )
        conn.execute(
            "UPDATE slots SET state='open', inputs_json=? WHERE job=? AND slot_id=?",
            (json.dumps(inputs), d["job"], d["slot_id"]),
        )
        log_audit(
            conn,
            "system",
            "readiness_opened",
            "ok",
            d["job"],
            d["slot_id"],
            {"upstream": f"{upstream_job}/{upstream_slot}"},
        )
        opened.append({"job": d["job"], "slot_id": d["slot_id"]})
    return opened
