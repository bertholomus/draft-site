"""SQLite WAL state store for jobs, slots, claims, uploads, verifications, rounds, and audit logs."""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from server import config


def utc_ms() -> int:
    return int(time.time() * 1000)


def utc_iso(ts_ms: int | None = None) -> str:
    t = (ts_ms / 1000.0) if ts_ms is not None else time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def hash_api_key(raw_key: str) -> str:
    """Deterministic SHA-256 with domain separation (keys are 256-bit random tokens)."""
    return hashlib.sha256(("drs-v1:" + raw_key).encode("utf-8")).hexdigest()


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


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.DB_PATH), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=15000;")
    return conn


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
    approved INTEGER NOT NULL DEFAULT 1,
    tiers_json TEXT NOT NULL DEFAULT '[]',
    strikes INTEGER NOT NULL DEFAULT 0,
    quarantined INTEGER NOT NULL DEFAULT 0,
    note TEXT,
    created_at INTEGER NOT NULL,
    last_seen_at INTEGER
);

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
    c0 = connect()
    try:
        c0.executescript(SCHEMA_SQL)
    finally:
        c0.close()
    with tx(immediate=True) as conn:
        now = utc_ms()
        # Bootstrap admin credential from mounted read-only secret file (separate from contributors)
        admin_key = config.load_bootstrap_admin_key()
        if admin_key:
            kh = hash_api_key(admin_key)
            conn.execute(
                """INSERT INTO contributors (login, key_hash, is_admin, approved, tiers_json, note, created_at)
                   VALUES ('admin', ?, 1, 1, '["admin"]', 'Bootstrap site operator', ?)
                   ON CONFLICT(login) DO UPDATE SET key_hash=excluded.key_hash, is_admin=1, approved=1""",
                (kh, now),
            )
        # Bootstrap closed-pilot internal machine keys (neutral public logins; private host map stays admin-only)
        tier_defaults = {
            "bertholomus-w1": (["tier-label", "tier-eval", "tier-cpu"], "private-host: host-1 (NAS CPU worker)"),
            "bertholomus-w2": (["tier-label", "tier-eval", "tier-cpu"], "private-host: host-2 (macOS workstation)"),
            "bertholomus-w3": (["tier-train", "tier-eval", "tier-capture"], "private-host: host-3 (96GB Blackwell GPU)"),
            "bertholomus-w4": (["tier-train", "tier-eval", "tier-label"], "private-host: host-4 (128GB GB10 GPU)"),
            "bertholomus-qa": (["tier-train", "tier-eval", "tier-label", "tier-capture"], "private-host: internal verification QA"),
        }
        for login, raw_key in config.load_bootstrap_worker_keys().items():
            if not raw_key:
                continue
            kh = hash_api_key(raw_key)
            tiers_list, note = tier_defaults.get(login, (["tier-eval"], "Closed-pilot internal worker"))
            tiers = json.dumps(tiers_list)
            conn.execute(
                """INSERT INTO contributors (login, key_hash, is_admin, approved, tiers_json, note, created_at)
                   VALUES (?, ?, 0, 1, ?, ?, ?)
                   ON CONFLICT(login) DO UPDATE SET key_hash=excluded.key_hash, tiers_json=excluded.tiers_json, note=excluded.note""",
                (login, kh, tiers, note, now),
            )


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
    return len(rows)


def lookup_auth(conn: sqlite3.Connection, raw_token: str | None) -> sqlite3.Row | None:
    if not raw_token:
        return None
    kh = hash_api_key(raw_token)
    row = conn.execute("SELECT * FROM contributors WHERE key_hash=?", (kh,)).fetchone()
    if row is not None and hmac.compare_digest(row["key_hash"], kh):
        conn.execute("UPDATE contributors SET last_seen_at=? WHERE login=?", (utc_ms(), row["login"]))
        return row
    return None


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
            {"upstream": f"{upstream_job}/{upstream_slot}", "artifact_id": art["artifact_id"]},
        )
        opened.append({"job": d["job"], "slot_id": d["slot_id"]})
    return opened
