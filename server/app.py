"""FastAPI application for draft.bertholomus.com (modelled on quant.kindlingai.dev + CDC PIPELINE.md)."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

from server import config, db, ui, verify

SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

app = FastAPI(title="draft.bertholomus.com", docs_url=None, redoc_url=None, openapi_url=None)


@app.on_event("startup")
def _startup() -> None:
    db.init_db()


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


def _require_auth(conn: sqlite3.Connection, req: Request, *, admin: bool = False) -> sqlite3.Row:
    tok = _extract_bearer(req)
    user = db.lookup_auth(conn, tok)
    if user is None:
        raise HTTPException(status_code=401, detail="valid Authorization: Bearer <key> required")
    if admin:
        if not user["is_admin"]:
            raise HTTPException(status_code=403, detail="admin credential required")
        return user
    if not user["approved"] or user["quarantined"]:
        raise HTTPException(status_code=403, detail="contributor account is not approved or is quarantined")
    return user


def _safe_under(base: Path, target: Path) -> Path:
    res = target.resolve()
    if not res.is_relative_to(base.resolve()):
        raise HTTPException(status_code=400, detail="invalid storage path")
    return res


@app.middleware("http")
async def security_and_origin_middleware(request: Request, call_next):
    # Enforce shared origin auth secret when requests arrive via Cloudflare Tunnel hostname
    host = (request.headers.get("host") or "").split(":")[0].lower()
    if config.REQUIRE_ORIGIN_AUTH and host == config.ORIGIN_HOST and request.url.path != "/healthz":
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
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


# ---------------------------------------------------------------------------
# Health & Landing UI
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "service": "draft-site", "utc": db.utc_iso()}


@app.get("/NOTICE", response_class=PlainTextResponse)
def notice_file() -> str:
    p = Path("/app/NOTICE")
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return "Apache-2.0 NOTICE — see repository NOTICE file."


@app.get("/", response_class=HTMLResponse)
def index_page() -> HTMLResponse:
    return HTMLResponse(
        content=ui.render_index_html(),
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "script-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "connect-src 'self'; "
                "frame-ancestors 'none'; base-uri 'self'"
            ),
        },
    )


# ---------------------------------------------------------------------------
# Public Read API (X-Quant: 1)
# ---------------------------------------------------------------------------

@app.get("/api/encoders")
@app.get("/api/runners")
def list_runners(request: Request) -> list[dict[str, Any]]:
    _require_x_quant(request)
    with db.tx() as conn:
        rows = conn.execute(
            "SELECT commit_id, kind, image_digest, note, added_at FROM runners WHERE active=1 ORDER BY added_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]


@app.get("/api/jobs")
def list_jobs(request: Request) -> list[dict[str, Any]]:
    _require_x_quant(request)
    with db.tx(immediate=True) as conn:
        db.sweep_expired_claims(conn)
        jobs = conn.execute("SELECT * FROM jobs ORDER BY created_at ASC").fetchall()
        out: list[dict[str, Any]] = []
        for j in jobs:
            slots = conn.execute("SELECT state, uploaded_by, uploaded_at, harvested FROM slots WHERE job=?", (j["job"],)).fetchall()
            counts = {k: 0 for k in ("not_ready", "open", "claimed", "uploaded", "verified", "rejected", "disabled")}
            contribs: set[str] = set()
            last_upload = None
            harvested = 0
            for s in slots:
                st = s["state"]
                counts[st] = counts.get(st, 0) + 1
                if st == "verified" and s["uploaded_by"]:
                    contribs.add(s["uploaded_by"])
                if s["uploaded_at"] and (last_upload is None or s["uploaded_at"] > last_upload):
                    last_upload = s["uploaded_at"]
                if s["harvested"]:
                    harvested += 1
            cfg = json.loads(j["config_json"] or "{}")
            out.append(
                {
                    "job": j["job"],
                    "title": j["title"],
                    "job_kind": j["job_kind"],
                    "model": j["model"],
                    "version": j["version"],
                    "active_round": j["active_round"],
                    "slots": len(slots),
                    "uploaded": counts["uploaded"] + counts["verified"],
                    "verified": counts["verified"],
                    "pending_verify": counts["uploaded"],
                    "claimed": counts["claimed"],
                    "open": counts["open"],
                    "not_ready": counts["not_ready"],
                    "rejected": counts["rejected"],
                    "disabled": counts["disabled"],
                    "harvested": harvested,
                    "contributors": len(contribs),
                    "last_upload": last_upload,
                    "weights": {
                        "hf": j["release_repo"],
                        "revision": j["release_revision"],
                    },
                    "config": cfg,
                }
            )
        return out


@app.get("/api/grid")
def get_grid(request: Request, job: str) -> dict[str, Any]:
    _require_x_quant(request)
    _validate_id(job, "job")
    with db.tx(immediate=True) as conn:
        db.sweep_expired_claims(conn)
        me = db.lookup_auth(conn, _extract_bearer(request))
        my_login = me["login"] if me else None
        j = conn.execute("SELECT * FROM jobs WHERE job=?", (job,)).fetchone()
        if j is None:
            raise HTTPException(status_code=404, detail=f"unknown job '{job}'")
        all_jobs = [r["job"] for r in conn.execute("SELECT job FROM jobs ORDER BY created_at ASC").fetchall()]
        slot_rows = conn.execute(
            "SELECT * FROM slots WHERE job=? ORDER BY round_num ASC, row_key ASC, col_key ASC, slot_id ASC",
            (job,),
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
                    "mine": mine,
                    "depends_on": f"{s['depends_on_job']}/{s['depends_on_slot']}" if s["depends_on_job"] else None,
                    "upload_bytes": s["upload_bytes"],
                    "upload_sha256": s["upload_sha256"],
                    "verify_status": s["verify_status"],
                    "verify_reason": s["verify_reason"],
                    "verified_at": s["verified_at"],
                    "metrics": {k: v for k, v in metrics.items() if k not in ("host_tag", "engines")},
                    "plan_summary": {k: plan[k] for k in ("codec", "profile_id", "checkpoint_rev", "local_steps", "n_prompts", "tap_layers") if k in plan},
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
                    "contributor_login": holder,
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
            "slots": slots_out,
        }


@app.get("/api/contributors")
def list_contributors(request: Request) -> list[dict[str, Any]]:
    _require_x_quant(request)
    with db.tx() as conn:
        contribs = conn.execute(
            "SELECT login, approved, tiers_json, strikes, quarantined, last_seen_at FROM contributors WHERE is_admin=0 ORDER BY login ASC"
        ).fetchall()
        out: list[dict[str, Any]] = []
        for c in contribs:
            rows = conn.execute(
                """SELECT j.job_kind, s.state, s.upload_bytes, s.host_tag
                   FROM slots s JOIN jobs j ON s.job = j.job
                   WHERE s.uploaded_by=?""",
                (c["login"],),
            ).fetchall()
            verified_total = 0
            rejected_total = 0
            by_kind: dict[str, int] = {}
            total_bytes = 0
            hosts: set[str] = set()
            for r in rows:
                if r["host_tag"]:
                    hosts.add(r["host_tag"])
                if r["state"] == "verified":
                    verified_total += 1
                    by_kind[r["job_kind"]] = by_kind.get(r["job_kind"], 0) + 1
                    total_bytes += int(r["upload_bytes"] or 0)
                elif r["state"] == "rejected":
                    rejected_total += 1
            # Also count audit rejections (when a rejected slot was reopened)
            audit_rej = conn.execute(
                "SELECT COUNT(*) AS n FROM audit_events WHERE actor=? AND action IN ('upload_rejected','verification_rejected')",
                (c["login"],),
            ).fetchone()["n"]
            out.append(
                {
                    "login": c["login"],
                    "approved": bool(c["approved"]),
                    "quarantined": bool(c["quarantined"]),
                    "tiers": json.loads(c["tiers_json"] or "[]"),
                    "strikes": c["strikes"],
                    "verified_slots": verified_total,
                    "rejected_attempts": max(rejected_total, int(audit_rej)),
                    "by_kind": by_kind,
                    "total_bytes_uploaded": total_bytes,
                    "last_seen_at": c["last_seen_at"],
                }
            )
        out.sort(key=lambda x: (-x["verified_slots"], x["login"]))
        return out


@app.get("/api/audit")
def list_recent_audit(request: Request, limit: int = 50) -> list[dict[str, Any]]:
    _require_x_quant(request)
    lim = max(1, min(200, int(limit)))
    with db.tx() as conn:
        rows = conn.execute(
            "SELECT id, ts, actor, action, job, slot_id, status, detail_json FROM audit_events ORDER BY id DESC LIMIT ?",
            (lim,),
        ).fetchall()
        out = []
        for r in rows:
            det = json.loads(r["detail_json"] or "{}")
            # Never expose private host tags in public audit log
            det.pop("host_tag", None)
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
    with db.tx() as conn:
        tok = _extract_bearer(request)
        user = db.lookup_auth(conn, tok)
        if user is None:
            raise HTTPException(status_code=401, detail="valid Authorization: Bearer <key> required")
        vcount = conn.execute(
            "SELECT COUNT(*) AS n FROM slots WHERE uploaded_by=? AND state='verified'", (user["login"],)
        ).fetchone()["n"]
        return {
            "login": user["login"],
            "approved": bool(user["approved"]),
            "admin": bool(user["is_admin"]),
            "quarantined": bool(user["quarantined"]),
            "tiers": json.loads(user["tiers_json"] or "[]"),
            "strikes": user["strikes"],
            "verified_slots": int(vcount),
        }


def _format_claim_response(conn: sqlite3.Connection, slot: sqlite3.Row, job_row: sqlite3.Row) -> dict[str, Any]:
    inputs_raw = json.loads(slot["inputs_json"] or "[]")
    inputs_out = []
    for item in inputs_raw:
        entry = dict(item)
        if "artifact_id" in entry:
            entry["url"] = f"{config.PUBLIC_URL}/api/artifacts/{entry['artifact_id']}"
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
            "direct_url": f"{config.PUBLIC_URL}/api/uploads/direct/{slot['claim_id']}",
            "init_url": f"{config.PUBLIC_URL}/api/uploads/init",
            "max_bytes": max_bytes,
            "max_direct_bytes": config.MAX_DIRECT_UPLOAD_BYTES,
            "chunk_bytes": config.MAX_CHUNK_BYTES,
        },
    }


@app.get("/api/claims")
def list_my_claims(request: Request) -> list[dict[str, Any]]:
    _require_x_quant(request)
    with db.tx(immediate=True) as conn:
        db.sweep_expired_claims(conn)
        user = _require_auth(conn, request)
        rows = conn.execute(
            "SELECT * FROM slots WHERE state='claimed' AND claimed_by=? ORDER BY claimed_at ASC",
            (user["login"],),
        ).fetchall()
        out = []
        for s in rows:
            j = conn.execute("SELECT * FROM jobs WHERE job=?", (s["job"],)).fetchone()
            out.append(_format_claim_response(conn, s, j))
        return out


@app.post("/api/claims", status_code=201)
async def create_claim(request: Request) -> JSONResponse:
    _require_x_quant(request)
    with db.tx() as conn:
        _require_auth(conn, request)
    body = await request.json()
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
        user = _require_auth(conn, request)

        # Verify runner commit is accepted
        runner = conn.execute("SELECT * FROM runners WHERE commit_id=? AND active=1", (runner_commit,)).fetchone()
        if runner is None:
            raise HTTPException(status_code=400, detail=f"runner commit '{runner_commit}' not in /api/runners")

        # Idempotent retry for same request_id by same contributor
        prev = conn.execute(
            "SELECT * FROM slots WHERE request_id=? AND claimed_by=? AND state='claimed'",
            (req_id, user["login"]),
        ).fetchone()
        if prev is not None:
            j = conn.execute("SELECT * FROM jobs WHERE job=?", (prev["job"],)).fetchone()
            return JSONResponse(status_code=201, content=_format_claim_response(conn, prev, j))

        # Select target slot
        if req_job and req_slot:
            slot = conn.execute("SELECT * FROM slots WHERE job=? AND slot_id=?", (req_job, req_slot)).fetchone()
            if slot is None:
                raise HTTPException(status_code=404, detail=f"slot {req_job}/{req_slot} not found")
            if slot["state"] != "open":
                raise HTTPException(status_code=409, detail=f"slot {req_slot} is {slot['state']} (taken or not ready)")
        else:
            q = "SELECT s.* FROM slots s JOIN jobs j ON s.job=j.job WHERE s.state='open'"
            params: list[Any] = []
            if req_job:
                q += " AND s.job=?"
                params.append(req_job)
            if req_profile:
                q += " AND s.profile_id=?"
                params.append(str(req_profile))
            if req_round is not None:
                q += " AND s.round_num=?"
                params.append(int(req_round))
            if capabilities:
                # Filter by job_kind if caller specifies kinds in capabilities
                kind_caps = [c for c in capabilities if str(c).startswith("drafter-")]
                if kind_caps:
                    placeholders = ",".join("?" for _ in kind_caps)
                    q += f" AND j.job_kind IN ({placeholders})"
                    params.extend(kind_caps)
            q += " ORDER BY s.round_num ASC, s.row_key ASC, s.col_key ASC, s.slot_id ASC LIMIT 1"
            slot = conn.execute(q, params).fetchone()
            if slot is None:
                raise HTTPException(status_code=409, detail="no open slots matching filter")

        job_row = conn.execute("SELECT * FROM jobs WHERE job=?", (slot["job"],)).fetchone()
        if runner["kind"] not in ("all", job_row["job_kind"]):
            raise HTTPException(status_code=400, detail=f"runner kind '{runner['kind']}' cannot claim '{job_row['job_kind']}'")

        now = utc_ms = db.utc_ms()
        claim_id = "clm_" + secrets.token_hex(12)
        expires_at = now + config.DEFAULT_CLAIM_TTL_SEC * 1000
        conn.execute(
            """UPDATE slots
               SET state='claimed', claim_id=?, claimed_by=?, claimed_at=?, expires_at=?, host_tag=?, request_id=?
               WHERE job=? AND slot_id=? AND state='open'""",
            (claim_id, user["login"], now, expires_at, host_tag or None, req_id, slot["job"], slot["slot_id"]),
        )
        updated = conn.execute("SELECT * FROM slots WHERE job=? AND slot_id=?", (slot["job"], slot["slot_id"])).fetchone()
        db.log_audit(
            conn,
            user["login"],
            "claim_created",
            "ok",
            slot["job"],
            slot["slot_id"],
            {"claim_id": claim_id, "runner": runner_commit, "host_tag": host_tag},
        )
        return JSONResponse(status_code=201, content=_format_claim_response(conn, updated, job_row))


@app.delete("/api/claims/{job}/{slot_id}")
def release_claim(request: Request, job: str, slot_id: str) -> dict[str, Any]:
    _require_x_quant(request)
    _validate_id(job, "job")
    _validate_id(slot_id, "slot_id")
    with db.tx(immediate=True) as conn:
        user = _require_auth(conn, request)
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
        db.log_audit(conn, user["login"], "claim_released", "ok", job, slot_id, {})
        return {"released": True, "job": job, "slot_id": slot_id}


# ---------------------------------------------------------------------------
# Authenticated Artifact Download (inputs, verified labels, base checkpoints)
# ---------------------------------------------------------------------------

@app.get("/api/artifacts/{artifact_id}")
def download_artifact(request: Request, artifact_id: str) -> FileResponse:
    _require_x_quant(request)
    _validate_id(artifact_id, "artifact_id")
    with db.tx() as conn:
        _require_auth(conn, request)
        art = conn.execute("SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
        if art is None:
            raise HTTPException(status_code=404, detail="artifact not found")
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
# Upload Processing & Verification
# ---------------------------------------------------------------------------

def _finalize_upload(
    conn: sqlite3.Connection,
    user: sqlite3.Row,
    slot: sqlite3.Row,
    job_row: sqlite3.Row,
    staged_file: Path,
    filename: str,
    expected_sha256: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    actual_bytes = staged_file.stat().st_size
    max_bytes = config.MAX_UPLOAD_BYTES_BY_KIND.get(job_row["job_kind"], config.MAX_DIRECT_UPLOAD_BYTES)
    if actual_bytes <= 0 or actual_bytes > max_bytes:
        staged_file.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=f"upload size {actual_bytes} out of bounds (max {max_bytes})")

    actual_sha = verify.sha256_file(staged_file)
    if expected_sha256 and actual_sha != expected_sha256.lower():
        staged_file.unlink(missing_ok=True)
        db.log_audit(
            conn,
            user["login"],
            "upload_rejected",
            "failed_sha256",
            slot["job"],
            slot["slot_id"],
            {"expected": expected_sha256[:16], "actual": actual_sha[:16]},
        )
        raise HTTPException(status_code=422, detail=f"SHA-256 mismatch: expected {expected_sha256}, computed {actual_sha}")

    slot_plan = json.loads(slot["plan_json"] or "{}")
    inputs = json.loads(slot["inputs_json"] or "[]")
    kind = job_row["job_kind"]

    if kind == "drafter-label":
        prompt_path = None
        for inp in inputs:
            if inp.get("name") in ("prompt_shard", "prompts") and inp.get("artifact_id"):
                art_row = conn.execute("SELECT rel_path FROM artifacts WHERE artifact_id=?", (inp["artifact_id"],)).fetchone()
                if art_row:
                    prompt_path = _safe_under(config.DATA_DIR, config.DATA_DIR / art_row["rel_path"])
        vres = verify.verify_drafter_label(staged_file, slot_plan, prompt_path, manifest)
    elif kind == "drafter-eval":
        peers = conn.execute(
            """SELECT metrics_json FROM slots
               WHERE job=? AND state='verified' AND slot_id != ?
                 AND profile_id=? AND row_key=?""",
            (slot["job"], slot["slot_id"], slot["profile_id"], slot["row_key"]),
        ).fetchall()
        peer_metrics = [json.loads(p["metrics_json"] or "{}") for p in peers]
        vres = verify.verify_drafter_eval(staged_file, slot_plan, peer_metrics)
    elif kind == "drafter-capture":
        vres = verify.verify_drafter_capture(staged_file, slot_plan, manifest)
    elif kind == "drafter-train":
        vres = verify.verify_drafter_train_phase_a(staged_file, slot_plan, manifest)
    elif kind == "drafter-pack":
        peers = conn.execute(
            """SELECT metrics_json FROM slots
               WHERE job=? AND state='verified' AND slot_id != ?
                 AND row_key=? AND col_key=?""",
            (slot["job"], slot["slot_id"], slot["row_key"], slot["col_key"]),
        ).fetchall()
        peer_metrics = [json.loads(p["metrics_json"] or "{}") for p in peers]
        vres = verify.verify_drafter_pack(staged_file, slot_plan, peer_metrics)
    else:
        staged_file.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"unsupported job_kind '{kind}'")

    now = db.utc_ms()
    art_id = "art_" + secrets.token_hex(12)
    safe_fn = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)[:96] or "artifact.bin"

    if not vres.accepted:
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
                    "verify_status": vres.verify_status,
                    "reason": vres.reason,
                    "metrics": vres.metrics,
                    "utc": db.utc_iso(now),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        # Reopen the slot for honest workers and add a strike to the contributor
        conn.execute(
            """UPDATE slots
               SET state='open', claim_id=NULL, claimed_by=NULL, claimed_at=NULL, expires_at=NULL,
                   request_id=NULL, verify_status=?, verify_reason=?
               WHERE job=? AND slot_id=?""",
            (vres.verify_status, vres.reason, slot["job"], slot["slot_id"]),
        )
        conn.execute("UPDATE contributors SET strikes = strikes + 1 WHERE login=?", (user["login"],))
        db.log_audit(
            conn,
            user["login"],
            "upload_rejected",
            vres.verify_status,
            slot["job"],
            slot["slot_id"],
            {"reason": vres.reason, "sha256": actual_sha[:16], "bytes": actual_bytes, "quarantine_id": art_id},
        )
        return {
            "accepted": False,
            "state": "rejected_and_reopened",
            "verify_status": vres.verify_status,
            "reason": vres.reason,
            "quarantine_id": art_id,
            "sha256": actual_sha,
            "bytes": actual_bytes,
        }

    # Accepted (either immediately verified or queued for GPU Held-Out Replay Gate)
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

    verified_at = now if vres.final_state == "verified" else None
    conn.execute(
        """UPDATE slots
           SET state=?, upload_artifact_id=?, uploaded_by=?, uploaded_at=?, upload_bytes=?, upload_sha256=?,
               metrics_json=?, verify_status=?, verify_reason=?, verified_at=?,
               claim_id=NULL, expires_at=NULL
           WHERE job=? AND slot_id=?""",
        (
            vres.final_state,
            art_id,
            user["login"],
            now,
            actual_bytes,
            actual_sha,
            json.dumps(vres.metrics),
            vres.verify_status,
            vres.reason,
            verified_at,
            slot["job"],
            slot["slot_id"],
        ),
    )
    opened = []
    if vres.final_state == "verified":
        opened = db.trigger_downstream_readiness(conn, slot["job"], slot["slot_id"], art_id)

    db.log_audit(
        conn,
        user["login"],
        "upload_accepted" if vres.final_state == "verified" else "upload_queued_for_gate",
        vres.verify_status,
        slot["job"],
        slot["slot_id"],
        {
            "artifact_id": art_id,
            "sha256": actual_sha[:16],
            "bytes": actual_bytes,
            "opened_downstream": opened,
            "reason": vres.reason,
        },
    )
    return {
        "accepted": True,
        "state": vres.final_state,
        "artifact_id": art_id,
        "sha256": actual_sha,
        "bytes": actual_bytes,
        "verify_status": vres.verify_status,
        "reason": vres.reason,
        "metrics": vres.metrics,
        "opened_downstream": opened,
    }


@app.post("/api/uploads/direct/{claim_id}")
async def direct_upload(request: Request, claim_id: str) -> JSONResponse:
    """Single-request upload for artifacts <= 12 MiB (e.g., eval JSON, label gz JSONL)."""
    _require_x_quant(request)
    with db.tx() as conn:
        _require_auth(conn, request)
    _validate_id(claim_id, "claim_id")

    raw = await request.body()
    if len(raw) > config.MAX_DIRECT_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="payload exceeds direct upload limit; use chunked upload")
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="empty upload body")

    expected_sha = (request.headers.get("x-content-sha256") or "").strip().lower()
    if expected_sha and not HEX64_RE.match(expected_sha):
        raise HTTPException(status_code=400, detail="X-Content-SHA256 must be 64-char hex")

    filename = (request.query_params.get("filename") or "upload.bin").strip()
    manifest_raw = request.headers.get("x-manifest-json", "{}")
    try:
        manifest = json.loads(manifest_raw) if manifest_raw else {}
    except Exception:
        manifest = {}

    with db.tx(immediate=True) as conn:
        db.sweep_expired_claims(conn)
        user = _require_auth(conn, request)
        slot = conn.execute("SELECT * FROM slots WHERE claim_id=? AND state='claimed'", (claim_id,)).fetchone()
        if slot is None:
            raise HTTPException(status_code=404, detail="active claim not found or expired")
        if slot["claimed_by"] != user["login"]:
            raise HTTPException(status_code=403, detail="claim belongs to another contributor")
        job_row = conn.execute("SELECT * FROM jobs WHERE job=?", (slot["job"],)).fetchone()

        staged = config.STAGING_DIR / f"direct_{secrets.token_hex(12)}.tmp"
        staged.write_bytes(raw)
        result = _finalize_upload(conn, user, slot, job_row, staged, filename, expected_sha, manifest)
        status = 200 if result["accepted"] else 422
        return JSONResponse(status_code=status, content=result)


@app.post("/api/uploads/init", status_code=201)
async def init_chunked_upload(request: Request) -> JSONResponse:
    _require_x_quant(request)
    with db.tx() as conn:
        _require_auth(conn, request)
    body = await request.json()
    claim_id = _validate_id(str(body.get("claim_id") or ""), "claim_id")
    filename = str(body.get("filename") or "artifact.safetensors").strip()
    total_bytes = int(body.get("total_bytes") or 0)
    total_sha256 = str(body.get("total_sha256") or "").strip().lower()
    chunk_bytes = int(body.get("chunk_bytes") or config.MAX_CHUNK_BYTES)
    manifest = body.get("manifest") or {}

    if not HEX64_RE.match(total_sha256):
        raise HTTPException(status_code=400, detail="total_sha256 must be 64-char lowercase hex")
    if chunk_bytes <= 0 or chunk_bytes > config.MAX_CHUNK_BYTES:
        raise HTTPException(status_code=400, detail=f"chunk_bytes must be in 1..{config.MAX_CHUNK_BYTES}")

    with db.tx(immediate=True) as conn:
        db.sweep_expired_claims(conn)
        user = _require_auth(conn, request)
        slot = conn.execute("SELECT * FROM slots WHERE claim_id=? AND state='claimed'", (claim_id,)).fetchone()
        if slot is None:
            raise HTTPException(status_code=404, detail="active claim not found")
        if slot["claimed_by"] != user["login"]:
            raise HTTPException(status_code=403, detail="claim belongs to another contributor")
        job_row = conn.execute("SELECT * FROM jobs WHERE job=?", (slot["job"],)).fetchone()
        max_bytes = config.MAX_UPLOAD_BYTES_BY_KIND.get(job_row["job_kind"], config.MAX_DIRECT_UPLOAD_BYTES)
        if total_bytes <= 0 or total_bytes > max_bytes:
            raise HTTPException(status_code=413, detail=f"total_bytes {total_bytes} exceeds cap {max_bytes}")

        total_chunks = (total_bytes + chunk_bytes - 1) // chunk_bytes
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
        return JSONResponse(
            status_code=201,
            content={
                "upload_id": upload_id,
                "total_chunks": total_chunks,
                "chunk_bytes": chunk_bytes,
                "chunk_url_template": f"{config.PUBLIC_URL}/api/uploads/{upload_id}/chunk/{{index}}",
                "complete_url": f"{config.PUBLIC_URL}/api/uploads/{upload_id}/complete",
            },
        )


@app.put("/api/uploads/{upload_id}/chunk/{index}")
async def put_upload_chunk(request: Request, upload_id: str, index: int) -> dict[str, Any]:
    _require_x_quant(request)
    with db.tx() as conn:
        _require_auth(conn, request)
    _validate_id(upload_id, "upload_id")
    raw = await request.body()
    if len(raw) == 0 or len(raw) > config.MAX_CHUNK_BYTES:
        raise HTTPException(status_code=413, detail=f"chunk size {len(raw)} out of bounds (max {config.MAX_CHUNK_BYTES})")

    chunk_sha = hashlib.sha256(raw).hexdigest()
    hdr_sha = (request.headers.get("x-chunk-sha256") or "").strip().lower()
    if hdr_sha and hdr_sha != chunk_sha:
        raise HTTPException(status_code=422, detail="X-Chunk-SHA256 mismatch")

    with db.tx(immediate=True) as conn:
        user = _require_auth(conn, request)
        sess = conn.execute("SELECT * FROM upload_sessions WHERE upload_id=? AND state='open'", (upload_id,)).fetchone()
        if sess is None:
            raise HTTPException(status_code=404, detail="open upload session not found")
        if sess["contributor_login"] != user["login"]:
            raise HTTPException(status_code=403, detail="upload session belongs to another contributor")
        if index < 0 or index >= int(sess["total_chunks"]):
            raise HTTPException(status_code=400, detail=f"chunk index {index} out of range 0..{sess['total_chunks']-1}")

        part_path = _safe_under(config.STAGING_DIR, config.STAGING_DIR / upload_id / f"part_{index:06d}")
        part_path.write_bytes(raw)
        rec = json.loads(sess["received_chunks_json"] or "{}")
        rec[str(index)] = {"bytes": len(raw), "sha256": chunk_sha}
        conn.execute("UPDATE upload_sessions SET received_chunks_json=? WHERE upload_id=?", (json.dumps(rec), upload_id))
        return {"upload_id": upload_id, "index": index, "bytes": len(raw), "sha256": chunk_sha, "received": len(rec), "total_chunks": sess["total_chunks"]}


@app.post("/api/uploads/{upload_id}/complete")
def complete_chunked_upload(request: Request, upload_id: str) -> JSONResponse:
    _require_x_quant(request)
    _validate_id(upload_id, "upload_id")
    with db.tx(immediate=True) as conn:
        db.sweep_expired_claims(conn)
        user = _require_auth(conn, request)
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

        sdir = _safe_under(config.STAGING_DIR, config.STAGING_DIR / upload_id)
        assembled = _safe_under(config.STAGING_DIR, config.STAGING_DIR / f"{upload_id}_assembled.tmp")
        written = 0
        with assembled.open("wb") as out_f:
            for i in range(total_chunks):
                p = _safe_under(sdir, sdir / f"part_{i:06d}")
                with p.open("rb") as pf:
                    while True:
                        buf = pf.read(1024 * 1024)
                        if not buf:
                            break
                        out_f.write(buf)
                        written += len(buf)

        shutil.rmtree(sdir, ignore_errors=True)
        if written != int(sess["total_bytes"]):
            assembled.unlink(missing_ok=True)
            conn.execute("UPDATE upload_sessions SET state='failed' WHERE upload_id=?", (upload_id,))
            raise HTTPException(status_code=422, detail=f"assembled size {written} != declared {sess['total_bytes']}")

        manifest = json.loads(sess["manifest_json"] or "{}")
        result = _finalize_upload(conn, user, slot, job_row, assembled, sess["filename"], sess["total_sha256"], manifest)
        conn.execute("UPDATE upload_sessions SET state=? WHERE upload_id=?", ("done" if result["accepted"] else "rejected", upload_id))
        status = 200 if result["accepted"] else 422
        return JSONResponse(status_code=status, content=result)


# ---------------------------------------------------------------------------
# Admin API (X-Quant: 1 + separate Admin Bearer credential ONLY)
# ---------------------------------------------------------------------------

@app.post("/api/admin/runners")
async def admin_add_runner(request: Request) -> dict[str, Any]:
    _require_x_quant(request)
    with db.tx() as conn:
        _require_auth(conn, request, admin=True)
    body = await request.json()
    commit_id = _validate_id(str(body.get("commit_id") or ""), "commit_id")
    kind = str(body.get("kind") or "all")
    image_digest = body.get("image_digest")
    note = body.get("note")
    with db.tx(immediate=True) as conn:
        admin = _require_auth(conn, request, admin=True)
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
    _require_x_quant(request)
    with db.tx() as conn:
        _require_auth(conn, request, admin=True)
    body = await request.json()
    login = _validate_id(str(body.get("login") or ""), "login")
    approved = 1 if body.get("approved", True) else 0
    quarantined = 1 if body.get("quarantined", False) else 0
    tiers = json.dumps(body.get("tiers") or ["tier-eval"])
    note = body.get("note")
    raw_key = None
    with db.tx(immediate=True) as conn:
        admin = _require_auth(conn, request, admin=True)
        existing = conn.execute("SELECT login FROM contributors WHERE login=?", (login,)).fetchone()
        if existing is None or body.get("rotate_key"):
            raw_key = "drs_wrk_" + secrets.token_urlsafe(32)
            kh = db.hash_api_key(raw_key)
            conn.execute(
                """INSERT INTO contributors (login, key_hash, is_admin, approved, tiers_json, quarantined, note, created_at)
                   VALUES (?, ?, 0, ?, ?, ?, ?, ?)
                   ON CONFLICT(login) DO UPDATE SET key_hash=excluded.key_hash, approved=excluded.approved,
                     tiers_json=excluded.tiers_json, quarantined=excluded.quarantined, note=excluded.note""",
                (login, kh, approved, tiers, quarantined, note, db.utc_ms()),
            )
        else:
            conn.execute(
                "UPDATE contributors SET approved=?, tiers_json=?, quarantined=?, note=? WHERE login=?",
                (approved, tiers, quarantined, note, login),
            )
        db.log_audit(conn, admin["login"], "contributor_updated", "ok", detail={"login": login, "approved": bool(approved), "quarantined": bool(quarantined)})
        resp: dict[str, Any] = {"ok": True, "login": login, "approved": bool(approved), "quarantined": bool(quarantined)}
        if raw_key:
            resp["api_key_shown_once"] = raw_key
        return resp


@app.post("/api/admin/artifacts/register")
async def admin_register_local_artifact(request: Request) -> dict[str, Any]:
    """Register an immutable input file already staged under /data/inputs/."""
    _require_x_quant(request)
    with db.tx() as conn:
        _require_auth(conn, request, admin=True)
    body = await request.json()
    rel_path = str(body.get("rel_path") or "").lstrip("/")
    artifact_id = _validate_id(str(body.get("artifact_id") or ("inp_" + secrets.token_hex(8))), "artifact_id")
    kind = str(body.get("kind") or "input")
    content_type = str(body.get("content_type") or "application/octet-stream")
    fpath = _safe_under(config.INPUTS_DIR, config.DATA_DIR / rel_path)
    if not fpath.is_file():
        raise HTTPException(status_code=404, detail=f"file not found under inputs: {rel_path}")
    sha = verify.sha256_file(fpath)
    nbytes = fpath.stat().st_size
    with db.tx(immediate=True) as conn:
        admin = _require_auth(conn, request, admin=True)
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
    with db.tx() as conn:
        _require_auth(conn, request, admin=True)
    body = await request.json()
    job = _validate_id(str(body.get("job") or ""), "job")
    title = str(body.get("title") or job)
    job_kind = str(body.get("job_kind") or "")
    if job_kind not in ("drafter-label", "drafter-train", "drafter-eval", "drafter-capture", "drafter-pack"):
        raise HTTPException(status_code=400, detail="invalid job_kind")
    model = str(body.get("model") or "GLM-5.3")
    release_repo = str(body.get("release_repo") or "kindlingai/glm-5.3-exlr8-k2-k3-k4")
    release_revision = str(body.get("release_revision") or "9e25c28a2e8808d8b2599d780521fd6af24e7558")
    active_round = int(body.get("active_round") or 1)
    cfg = json.dumps(body.get("config") or {})
    slots = body.get("slots") or []
    rounds = body.get("rounds") or []

    with db.tx(immediate=True) as conn:
        admin = _require_auth(conn, request, admin=True)
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
            conn.execute(
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
            inserted += 1
        db.log_audit(conn, admin["login"], "job_upserted", "ok", job, detail={"job_kind": job_kind, "slots_defined": inserted})
        return {"ok": True, "job": job, "slots_processed": inserted}


@app.post("/api/admin/slots/{job}/{slot_id}/ready")
async def admin_mark_slot_ready(request: Request, job: str, slot_id: str) -> dict[str, Any]:
    _require_x_quant(request)
    _validate_id(job, "job")
    _validate_id(slot_id, "slot_id")
    body = await request.json()
    with db.tx(immediate=True) as conn:
        admin = _require_auth(conn, request, admin=True)
        slot = conn.execute("SELECT * FROM slots WHERE job=? AND slot_id=?", (job, slot_id)).fetchone()
        if slot is None:
            raise HTTPException(status_code=404, detail="slot not found")
        inputs = body.get("inputs")
        plan = body.get("plan")
        state = str(body.get("state") or "open")
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
    """Record Owner GPU Verification Gate verdict (Held-Out Replay + FLTrust Cosine + L2 Norm)."""
    _require_x_quant(request)
    _validate_id(job, "job")
    _validate_id(slot_id, "slot_id")
    body = await request.json()
    passed = bool(body.get("passed"))
    reason = str(body.get("reason") or ("passed GPU replay + FLTrust gate" if passed else "failed GPU gate"))
    gate_metrics = body.get("gate_metrics") or {}
    reopen_on_reject = bool(body.get("reopen_on_reject", True))

    with db.tx(immediate=True) as conn:
        admin = _require_auth(conn, request, admin=True)
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
                   SET state='verified', verify_status='verified_replay_fltrust_gate', verify_reason=?,
                       verified_at=?, metrics_json=?
                   WHERE job=? AND slot_id=?""",
                (reason, now, json.dumps(merged_metrics), job, slot_id),
            )
            opened = []
            if slot["upload_artifact_id"]:
                opened = db.trigger_downstream_readiness(conn, job, slot_id, slot["upload_artifact_id"])
            db.log_audit(
                conn,
                actor,
                "verification_passed",
                "verified_replay_fltrust_gate",
                job,
                slot_id,
                {"reason": reason, "gate_metrics": gate_metrics, "opened_downstream": opened},
            )
            return {"ok": True, "state": "verified", "opened_downstream": opened}

        # Failed Owner GPU Gate -> move artifact to quarantine, add strike, reopen slot if requested
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
                       upload_bytes=0, upload_sha256=NULL, verify_status='rejected_replay_fltrust_gate',
                       verify_reason=?, metrics_json=?, claim_id=NULL, claimed_by=NULL, claimed_at=NULL, expires_at=NULL, host_tag=NULL, request_id=NULL
                   WHERE job=? AND slot_id=?""",
                (reason, json.dumps(merged_metrics), job, slot_id),
            )
        else:
            conn.execute(
                """UPDATE slots
                   SET state='rejected', verify_status='rejected_replay_fltrust_gate', verify_reason=?, metrics_json=?
                   WHERE job=? AND slot_id=?""",
                (reason, json.dumps(merged_metrics), job, slot_id),
            )
        if actor != "unknown":
            conn.execute("UPDATE contributors SET strikes = strikes + 1 WHERE login=?", (actor,))
        db.log_audit(
            conn,
            actor,
            "verification_rejected",
            "rejected_replay_fltrust_gate",
            job,
            slot_id,
            {"reason": reason, "gate_metrics": gate_metrics, "quarantined_artifact": art_id, "next_state": next_state},
        )
        return {"ok": True, "state": next_state, "quarantined_artifact": art_id}


@app.post("/api/admin/rounds/{job}/{round_num}/aggregate")
async def admin_record_round_aggregate(request: Request, job: str, round_num: int) -> dict[str, Any]:
    _require_x_quant(request)
    _validate_id(job, "job")
    body = await request.json()
    with db.tx(immediate=True) as conn:
        admin = _require_auth(conn, request, admin=True)
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
            {"agg_method": body.get("agg_method"), "agg_ckpt_name": body.get("agg_ckpt_name"), "agg_metrics": body.get("agg_metrics")},
        )
        return {"ok": True, "job": job, "round_num": round_num}


@app.get("/api/admin/host-map")
def admin_host_map(request: Request) -> list[dict[str, Any]]:
    """Private admin-only mapping of neutral contributor logins to internal host tags and notes."""
    _require_x_quant(request)
    with db.tx() as conn:
        _require_auth(conn, request, admin=True)
        contribs = conn.execute(
            "SELECT login, approved, quarantined, tiers_json, strikes, note, last_seen_at FROM contributors WHERE is_admin=0 ORDER BY login ASC"
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
    """List uploaded slots awaiting Owner GPU Held-Out Replay + FLTrust verification."""
    _require_x_quant(request)
    with db.tx() as conn:
        _require_auth(conn, request, admin=True)
        rows = conn.execute(
            """SELECT s.job, j.job_kind, s.slot_id, s.round_num, s.profile_id, s.uploaded_by, s.uploaded_at,
                      s.upload_artifact_id, s.upload_bytes, s.upload_sha256, s.metrics_json, s.plan_json, a.rel_path
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
                "metrics": json.loads(r["metrics_json"] or "{}"),
                "plan": json.loads(r["plan_json"] or "{}"),
            }
            for r in rows
        ]
