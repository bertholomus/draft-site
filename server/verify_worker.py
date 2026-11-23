"""Isolated subprocess worker for Phase-A artifact assembly, hashing, and verification (S6).

Runs outside the uvicorn web server process and outside any SQLite transaction, with an optional
address-space memory limit (RLIMIT_AS) so malicious or oversized payloads cannot OOM-kill or
stall the HTTP coordinator.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from server import config, verify


def _apply_mem_limit(mem_bytes: int) -> None:
    if mem_bytes <= 0:
        return
    try:
        import resource

        if hasattr(resource, "RLIMIT_AS"):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            target = mem_bytes if hard in (-1, resource.RLIM_INFINITY) else min(mem_bytes, hard)
            if soft in (-1, resource.RLIM_INFINITY) or target < soft:
                resource.setrlimit(resource.RLIMIT_AS, (target, hard))
    except Exception:
        # Best-effort on platforms where RLIMIT_AS is unsupported or constrained
        pass


def _assemble_chunks(
    staging_dir: Path,
    assembled_path: Path,
    total_chunks: int,
    declared_bytes: int,
) -> tuple[int, str]:
    written = 0
    h = hashlib.sha256()
    with assembled_path.open("wb") as out_f:
        for i in range(total_chunks):
            part = (staging_dir / f"part_{i:06d}").resolve()
            if not part.is_relative_to(staging_dir.resolve()) or not part.is_file():
                raise ValueError(f"missing staged chunk part_{i:06d}")
            with part.open("rb") as pf:
                while True:
                    buf = pf.read(1024 * 1024)
                    if not buf:
                        break
                    written += len(buf)
                    if written > declared_bytes:
                        raise ValueError(f"assembled size exceeds declared {declared_bytes}")
                    h.update(buf)
                    out_f.write(buf)
    if written != declared_bytes:
        raise ValueError(f"assembled size {written} != declared {declared_bytes}")
    return written, h.hexdigest()


def run_job(req: dict[str, Any]) -> dict[str, Any]:
    mode = str(req["mode"])
    kind = str(req["kind"])
    max_bytes = int(req["max_bytes"])
    expected_sha = str(req.get("expected_sha256") or "").strip().lower()
    slot_plan = req.get("slot_plan") or {}
    manifest = req.get("manifest") or {}
    peer_metrics = req.get("peer_metrics") or []
    prompt_path = Path(req["prompt_path"]) if req.get("prompt_path") else None
    require_owner_gate = bool(req.get("require_owner_gate", True))

    if mode == "chunked":
        sdir = Path(req["staging_dir"])
        staged_file = Path(req["assembled_path"])
        try:
            actual_bytes, actual_sha = _assemble_chunks(
                sdir,
                staged_file,
                int(req["total_chunks"]),
                int(req["declared_bytes"]),
            )
        except Exception as e:
            return {"ok": False, "error_status": 422, "error_code": "failed_assembly", "error_detail": str(e)}
    elif mode == "direct":
        staged_file = Path(req["staged_path"])
        if not staged_file.is_file():
            return {"ok": False, "error_status": 400, "error_code": "failed_missing", "error_detail": "staged file missing"}
        actual_bytes = staged_file.stat().st_size
        if actual_bytes <= 0 or actual_bytes > max_bytes:
            return {
                "ok": False,
                "error_status": 413,
                "error_code": "failed_size",
                "error_detail": f"upload size {actual_bytes} out of bounds (max {max_bytes})",
            }
        actual_sha = verify.sha256_file(staged_file)
    else:
        return {"ok": False, "error_status": 400, "error_code": "failed_mode", "error_detail": f"invalid mode {mode}"}

    if actual_bytes <= 0 or actual_bytes > max_bytes:
        return {
            "ok": False,
            "error_status": 413,
            "error_code": "failed_size",
            "error_detail": f"upload size {actual_bytes} out of bounds (max {max_bytes})",
            "actual_bytes": actual_bytes,
            "actual_sha256": actual_sha,
        }

    if expected_sha and actual_sha != expected_sha:
        return {
            "ok": False,
            "error_status": 422,
            "error_code": "failed_sha256",
            "error_detail": f"SHA-256 mismatch: expected {expected_sha}, computed {actual_sha}",
            "actual_bytes": actual_bytes,
            "actual_sha256": actual_sha,
        }

    if kind == "drafter-label":
        vres = verify.verify_drafter_label(
            staged_file,
            slot_plan,
            prompt_path,
            manifest,
            require_owner_gate=require_owner_gate,
        )
    elif kind == "drafter-eval":
        vres = verify.verify_drafter_eval(staged_file, slot_plan, peer_metrics)
    elif kind == "drafter-capture":
        vres = verify.verify_drafter_capture(staged_file, slot_plan, manifest)
    elif kind == "drafter-train":
        vres = verify.verify_drafter_train_phase_a(staged_file, slot_plan, manifest)
    elif kind == "drafter-pack":
        vres = verify.verify_drafter_pack(staged_file, slot_plan, peer_metrics)
    else:
        return {"ok": False, "error_status": 400, "error_code": "failed_kind", "error_detail": f"unsupported job_kind '{kind}'"}

    return {
        "ok": True,
        "actual_bytes": actual_bytes,
        "actual_sha256": actual_sha,
        "verify_result": {
            "accepted": vres.accepted,
            "final_state": vres.final_state,
            "verify_status": vres.verify_status,
            "reason": vres.reason,
            "metrics": vres.metrics,
        },
    }


def main() -> None:
    _apply_mem_limit(int(os.environ.get("DRAFT_VERIFY_MEM_BYTES", str(config.VERIFY_SUBPROCESS_MEM_LIMIT_BYTES))))
    raw = sys.stdin.read()
    req = json.loads(raw)
    res = run_job(req)
    sys.stdout.write(json.dumps(res, separators=(",", ":")))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
