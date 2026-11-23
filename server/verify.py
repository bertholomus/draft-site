"""Zero-pickle, non-executing server-side Phase-A verification gates for uploaded artifacts.

- Uploaded files are NEVER executed and NEVER loaded with pickle / torch.load.
- Label uploads (Phase A): fail-closed prompt shard check, gzip zip-bomb guard, strict UTF-8 & JSONL schema,
  prompt SHA-256 match, strict int vocab range, gen_sha256 check, length & short-row structural guards,
  provenance extraction. Never trusts client-reported semantic agreement; non-owner uploads wait at
  state='uploaded' (verify_status='pending_label_gate') until the owner-side teacher-forced gate passes them.
- Eval/capture/train/pack uploads (owner-only): safe JSON/safetensors checks with bounded memory usage
  and strict type validation (zero uncaught ValueError/UnicodeDecodeError).
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from server import config

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_PROVENANCE_STR = 256
MAX_EVAL_BY_CAT_BYTES = 64 * 1024
NORM_CHUNK_ELEMS = 1_000_000


@dataclass
class VerifyResult:
    accepted: bool
    final_state: str  # "verified", "uploaded" (pending owner gate), or "rejected"
    verify_status: str
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _safe_float(val: Any) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _safe_int(val: Any) -> int | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _safe_gzip_read(path: Path, max_bytes: int) -> bytes:
    with gzip.open(path, "rb") as gz:
        out = io.BytesIO()
        total = 0
        while True:
            chunk = gz.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"decompressed payload exceeds {max_bytes} byte cap")
            out.write(chunk)
        return out.getvalue()


def _chunked_sq_norm_int8(q: np.ndarray, scale: float) -> float:
    """Compute sum((q * scale)^2) without materializing a full float32 copy of huge tensors."""
    flat = q.reshape(-1)
    s2 = float(scale) * float(scale)
    acc = 0.0
    for i in range(0, flat.size, NORM_CHUNK_ELEMS):
        sl = flat[i : i + NORM_CHUNK_ELEMS].astype(np.int64)
        acc += float(np.dot(sl, sl)) * s2
    return acc


def _chunked_sq_norm_float(arr: np.ndarray) -> tuple[float, bool]:
    """Compute sum(arr^2) and finiteness check without full float32 blowup."""
    flat = arr.reshape(-1)
    acc = 0.0
    for i in range(0, flat.size, NORM_CHUNK_ELEMS):
        sl = flat[i : i + NORM_CHUNK_ELEMS].astype(np.float32)
        if not np.all(np.isfinite(sl)):
            return 0.0, False
        acc += float(np.dot(sl, sl))
    return acc, True


def verify_drafter_label(
    artifact_path: Path,
    slot_plan: dict[str, Any],
    input_prompt_path: Path | None,
    manifest: dict[str, Any],
    *,
    require_owner_gate: bool = True,
) -> VerifyResult:
    """Phase-A structural & cryptographic verification of a drafter-label upload.

    Never trusts client-reported semantic agreement. When require_owner_gate is True
    (all non-trusted / external logins), accepted uploads land in state='uploaded'
    with verify_status='pending_label_gate' until owner-side teacher-forced spot-rescoring passes.
    """
    if input_prompt_path is None or not input_prompt_path.is_file():
        return VerifyResult(False, "rejected", "failed_no_prompt_shard", "slot missing registered prompt shard")

    try:
        with artifact_path.open("rb") as f:
            raw_head = f.read(2)
        if raw_head == b"\x1f\x8b":
            raw = _safe_gzip_read(artifact_path, config.MAX_DECOMPRESSED_LABEL_BYTES)
        else:
            if artifact_path.stat().st_size > config.MAX_DECOMPRESSED_LABEL_BYTES:
                return VerifyResult(False, "rejected", "failed_size", "uncompressed label file too large")
            raw = artifact_path.read_bytes()
    except Exception as e:
        return VerifyResult(False, "rejected", "failed_decompress", f"gzip decode failed: {e}")

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as e:
        return VerifyResult(False, "rejected", "failed_utf8", f"label payload is not valid UTF-8: {e}")

    expected_prompts: dict[str, str] = {}
    expected_max_tokens: dict[str, int] = {}
    default_short = _safe_int(slot_plan.get("max_tokens_short")) or _safe_int(slot_plan.get("max_tokens")) or 1536
    default_long = _safe_int(slot_plan.get("max_tokens_long")) or _safe_int(slot_plan.get("max_tokens")) or 1024
    try:
        for line in input_prompt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = str(row["id"])
            psha = row.get("prompt_sha256")
            if not psha:
                if "prompt_ids" in row:
                    psha = hashlib.sha256(json.dumps(row["prompt_ids"], separators=(",", ":")).encode("utf-8")).hexdigest()
                else:
                    psha = hashlib.sha256(str(row.get("prompt", "")).encode("utf-8")).hexdigest()
            expected_prompts[pid] = str(psha)
            rmax = _safe_int(row.get("max_tokens"))
            if rmax is None or rmax <= 0:
                plen = len(row.get("prompt_ids") or [])
                rmax = default_long if plen > 4096 else default_short
            expected_max_tokens[pid] = max(16, rmax)
    except Exception as e:
        return VerifyResult(False, "rejected", "failed_prompt_shard", f"failed to read prompt shard: {e}")

    if not expected_prompts:
        return VerifyResult(False, "rejected", "failed_no_prompt_shard", "registered prompt shard is empty")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return VerifyResult(False, "rejected", "failed_empty", "label JSONL has 0 rows")

    seen: set[str] = set()
    total_gen_tokens = 0
    short_rows = 0
    engines: set[str] = set()
    image_digests: set[str] = set()
    engine_sigs: set[str] = set()
    quant_profiles: set[str] = set()
    concurrencies: set[int] = set()
    token_sources: set[str] = set()
    finish_reasons: dict[str, int] = {}

    require_prov = bool(slot_plan.get("require_provenance", False))
    allowed_profiles = set(slot_plan.get("allowed_quant_profiles") or [])

    for idx, line in enumerate(lines):
        try:
            row = json.loads(line)
        except Exception as e:
            return VerifyResult(False, "rejected", "failed_json", f"row {idx} invalid JSON: {e}")
        if not isinstance(row, dict):
            return VerifyResult(False, "rejected", "failed_schema", f"row {idx} must be a JSON object")
        for req_k in ("id", "prompt_sha256", "gen_ids", "finish_reason", "engine_tag", "gen_sha256"):
            if req_k not in row:
                return VerifyResult(False, "rejected", "failed_schema", f"row {idx} missing key '{req_k}'")

        pid = str(row["id"])
        if pid in seen:
            return VerifyResult(False, "rejected", "failed_dup_id", f"duplicate prompt id '{pid}'")
        seen.add(pid)

        if pid not in expected_prompts:
            return VerifyResult(False, "rejected", "failed_prompt_id", f"unexpected prompt id '{pid}'")
        if str(row["prompt_sha256"]) != expected_prompts[pid]:
            return VerifyResult(False, "rejected", "failed_prompt_sha", f"prompt_sha256 mismatch on '{pid}'")

        gen_ids = row["gen_ids"]
        if not isinstance(gen_ids, list) or len(gen_ids) == 0:
            return VerifyResult(False, "rejected", "failed_gen_empty", f"empty gen_ids on '{pid}'")
        if any((type(t) is not int) or t < 0 or t >= config.VOCAB_SIZE for t in gen_ids):
            return VerifyResult(False, "rejected", "failed_vocab", f"out-of-range or non-int token id on '{pid}'")

        expect_gen_sha = hashlib.sha256(json.dumps(gen_ids, separators=(",", ":")).encode("utf-8")).hexdigest()
        if str(row["gen_sha256"]) != expect_gen_sha:
            return VerifyResult(False, "rejected", "failed_gen_sha", f"gen_sha256 mismatch on '{pid}'")

        # Phase-A structural guards (LABEL-VERIFY-FIX.md §2.2)
        fr = str(row["finish_reason"])
        if fr == "length" and len(gen_ids) < 16:
            return VerifyResult(
                False,
                "rejected",
                "failed_short_length_finish",
                f"row '{pid}' has finish_reason='length' with only {len(gen_ids)} tokens",
            )
        row_max_tok = expected_max_tokens[pid]
        if len(gen_ids) > row_max_tok + 8:
            return VerifyResult(
                False,
                "rejected",
                "failed_max_tokens_exceeded",
                f"row '{pid}' has {len(gen_ids)} tokens exceeding max_tokens={row_max_tok}+8",
            )
        if len(gen_ids) <= 2:
            short_rows += 1

        # Per-row provenance metadata (Fable 4.4)
        etag = str(row["engine_tag"]).strip()
        if not etag or len(etag) > MAX_PROVENANCE_STR:
            return VerifyResult(False, "rejected", "failed_provenance", f"invalid engine_tag on '{pid}'")
        engines.add(etag)

        if require_prov:
            for pk in ("engine_image_digest", "engine_signature", "quant_profile", "concurrency"):
                if pk not in row or row[pk] is None or str(row[pk]).strip() == "":
                    return VerifyResult(False, "rejected", "failed_provenance", f"row '{pid}' missing provenance '{pk}'")

        if "engine_image_digest" in row and row["engine_image_digest"] is not None:
            dig = str(row["engine_image_digest"]).strip()
            if len(dig) > MAX_PROVENANCE_STR:
                return VerifyResult(False, "rejected", "failed_provenance", "engine_image_digest too long")
            if dig:
                image_digests.add(dig)
        if "engine_signature" in row and row["engine_signature"] is not None:
            sig = str(row["engine_signature"]).strip()
            if len(sig) > MAX_PROVENANCE_STR:
                return VerifyResult(False, "rejected", "failed_provenance", "engine_signature too long")
            if sig:
                engine_sigs.add(sig)
        if "quant_profile" in row and row["quant_profile"] is not None:
            qp = str(row["quant_profile"]).strip()
            if len(qp) > 128:
                return VerifyResult(False, "rejected", "failed_provenance", "quant_profile too long")
            if allowed_profiles and qp not in allowed_profiles:
                return VerifyResult(False, "rejected", "failed_provenance_profile", f"quant_profile '{qp}' not allowlisted")
            if qp:
                quant_profiles.add(qp)
        if "concurrency" in row and row["concurrency"] is not None:
            conc = _safe_int(row["concurrency"])
            if conc is None or conc <= 0 or conc > 256:
                return VerifyResult(False, "rejected", "failed_provenance", f"invalid concurrency on '{pid}'")
            concurrencies.add(conc)
        if "token_id_source" in row and row["token_id_source"] is not None:
            tsrc = str(row["token_id_source"]).strip()
            if tsrc not in ("engine", "pinned_tokenizer"):
                return VerifyResult(False, "rejected", "failed_provenance", f"invalid token_id_source '{tsrc}'")
            token_sources.add(tsrc)

        total_gen_tokens += len(gen_ids)
        finish_reasons[fr] = finish_reasons.get(fr, 0) + 1

    if seen != set(expected_prompts.keys()):
        missing = sorted(set(expected_prompts.keys()) - seen)
        return VerifyResult(
            False,
            "rejected",
            "failed_incomplete",
            f"missing {len(missing)} prompts (first: {missing[:3]})",
        )

    if (short_rows / max(1, len(seen))) > 0.10:
        return VerifyResult(
            False,
            "rejected",
            "failed_short_rows",
            f"{short_rows}/{len(seen)} rows have <=2 tokens (exceeds 10% structural guard)",
            {"short_rows": short_rows, "n_rows": len(seen)},
        )

    metrics: dict[str, Any] = {
        "n_rows": len(seen),
        "total_gen_tokens": total_gen_tokens,
        "mean_gen_tokens": round(total_gen_tokens / max(1, len(seen)), 2),
        "short_rows": short_rows,
        "engines": sorted(engines)[:8],
        "finish_reasons": finish_reasons,
    }
    if image_digests:
        metrics["engine_image_digests"] = sorted(image_digests)[:4]
    if engine_sigs:
        metrics["engine_signatures"] = sorted(engine_sigs)[:4]
    if quant_profiles:
        metrics["quant_profiles"] = sorted(quant_profiles)[:4]
    if concurrencies:
        metrics["concurrencies"] = sorted(concurrencies)[:4]
    if token_sources:
        metrics["token_id_sources"] = sorted(token_sources)

    # Note: client-reported manifest.semantic_top1_agree / semantic_top5_agree are intentionally
    # ignored for gating; semantic verification is strictly owner-side (label_verify_gate.py).
    if require_owner_gate:
        return VerifyResult(
            True,
            "uploaded",
            "pending_label_gate",
            f"Phase A passed ({len(seen)} prompts, {total_gen_tokens} tokens); awaiting owner teacher-forced label gate",
            metrics,
        )
    return VerifyResult(
        True,
        "verified",
        "verified_label_trusted_owner",
        f"verified trusted owner shard ({len(seen)} prompts, {total_gen_tokens} gen tokens)",
        metrics,
    )


def verify_drafter_eval(
    artifact_path: Path,
    slot_plan: dict[str, Any],
    peer_verified_metrics: list[dict[str, Any]],
) -> VerifyResult:
    """Verify an owner drafter-eval JSON upload (schema, finite bounds, duplicate-slot quorum)."""
    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception as e:
        return VerifyResult(False, "rejected", "failed_json", f"invalid JSON: {e}")

    if not isinstance(data, dict):
        return VerifyResult(False, "rejected", "failed_schema", "top-level JSON must be an object")

    req_keys = (
        "checkpoint_rev",
        "profile_id",
        "selfcheck_max_err",
        "n_blocks",
        "b8_ref_tau",
        "b16_ref_tau",
        "ce8",
        "top1",
        "per_pos_ref",
        "by_cat",
        "pred_sha256",
    )
    missing = [k for k in req_keys if k not in data]
    if missing:
        return VerifyResult(False, "rejected", "failed_schema", f"missing required keys: {missing}")

    if slot_plan.get("profile_id") and data["profile_id"] != slot_plan["profile_id"]:
        return VerifyResult(False, "rejected", "failed_profile_mismatch", "profile_id does not match claim plan")
    if slot_plan.get("checkpoint_rev") and data["checkpoint_rev"] != slot_plan["checkpoint_rev"]:
        return VerifyResult(False, "rejected", "failed_ckpt_mismatch", "checkpoint_rev does not match claim plan")

    sc = _safe_float(data["selfcheck_max_err"])
    b8 = _safe_float(data["b8_ref_tau"])
    b16 = _safe_float(data["b16_ref_tau"])
    ce8 = _safe_float(data["ce8"])
    top1 = _safe_float(data["top1"])
    n_blocks = _safe_int(data["n_blocks"])
    if None in (sc, b8, b16, ce8, top1, n_blocks):
        return VerifyResult(False, "rejected", "failed_types", "numeric parse error or non-finite metric")

    assert sc is not None and b8 is not None and b16 is not None and ce8 is not None and top1 is not None and n_blocks is not None
    if sc < 0.0 or sc > config.TRAIN_MAX_SELFCHECK_ERR:
        return VerifyResult(False, "rejected", "failed_selfcheck", "selfcheck_max_err exceeds tolerance")
    if n_blocks <= 0 or not (1.0 <= b8 <= 9.0) or not (1.0 <= b16 <= 17.0) or not (0.0 <= top1 <= 1.0) or ce8 <= 0.0:
        return VerifyResult(False, "rejected", "failed_bounds", "metrics out of plausible physical bounds")

    pred_sha = str(data["pred_sha256"]).lower()
    if not HEX64_RE.match(pred_sha):
        return VerifyResult(False, "rejected", "failed_pred_sha", "pred_sha256 must be 64-char lowercase hex")

    if not isinstance(data["per_pos_ref"], list) or len(data["per_pos_ref"]) < 7:
        return VerifyResult(False, "rejected", "failed_per_pos", "per_pos_ref must contain >=7 positions")
    per_pos: list[float] = []
    for x in data["per_pos_ref"][:15]:
        fx = _safe_float(x)
        if fx is None or not (0.0 <= fx <= 1.0):
            return VerifyResult(False, "rejected", "failed_per_pos", "invalid per_pos_ref entry")
        per_pos.append(round(fx, 4))

    by_cat = data["by_cat"]
    if not isinstance(by_cat, dict) or len(json.dumps(by_cat)) > MAX_EVAL_BY_CAT_BYTES:
        return VerifyResult(False, "rejected", "failed_schema", "by_cat must be a bounded JSON object")

    quorum_note = "primary_accepted"
    for peer in peer_verified_metrics:
        p_b16 = _safe_float(peer.get("b16_ref_tau"))
        if p_b16 is None:
            continue
        p_sha = str(peer.get("pred_sha256", "")).lower()
        p_quant = str(peer.get("quant", "mxfp8"))
        c_quant = str(data.get("quant", "mxfp8"))
        if abs(b16 - p_b16) > config.EVAL_DUP_MAX_TAU_DIFF:
            return VerifyResult(
                False,
                "rejected",
                "failed_duplicate_quorum",
                "duplicate quorum tau mismatch against peer",
            )
        if p_quant == c_quant and p_sha and pred_sha != p_sha:
            return VerifyResult(
                False,
                "rejected",
                "failed_pred_sha_quorum",
                f"duplicate quorum pred_sha256 mismatch for quant={c_quant}",
            )
        quorum_note = "duplicate_quorum_matched"

    b16_bb = _safe_float(data.get("b16_bb_tau"))
    roll16 = _safe_float(data.get("roll16_tau")) if data.get("roll16_tau") is not None else None
    metrics = {
        "checkpoint_rev": str(data["checkpoint_rev"])[:128],
        "profile_id": str(data["profile_id"])[:64],
        "quant": str(data.get("quant", "mxfp8"))[:32],
        "device_class": str(data.get("device_class", "unknown"))[:128],
        "selfcheck_max_err": sc,
        "n_blocks": n_blocks,
        "b8_ref_tau": round(b8, 4),
        "b16_ref_tau": round(b16, 4),
        "b16_bb_tau": round(b16_bb if b16_bb is not None else b16, 4),
        "roll16_tau": round(roll16, 4) if roll16 is not None else None,
        "ce8": round(ce8, 4),
        "top1": round(top1, 4),
        "per_pos_ref": per_pos,
        "by_cat": by_cat,
        "pred_sha256": pred_sha,
        "quorum": quorum_note,
        "elapsed_s": _safe_float(data.get("elapsed_s")),
    }
    return VerifyResult(
        True,
        "verified",
        quorum_note,
        f"verified eval {metrics['checkpoint_rev']}@{metrics['profile_id']} B16={b16:.4f} ({quorum_note})",
        metrics,
    )


def verify_drafter_capture(
    artifact_path: Path,
    slot_plan: dict[str, Any],
    manifest: dict[str, Any],
) -> VerifyResult:
    """Verify an owner 4-tap INT8+zstd/safetensors capture shard without pickle."""
    sc = _safe_float(manifest.get("selfcheck_max_err"))
    if sc is None or sc > config.TRAIN_MAX_SELFCHECK_ERR or sc < 0.0:
        return VerifyResult(False, "rejected", "failed_selfcheck", "invalid selfcheck_max_err")

    top1 = _safe_float(manifest.get("label_top1_agree"))
    top5 = _safe_float(manifest.get("label_top5_agree")) if manifest.get("label_top5_agree") is not None else None
    if top1 is None or not (config.LABEL_MIN_TOP1_AGREE <= top1 <= 1.0):
        return VerifyResult(False, "rejected", "failed_agreement_band", "label_top1_agree outside required band")
    if manifest.get("label_top5_agree") is not None and (top5 is None or top5 < config.LABEL_MIN_TOP5_AGREE):
        return VerifyResult(False, "rejected", "failed_agreement_band", "label_top5_agree outside required band")

    expected_layers = slot_plan.get("tap_layers", [33, 47, 61, 75])
    if manifest.get("tap_layers") != expected_layers:
        return VerifyResult(False, "rejected", "failed_tap_layers", "tap_layers mismatch")

    try:
        with artifact_path.open("rb") as f:
            magic = f.read(4)
        if magic == b"\x28\xb5\x2f\xfd":  # zstd magic
            import zstandard as zstd

            dctx = zstd.ZstdDecompressor()
            max_decomp = 512 * 1024 * 1024
            buf = io.BytesIO()
            total = 0
            with artifact_path.open("rb") as f, dctx.stream_reader(f) as reader:
                while True:
                    chunk = reader.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_decomp:
                        return VerifyResult(False, "rejected", "failed_decompress", "zstd payload exceeds 512 MiB cap")
                    buf.write(chunk)
            st_bytes = buf.getvalue()
        else:
            st_bytes = artifact_path.read_bytes()
        from safetensors.numpy import load as st_load_bytes

        tensors = st_load_bytes(st_bytes)
    except Exception as e:
        return VerifyResult(False, "rejected", "failed_safetensors", f"safe tensor parse failed: {e}")

    req_tensors = ("taps_int8", "taps_scale", "tokens", "topk_indices", "topk_values")
    missing = [k for k in req_tensors if k not in tensors]
    if missing:
        return VerifyResult(False, "rejected", "failed_tensors", f"missing tensors: {missing}")

    taps_q = tensors["taps_int8"]
    taps_s = tensors["taps_scale"].astype(np.float32)
    topk_v = tensors["topk_values"]
    topk_i = tensors["topk_indices"]
    tokens = tensors["tokens"]

    if taps_q.dtype != np.int8 or taps_q.ndim != 3 or taps_q.shape[1] != len(expected_layers) or taps_q.shape[2] != 6144:
        return VerifyResult(False, "rejected", "failed_shape", "taps_int8 shape/dtype invalid")
    if taps_s.shape[:2] != taps_q.shape[:2]:
        return VerifyResult(False, "rejected", "failed_scale_shape", "taps_scale shape mismatch")
    if tokens.ndim != 1 or tokens.shape[0] < taps_q.shape[0]:
        return VerifyResult(False, "rejected", "failed_shape", "tokens shape invalid")
    if topk_i.ndim != 2 or topk_i.shape != topk_v.shape:
        return VerifyResult(False, "rejected", "failed_shape", "topk_indices/topk_values shape mismatch")
    if topk_i.size > 0 and (int(topk_i.min()) < 0 or int(topk_i.max()) >= config.VOCAB_SIZE):
        return VerifyResult(False, "rejected", "failed_vocab", "topk_indices out of vocab range")
    if not np.all(np.isfinite(taps_s)) or not np.all(np.isfinite(topk_v)):
        return VerifyResult(False, "rejected", "failed_nan", "NaN or Inf detected in capture tensors")

    # Compute variance and scaled abs mean row-by-row to avoid 4x full-array float32 blowup
    n_rows = int(taps_q.shape[0])
    if n_rows == 0:
        return VerifyResult(False, "rejected", "failed_shape", "0 rows in taps_int8")
    sum_q = 0.0
    sum_q2 = 0.0
    abs_scaled_sum = 0.0
    elems_per_row = taps_q.shape[1] * taps_q.shape[2]
    for r in range(n_rows):
        row_f = taps_q[r].astype(np.float32)
        sum_q += float(np.sum(row_f))
        sum_q2 += float(np.sum(row_f * row_f))
        sc_r = taps_s[r]
        if sc_r.ndim == 1:
            sc_r = sc_r[:, None]
        abs_scaled_sum += float(np.sum(np.abs(row_f * sc_r)))
    tot_elems = n_rows * elems_per_row
    mean_q = sum_q / tot_elems
    std_q = math.sqrt(max(0.0, (sum_q2 / tot_elems) - mean_q * mean_q))
    if std_q < 1.0:
        return VerifyResult(False, "rejected", "failed_degenerate", "degenerate near-constant taps_int8 tensor")

    comp_bytes = artifact_path.stat().st_size
    metrics = {
        "profile_id": slot_plan.get("profile_id"),
        "format": str(manifest.get("format", "4tap_int8_zstd"))[:64],
        "tap_layers": expected_layers,
        "n_rows": n_rows,
        "n_tokens": int(tokens.shape[0]),
        "selfcheck_max_err": sc,
        "label_top1_agree": round(top1, 5),
        "label_top5_agree": round(top5, 5) if top5 is not None else None,
        "compressed_bytes": comp_bytes,
        "uncompressed_st_bytes": len(st_bytes),
        "compression_ratio": round(len(st_bytes) / max(1, comp_bytes), 2),
        "taps_abs_mean": round(abs_scaled_sum / tot_elems, 5),
    }
    return VerifyResult(
        True,
        "verified",
        "verified_capture_4tap_int8",
        f"verified 4-tap INT8 capture ({n_rows} rows, ratio={metrics['compression_ratio']}x)",
        metrics,
    )


def verify_drafter_train_phase_a(
    artifact_path: Path,
    slot_plan: dict[str, Any],
    manifest: dict[str, Any],
) -> VerifyResult:
    """Phase A (CPU) safe-parse gate for owner drafter-train .safetensors deltas (zero pickle)."""
    sc = _safe_float(manifest.get("selfcheck_max_err"))
    if sc is None or sc > config.TRAIN_MAX_SELFCHECK_ERR or sc < 0.0:
        return VerifyResult(False, "rejected", "failed_selfcheck", "invalid selfcheck_max_err")

    top1 = _safe_float(manifest.get("label_top1_agree"))
    top5 = _safe_float(manifest.get("label_top5_agree")) if manifest.get("label_top5_agree") is not None else None
    if top1 is None or not (config.LABEL_MIN_TOP1_AGREE <= top1 <= 1.0):
        return VerifyResult(False, "rejected", "failed_agreement_band", "label_top1_agree outside required band")
    if manifest.get("label_top5_agree") is not None and (top5 is None or top5 < config.LABEL_MIN_TOP5_AGREE):
        return VerifyResult(False, "rejected", "failed_agreement_band", "label_top5_agree outside required band")

    expected_codec = slot_plan.get("codec")
    codec = str(manifest.get("codec") or "")
    if expected_codec and codec != expected_codec:
        return VerifyResult(False, "rejected", "failed_codec", "codec does not match claim plan")

    try:
        n_tensors = 0
        sq_norm = 0.0
        total_nonzero = 0
        total_elems = 0
        safe_meta: dict[str, str] = {}
        with safe_open(str(artifact_path), framework="numpy") as f:
            raw_meta = f.metadata() or {}
            for mk in ("codec", "keep_frac", "base_ckpt_sha256"):
                if mk in raw_meta:
                    safe_meta[mk] = str(raw_meta[mk])[:128]
            keys = list(f.keys())
            key_set = set(keys)
            if not keys:
                return VerifyResult(False, "rejected", "failed_empty_safetensors", "safetensors has 0 tensors")

            if codec == "sparseloco_int8_topk10":
                base_names: set[str] = set()
                for k in keys:
                    if k.endswith(".__q"):
                        base_names.add(k[:-4])
                    elif k.endswith(".__dense"):
                        base_names.add(k[:-8])
                if not base_names:
                    return VerifyResult(False, "rejected", "failed_codec_keys", "no SparseLoCo .__q/.__dense keys found")
                for b in sorted(base_names):
                    if f"{b}.__q" in key_set:
                        if f"{b}.__scale" not in key_set:
                            return VerifyResult(False, "rejected", "failed_codec_keys", f"missing __scale on {b}")
                        q = f.get_tensor(f"{b}.__q")
                        scale = f.get_tensor(f"{b}.__scale").astype(np.float32)
                        if q.dtype != np.int8:
                            return VerifyResult(False, "rejected", "failed_tensor_dtype", f"non-int8 __q on {b}")
                        if scale.size == 0 or not np.all(np.isfinite(scale)):
                            return VerifyResult(False, "rejected", "failed_nan", f"NaN/Inf scale on {b}")
                        if f"{b}.__idx" in key_set:
                            idx = f.get_tensor(f"{b}.__idx")
                            if q.shape != idx.shape or not np.issubdtype(idx.dtype, np.integer):
                                return VerifyResult(False, "rejected", "failed_tensor_shape", f"bad sparse idx on {b}")
                            if idx.size > 0 and int(idx.min()) < 0:
                                return VerifyResult(False, "rejected", "failed_tensor_shape", f"negative idx on {b}")
                        elif f"{b}.__mask" in key_set:
                            msk = f.get_tensor(f"{b}.__mask")
                            if msk.dtype != np.uint8:
                                return VerifyResult(False, "rejected", "failed_tensor_dtype", f"non-uint8 __mask on {b}")
                            # Validate mask popcount matches q.size (bounded check)
                            popcnt = int(np.unpackbits(msk).sum())
                            if popcnt < q.size:
                                return VerifyResult(False, "rejected", "failed_tensor_shape", f"mask popcount < q.size on {b}")
                        else:
                            return VerifyResult(False, "rejected", "failed_codec_keys", f"missing __mask or __idx on {b}")
                        sq_norm += _chunked_sq_norm_int8(q, float(scale.reshape(-1)[0]))
                        total_nonzero += int(np.count_nonzero(q))
                        total_elems += int(q.size)
                        n_tensors += 1
                    else:
                        arr = f.get_tensor(f"{b}.__dense")
                        sn, ok = _chunked_sq_norm_float(arr)
                        if not ok:
                            return VerifyResult(False, "rejected", "failed_nan", f"NaN/Inf dense delta on {b}")
                        sq_norm += sn
                        total_nonzero += int(np.count_nonzero(arr))
                        total_elems += int(arr.size)
                        n_tensors += 1
            else:
                base_q = {k[:-4] for k in keys if k.endswith(".__q")}
                if base_q:
                    for b in sorted(base_q):
                        if f"{b}.__scale" not in key_set:
                            return VerifyResult(False, "rejected", "failed_codec_keys", f"missing __scale on {b}")
                        q = f.get_tensor(f"{b}.__q")
                        scale = f.get_tensor(f"{b}.__scale").astype(np.float32)
                        if q.dtype != np.int8 or scale.size == 0 or not np.all(np.isfinite(scale)):
                            return VerifyResult(False, "rejected", "failed_tensor_dtype", f"invalid int8 tensor {b}")
                        sq_norm += _chunked_sq_norm_int8(q, float(scale.reshape(-1)[0]))
                        total_nonzero += int(np.count_nonzero(q))
                        total_elems += int(q.size)
                        n_tensors += 1
                else:
                    for k in keys:
                        arr = f.get_tensor(k)
                        sn, ok = _chunked_sq_norm_float(arr)
                        if not ok:
                            return VerifyResult(False, "rejected", "failed_nan", f"NaN/Inf in tensor {k}")
                        sq_norm += sn
                        total_nonzero += int(np.count_nonzero(arr))
                        total_elems += int(arr.size)
                        n_tensors += 1
    except Exception as e:
        return VerifyResult(False, "rejected", "failed_safetensors", f"safetensors parse error: {e}")

    l2_norm = math.sqrt(max(0.0, sq_norm))
    if not math.isfinite(l2_norm):
        return VerifyResult(False, "rejected", "failed_nan", "non-finite delta L2 norm")

    metrics = {
        "codec": codec,
        "profile_id": slot_plan.get("profile_id"),
        "round_num": _safe_int(slot_plan.get("round_num")) or 1,
        "selfcheck_max_err": sc,
        "label_top1_agree": round(top1, 5),
        "label_top5_agree": round(top5, 5) if top5 is not None else None,
        "n_param_tensors": n_tensors,
        "encoded_elems": total_elems,
        "nonzero_elems": total_nonzero,
        "server_delta_l2_norm": round(l2_norm, 4),
        "reported_delta_l2_norm": _safe_float(manifest.get("delta_l2_norm")),
        "local_loss_start": _safe_float(manifest.get("local_loss_start")),
        "local_loss_end": _safe_float(manifest.get("local_loss_end")),
        "local_steps": _safe_int(manifest.get("local_steps")),
        "upload_bytes": artifact_path.stat().st_size,
        "st_meta": safe_meta,
    }
    return VerifyResult(
        True,
        "uploaded",
        "pending_replay_gate",
        f"Phase A safe-parse passed ({n_tensors} tensors, L2={l2_norm:.3f}, 0 NaNs); queued for Owner GPU Gate",
        metrics,
    )


def verify_drafter_pack(
    artifact_path: Path,
    slot_plan: dict[str, Any],
    peer_verified_metrics: list[dict[str, Any]],
) -> VerifyResult:
    """Verify an owner CPU g6load dspine rank-pack manifest JSON upload."""
    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception as e:
        return VerifyResult(False, "rejected", "failed_json", f"pack manifest JSON parse error: {e}")
    if not isinstance(data, dict):
        return VerifyResult(False, "rejected", "failed_schema", "pack manifest must be a JSON object")

    for k in ("checkpoint_rev", "ckpt_sha256", "nr", "format_version", "ranks"):
        if k not in data:
            return VerifyResult(False, "rejected", "failed_schema", f"pack manifest missing key '{k}'")

    nr = _safe_int(data["nr"])
    if nr is None or nr not in (1, 2, 4, 8):
        return VerifyResult(False, "rejected", "failed_nr", "unsupported tp nr")
    if slot_plan.get("nr") and nr != _safe_int(slot_plan["nr"]):
        return VerifyResult(False, "rejected", "failed_nr", "nr does not match plan")

    if str(data["format_version"]) != "g6dspine-v1":
        return VerifyResult(False, "rejected", "failed_format", "unexpected format_version")

    ckpt_sha = str(data["ckpt_sha256"]).lower()
    if not HEX64_RE.match(ckpt_sha):
        return VerifyResult(False, "rejected", "failed_ckpt_sha", "ckpt_sha256 must be 64-char hex")
    if slot_plan.get("ckpt_sha256") and ckpt_sha != str(slot_plan["ckpt_sha256"]).lower():
        return VerifyResult(False, "rejected", "failed_ckpt_sha", "ckpt_sha256 does not match planned checkpoint")

    ranks = data["ranks"]
    if not isinstance(ranks, list) or len(ranks) != nr:
        return VerifyResult(False, "rejected", "failed_ranks", f"expected {nr} rank entries")

    rank_shas: list[str] = []
    total_pack_bytes = 0
    for idx, rk in enumerate(ranks):
        if not isinstance(rk, dict) or _safe_int(rk.get("rank")) != idx:
            return VerifyResult(False, "rejected", "failed_rank_index", f"rank entry {idx} malformed")
        rsha = str(rk.get("pack_sha256", "")).lower()
        rbytes = _safe_int(rk.get("pack_bytes"))
        if not HEX64_RE.match(rsha) or rbytes is None or rbytes <= 0:
            return VerifyResult(False, "rejected", "failed_rank_sha", f"rank {idx} invalid sha256/bytes")
        rank_shas.append(rsha)
        total_pack_bytes += rbytes

    quorum_note = "verified_bitexact_pack"
    for peer in peer_verified_metrics:
        p_shas = peer.get("rank_pack_sha256s") or []
        if p_shas and p_shas != rank_shas:
            return VerifyResult(False, "rejected", "failed_pack_quorum", "bit-exact rank pack SHA-256 mismatch against peer")
        if p_shas:
            quorum_note = "bitexact_pack_quorum_matched"

    metrics = {
        "checkpoint_rev": str(data["checkpoint_rev"])[:128],
        "ckpt_sha256": ckpt_sha,
        "nr": nr,
        "format_version": str(data["format_version"]),
        "rank_pack_sha256s": rank_shas,
        "total_pack_bytes": total_pack_bytes,
        "elapsed_s": _safe_float(data.get("elapsed_s")),
        "g6load_sha256": str(data.get("g6load_sha256") or "")[:64] or None,
        "quorum": quorum_note,
    }
    return VerifyResult(
        True,
        "verified",
        quorum_note,
        f"verified deterministic CPU g6load dspine TP{nr} pack ({nr} ranks, {total_pack_bytes/1e9:.2f} GB)",
        metrics,
    )
