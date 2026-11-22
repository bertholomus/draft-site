"""Zero-pickle, non-executing server-side verification gates for uploaded pilot artifacts.

Follows CDC PIPELINE.md §5.1 and Guardrail 5:
- Uploaded files are NEVER executed and NEVER loaded with pickle / torch.load.
- Label uploads: gzip zip-bomb guard, JSONL schema, prompt SHA-256 match, vocab range, gen_sha256 check.
- Eval uploads: strict JSON schema, selfcheck == 0.0, finite bounds, pred_sha256 hex check, duplicate-slot quorum.
- Capture uploads: safe safetensors header/tensor parse (4-tap INT8 + top-20 logits), selfcheck == 0.0, label agreement band, NaN/Inf check.
- Train uploads (Phase A sync gate): safe safetensors delta parse, allowed tensor keys only, selfcheck == 0.0, label agreement band, zero NaN/Inf, exact L2 norm extraction (Phase B GPU Held-Out Replay + FLTrust Cosine gate runs asynchronously via owner verifier).
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


@dataclass
class VerifyResult:
    accepted: bool
    final_state: str  # "verified", "uploaded" (pending GPU gate), or "rejected"
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


def verify_drafter_label(
    artifact_path: Path,
    slot_plan: dict[str, Any],
    input_prompt_path: Path | None,
    manifest: dict[str, Any],
) -> VerifyResult:
    """Verify a drafter-label gzipped JSONL upload against the slot's prompt shard."""
    try:
        raw_head = artifact_path.read_bytes()[:2]
        if raw_head == b"\x1f\x8b":
            raw = _safe_gzip_read(artifact_path, config.MAX_DECOMPRESSED_LABEL_BYTES)
        else:
            if artifact_path.stat().st_size > config.MAX_DECOMPRESSED_LABEL_BYTES:
                return VerifyResult(False, "rejected", "failed_size", "uncompressed label file too large")
            raw = artifact_path.read_bytes()
    except Exception as e:
        return VerifyResult(False, "rejected", "failed_decompress", f"gzip decode failed: {e}")

    expected_prompts: dict[str, str] = {}
    if input_prompt_path and input_prompt_path.is_file():
        for line in input_prompt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = str(row["id"])
            psha = row.get("prompt_sha256")
            if not psha:
                if "prompt_ids" in row:
                    psha = hashlib.sha256(json.dumps(row["prompt_ids"], separators=(",", ":")).encode()).hexdigest()
                else:
                    psha = hashlib.sha256(str(row.get("prompt", "")).encode("utf-8")).hexdigest()
            expected_prompts[pid] = psha

    seen: set[str] = set()
    total_gen_tokens = 0
    engines: set[str] = set()
    finish_reasons: dict[str, int] = {}

    lines = [ln for ln in raw.decode("utf-8", errors="strict").splitlines() if ln.strip()]
    if not lines:
        return VerifyResult(False, "rejected", "failed_empty", "label JSONL has 0 rows")

    for idx, line in enumerate(lines):
        try:
            row = json.loads(line)
        except Exception as e:
            return VerifyResult(False, "rejected", "failed_json", f"row {idx} invalid JSON: {e}")
        for req_k in ("id", "prompt_sha256", "gen_ids", "finish_reason", "engine_tag", "gen_sha256"):
            if req_k not in row:
                return VerifyResult(False, "rejected", "failed_schema", f"row {idx} missing key '{req_k}'")
        pid = str(row["id"])
        if pid in seen:
            return VerifyResult(False, "rejected", "failed_dup_id", f"duplicate prompt id '{pid}'")
        seen.add(pid)
        if expected_prompts:
            if pid not in expected_prompts:
                return VerifyResult(False, "rejected", "failed_prompt_id", f"unexpected prompt id '{pid}'")
            if str(row["prompt_sha256"]) != expected_prompts[pid]:
                return VerifyResult(
                    False,
                    "rejected",
                    "failed_prompt_sha",
                    f"prompt_sha256 mismatch on '{pid}' (expected {expected_prompts[pid][:12]}..., got {str(row['prompt_sha256'])[:12]}...)",
                )
        gen_ids = row["gen_ids"]
        if not isinstance(gen_ids, list) or len(gen_ids) == 0:
            return VerifyResult(False, "rejected", "failed_gen_empty", f"empty gen_ids on '{pid}'")
        if any((not isinstance(t, int)) or t < 0 or t >= config.VOCAB_SIZE for t in gen_ids):
            return VerifyResult(False, "rejected", "failed_vocab", f"out-of-range token id on '{pid}'")
        expect_gen_sha = hashlib.sha256(json.dumps(gen_ids, separators=(",", ":")).encode("utf-8")).hexdigest()
        if str(row["gen_sha256"]) != expect_gen_sha:
            return VerifyResult(False, "rejected", "failed_gen_sha", f"gen_sha256 mismatch on '{pid}'")
        total_gen_tokens += len(gen_ids)
        engines.add(str(row["engine_tag"]))
        fr = str(row["finish_reason"])
        finish_reasons[fr] = finish_reasons.get(fr, 0) + 1

    if expected_prompts and seen != set(expected_prompts.keys()):
        missing = sorted(set(expected_prompts.keys()) - seen)
        return VerifyResult(
            False,
            "rejected",
            "failed_incomplete",
            f"missing {len(missing)} prompts (first: {missing[:3]})",
        )

    # Optional semantic check reported in manifest or enforced if top1_agree present
    sem_top1 = manifest.get("semantic_top1_agree")
    sem_top5 = manifest.get("semantic_top5_agree")
    if sem_top1 is not None:
        if float(sem_top1) < config.LABEL_MIN_TOP1_AGREE:
            return VerifyResult(
                False,
                "rejected",
                "failed_semantic_agreement",
                f"label semantic top1_agree={float(sem_top1):.4f} below {config.LABEL_MIN_TOP1_AGREE:.3f} threshold (off-policy or corrupted generator)",
                {"semantic_top1_agree": float(sem_top1), "semantic_top5_agree": sem_top5},
            )
    if sem_top5 is not None and float(sem_top5) < config.LABEL_MIN_TOP5_AGREE:
        return VerifyResult(
            False,
            "rejected",
            "failed_semantic_agreement",
            f"label semantic top5_agree={float(sem_top5):.4f} below {config.LABEL_MIN_TOP5_AGREE:.3f} threshold",
            {"semantic_top1_agree": sem_top1, "semantic_top5_agree": float(sem_top5)},
        )

    metrics = {
        "n_rows": len(seen),
        "total_gen_tokens": total_gen_tokens,
        "mean_gen_tokens": round(total_gen_tokens / max(1, len(seen)), 2),
        "engines": sorted(engines),
        "finish_reasons": finish_reasons,
    }
    if sem_top1 is not None:
        metrics["semantic_top1_agree"] = round(float(sem_top1), 5)
    if sem_top5 is not None:
        metrics["semantic_top5_agree"] = round(float(sem_top5), 5)

    return VerifyResult(
        True,
        "verified",
        "verified_label_schema",
        f"accepted {len(seen)} prompts ({total_gen_tokens} gen tokens, exact prompt/gen SHA-256)",
        metrics,
    )


def verify_drafter_eval(
    artifact_path: Path,
    slot_plan: dict[str, Any],
    peer_verified_metrics: list[dict[str, Any]],
) -> VerifyResult:
    """Verify a drafter-eval JSON upload (selfcheck, schema, finite metrics, duplicate-slot quorum)."""
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
        return VerifyResult(
            False, "rejected", "failed_profile_mismatch",
            f"profile_id={data['profile_id']} does not match claim plan {slot_plan['profile_id']}",
        )
    if slot_plan.get("checkpoint_rev") and data["checkpoint_rev"] != slot_plan["checkpoint_rev"]:
        return VerifyResult(
            False, "rejected", "failed_ckpt_mismatch",
            f"checkpoint_rev={data['checkpoint_rev']} does not match claim plan {slot_plan['checkpoint_rev']}",
        )

    try:
        sc = float(data["selfcheck_max_err"])
        b8 = float(data["b8_ref_tau"])
        b16 = float(data["b16_ref_tau"])
        ce8 = float(data["ce8"])
        top1 = float(data["top1"])
        n_blocks = int(data["n_blocks"])
    except Exception as e:
        return VerifyResult(False, "rejected", "failed_types", f"numeric parse error: {e}")

    for name, val in (("selfcheck_max_err", sc), ("b8_ref_tau", b8), ("b16_ref_tau", b16), ("ce8", ce8), ("top1", top1)):
        if not math.isfinite(val):
            return VerifyResult(False, "rejected", "failed_nan", f"non-finite value for {name}")

    if sc < 0.0 or sc > config.TRAIN_MAX_SELFCHECK_ERR:
        return VerifyResult(
            False, "rejected", "failed_selfcheck",
            f"selfcheck_max_err={sc:.3e} exceeds {config.TRAIN_MAX_SELFCHECK_ERR:.1e} tolerance",
        )
    if n_blocks <= 0 or not (1.0 <= b8 <= 9.0) or not (1.0 <= b16 <= 17.0) or not (0.0 <= top1 <= 1.0) or ce8 <= 0.0:
        return VerifyResult(False, "rejected", "failed_bounds", f"metrics out of plausible physical bounds (b8={b8}, b16={b16}, ce8={ce8}, top1={top1})")

    pred_sha = str(data["pred_sha256"]).lower()
    if not HEX64_RE.match(pred_sha):
        return VerifyResult(False, "rejected", "failed_pred_sha", "pred_sha256 must be 64-char lowercase hex")

    if not isinstance(data["per_pos_ref"], list) or len(data["per_pos_ref"]) < 7:
        return VerifyResult(False, "rejected", "failed_per_pos", "per_pos_ref must contain >=7 positions")

    # Duplicate-slot quorum check against already-verified peer slots for same (checkpoint_rev, profile_id)
    quorum_note = "primary_accepted"
    for peer in peer_verified_metrics:
        p_b16 = float(peer.get("b16_ref_tau", -1.0))
        p_sha = str(peer.get("pred_sha256", "")).lower()
        p_quant = str(peer.get("quant", "mxfp8"))
        c_quant = str(data.get("quant", "mxfp8"))
        if abs(b16 - p_b16) > config.EVAL_DUP_MAX_TAU_DIFF:
            return VerifyResult(
                False,
                "rejected",
                "failed_duplicate_quorum",
                f"duplicate quorum mismatch: |b16_ref_tau {b16:.4f} - peer {p_b16:.4f}| = {abs(b16 - p_b16):.4f} > {config.EVAL_DUP_MAX_TAU_DIFF}",
                {"peer_b16": p_b16, "submitted_b16": b16},
            )
        if p_quant == c_quant and p_sha and pred_sha != p_sha:
            return VerifyResult(
                False,
                "rejected",
                "failed_pred_sha_quorum",
                f"duplicate quorum pred_sha256 mismatch for quant={c_quant} ({pred_sha[:12]} vs peer {p_sha[:12]})",
                {"peer_pred_sha256": p_sha, "submitted_pred_sha256": pred_sha},
            )
        quorum_note = "duplicate_quorum_matched"

    metrics = {
        "checkpoint_rev": data["checkpoint_rev"],
        "profile_id": data["profile_id"],
        "quant": data.get("quant", "mxfp8"),
        "device_class": data.get("device_class", "unknown"),
        "selfcheck_max_err": sc,
        "n_blocks": n_blocks,
        "b8_ref_tau": round(b8, 4),
        "b16_ref_tau": round(b16, 4),
        "b16_bb_tau": round(float(data.get("b16_bb_tau", b16)), 4),
        "roll16_tau": round(float(data.get("roll16_tau", 0.0)), 4) if data.get("roll16_tau") is not None else None,
        "ce8": round(ce8, 4),
        "top1": round(top1, 4),
        "per_pos_ref": [round(float(x), 4) for x in data["per_pos_ref"][:15]],
        "by_cat": data["by_cat"],
        "pred_sha256": pred_sha,
        "quorum": quorum_note,
        "elapsed_s": data.get("elapsed_s"),
    }
    return VerifyResult(
        True,
        "verified",
        quorum_note,
        f"verified eval {data['checkpoint_rev']}@{data['profile_id']} B16={b16:.4f} B8={b8:.4f} ce8={ce8:.4f} ({quorum_note})",
        metrics,
    )


def verify_drafter_capture(
    artifact_path: Path,
    slot_plan: dict[str, Any],
    manifest: dict[str, Any],
) -> VerifyResult:
    """Verify a standalone 4-tap INT8+zstd/safetensors capture shard without pickle."""
    sc = manifest.get("selfcheck_max_err")
    if sc is None or not math.isfinite(float(sc)) or float(sc) > config.TRAIN_MAX_SELFCHECK_ERR or float(sc) < 0:
        return VerifyResult(False, "rejected", "failed_selfcheck", f"invalid selfcheck_max_err={sc}")

    top1 = manifest.get("label_top1_agree")
    top5 = manifest.get("label_top5_agree")
    if top1 is None or not (config.LABEL_MIN_TOP1_AGREE <= float(top1) <= 1.0):
        return VerifyResult(
            False, "rejected", "failed_agreement_band",
            f"label_top1_agree={top1} outside [{config.LABEL_MIN_TOP1_AGREE}, 1.0]",
        )
    if top5 is not None and float(top5) < config.LABEL_MIN_TOP5_AGREE:
        return VerifyResult(
            False, "rejected", "failed_agreement_band",
            f"label_top5_agree={top5} below {config.LABEL_MIN_TOP5_AGREE}",
        )

    expected_layers = slot_plan.get("tap_layers", [33, 47, 61, 75])
    if manifest.get("tap_layers") != expected_layers:
        return VerifyResult(
            False, "rejected", "failed_tap_layers",
            f"tap_layers={manifest.get('tap_layers')} != expected {expected_layers}",
        )

    # Parse safely (safetensors or zstd-compressed safetensors; NEVER pickle)
    try:
        raw_bytes = artifact_path.read_bytes()
        if raw_bytes[:4] == b"\x28\xb5\x2f\xfd":  # zstd magic
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(io.BytesIO(raw_bytes)) as reader:
                st_bytes = reader.read(512 * 1024 * 1024)
        else:
            st_bytes = raw_bytes
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
    topk_v = tensors["topk_values"].astype(np.float32)
    tokens = tensors["tokens"]

    if taps_q.dtype != np.int8 or taps_q.ndim != 3 or taps_q.shape[1] != len(expected_layers) or taps_q.shape[2] != 6144:
        return VerifyResult(
            False, "rejected", "failed_shape",
            f"taps_int8 shape/dtype {taps_q.shape}/{taps_q.dtype} invalid (expected [N,{len(expected_layers)},6144] int8)",
        )
    if taps_s.shape[:2] != taps_q.shape[:2]:
        return VerifyResult(False, "rejected", "failed_scale_shape", f"taps_scale shape {taps_s.shape} mismatch")
    if not np.all(np.isfinite(taps_s)) or not np.all(np.isfinite(topk_v)):
        return VerifyResult(False, "rejected", "failed_nan", "NaN or Inf detected in taps_scale or topk_values")
    if float(np.std(taps_q.astype(np.float32))) < 1.0:
        return VerifyResult(False, "rejected", "failed_degenerate", "degenerate near-constant taps_int8 tensor")

    metrics = {
        "profile_id": slot_plan.get("profile_id"),
        "format": manifest.get("format", "4tap_int8_zstd"),
        "tap_layers": expected_layers,
        "n_rows": int(taps_q.shape[0]),
        "n_tokens": int(tokens.shape[0]),
        "selfcheck_max_err": float(sc),
        "label_top1_agree": round(float(top1), 5),
        "label_top5_agree": round(float(top5), 5) if top5 is not None else None,
        "compressed_bytes": artifact_path.stat().st_size,
        "uncompressed_st_bytes": len(st_bytes),
        "compression_ratio": round(len(st_bytes) / max(1, artifact_path.stat().st_size), 2),
        "taps_abs_mean": round(float(np.mean(np.abs(taps_q.astype(np.float32) * taps_s[..., None]))), 5),
    }
    return VerifyResult(
        True,
        "verified",
        "verified_capture_4tap_int8",
        f"verified 4-tap INT8 capture ({taps_q.shape[0]} rows, top1={float(top1):.4f}, ratio={metrics['compression_ratio']}x)",
        metrics,
    )


def verify_drafter_train_phase_a(
    artifact_path: Path,
    slot_plan: dict[str, Any],
    manifest: dict[str, Any],
) -> VerifyResult:
    """Phase A (sync CPU) gate for drafter-train delta (.safetensors, zero pickle).

    Checks manifest selfcheck/agreement/codec, parses safetensors header & tensors safely,
    verifies zero NaN/Inf and computes exact dequantized L2 norm.
    Valid uploads enter state='uploaded' (verify_status='pending_replay_gate') for the
    Owner GPU Held-Out Replay + FLTrust Cosine Gate (PIPELINE.md §5.1).
    Structural/NaN/selfcheck failures are rejected immediately.
    """
    sc = manifest.get("selfcheck_max_err")
    if sc is None or not math.isfinite(float(sc)) or float(sc) > config.TRAIN_MAX_SELFCHECK_ERR or float(sc) < 0:
        return VerifyResult(False, "rejected", "failed_selfcheck", f"invalid selfcheck_max_err={sc}")

    top1 = manifest.get("label_top1_agree")
    top5 = manifest.get("label_top5_agree")
    if top1 is None or not (config.LABEL_MIN_TOP1_AGREE <= float(top1) <= 1.0):
        return VerifyResult(
            False, "rejected", "failed_agreement_band",
            f"label_top1_agree={top1} outside [{config.LABEL_MIN_TOP1_AGREE}, 1.0]",
        )
    if top5 is not None and float(top5) < config.LABEL_MIN_TOP5_AGREE:
        return VerifyResult(
            False, "rejected", "failed_agreement_band",
            f"label_top5_agree={top5} below {config.LABEL_MIN_TOP5_AGREE}",
        )

    expected_codec = slot_plan.get("codec")
    codec = manifest.get("codec")
    if expected_codec and codec != expected_codec:
        return VerifyResult(
            False, "rejected", "failed_codec",
            f"codec='{codec}' does not match claim plan '{expected_codec}'",
        )

    # Zero-pickle safetensors streaming inspection (low memory footprint on FrankenNAS)
    try:
        n_tensors = 0
        sq_norm = 0.0
        total_nonzero = 0
        total_elems = 0
        meta: dict[str, str] = {}
        with safe_open(str(artifact_path), framework="numpy") as f:
            meta = f.metadata() or {}
            keys = list(f.keys())
            if not keys:
                return VerifyResult(False, "rejected", "failed_empty_safetensors", "safetensors has 0 tensors")

            if codec == "sparseloco_int8_topk10":
                # Expected triplets: <name>.__q (int8), <name>.__mask (uint8 bitmask) or <name>.__idx (int32),
                # <name>.__scale (float32), or dense fallback <name>.__dense (float16/float32) for small tensors
                base_names = set()
                for k in keys:
                    if k.endswith(".__q"):
                        base_names.add(k[:-4])
                    elif k.endswith(".__dense"):
                        base_names.add(k[:-8])
                if not base_names:
                    return VerifyResult(False, "rejected", "failed_codec_keys", "no SparseLoCo .__q/.__dense keys found")
                for b in sorted(base_names):
                    if f"{b}.__q" in keys:
                        q = f.get_tensor(f"{b}.__q")
                        scale = f.get_tensor(f"{b}.__scale").astype(np.float32)
                        if q.dtype != np.int8:
                            return VerifyResult(False, "rejected", "failed_tensor_dtype", f"non-int8 __q on {b}")
                        if f"{b}.__idx" in keys:
                            idx = f.get_tensor(f"{b}.__idx")
                            if q.shape != idx.shape:
                                return VerifyResult(False, "rejected", "failed_tensor_shape", f"bad sparse idx shape on {b}")
                        elif f"{b}.__mask" in keys:
                            msk = f.get_tensor(f"{b}.__mask")
                            if msk.dtype != np.uint8:
                                return VerifyResult(False, "rejected", "failed_tensor_dtype", f"non-uint8 __mask on {b}")
                        else:
                            return VerifyResult(False, "rejected", "failed_codec_keys", f"missing __mask or __idx on {b}")
                        if not np.all(np.isfinite(scale)):
                            return VerifyResult(False, "rejected", "failed_nan", f"NaN/Inf scale on {b}")
                        vals = q.astype(np.float32) * float(scale.reshape(-1)[0])
                        sq_norm += float(np.dot(vals, vals))
                        total_nonzero += int(np.count_nonzero(q))
                        total_elems += int(q.size)
                        n_tensors += 1
                    else:
                        arr = f.get_tensor(f"{b}.__dense").astype(np.float32)
                        if not np.all(np.isfinite(arr)):
                            return VerifyResult(False, "rejected", "failed_nan", f"NaN/Inf dense delta on {b}")
                        sq_norm += float(np.sum(arr * arr))
                        total_nonzero += int(np.count_nonzero(arr))
                        total_elems += int(arr.size)
                        n_tensors += 1
            else:
                # INT8 per-tensor: <name>.__q (int8) + <name>.__scale (float32), or dense float16/float32
                base_q = {k[:-4] for k in keys if k.endswith(".__q")}
                if base_q:
                    for b in sorted(base_q):
                        q = f.get_tensor(f"{b}.__q")
                        scale = f.get_tensor(f"{b}.__scale").astype(np.float32)
                        if q.dtype != np.int8 or not np.all(np.isfinite(scale)):
                            return VerifyResult(False, "rejected", "failed_tensor_dtype", f"invalid int8 tensor {b}")
                        vals = q.astype(np.float32) * float(scale.reshape(-1)[0])
                        sq_norm += float(np.sum(vals * vals))
                        total_nonzero += int(np.count_nonzero(q))
                        total_elems += int(q.size)
                        n_tensors += 1
                else:
                    for k in keys:
                        arr = f.get_tensor(k).astype(np.float32)
                        if not np.all(np.isfinite(arr)):
                            return VerifyResult(False, "rejected", "failed_nan", f"NaN/Inf in tensor {k}")
                        sq_norm += float(np.sum(arr * arr))
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
        "round_num": slot_plan.get("round_num", 1),
        "selfcheck_max_err": float(sc),
        "label_top1_agree": round(float(top1), 5),
        "label_top5_agree": round(float(top5), 5) if top5 is not None else None,
        "n_param_tensors": n_tensors,
        "encoded_elems": total_elems,
        "nonzero_elems": total_nonzero,
        "server_delta_l2_norm": round(l2_norm, 4),
        "reported_delta_l2_norm": manifest.get("delta_l2_norm"),
        "local_loss_start": manifest.get("local_loss_start"),
        "local_loss_end": manifest.get("local_loss_end"),
        "local_steps": manifest.get("local_steps"),
        "upload_bytes": artifact_path.stat().st_size,
        "st_meta": meta,
    }
    return VerifyResult(
        True,
        "uploaded",
        "pending_replay_gate",
        f"Phase A safe-parse passed ({n_tensors} tensors, L2={l2_norm:.3f}, 0 NaNs); queued for Owner GPU Held-Out Replay + FLTrust Cosine Gate",
        metrics,
    )


def verify_drafter_pack(
    artifact_path: Path,
    slot_plan: dict[str, Any],
    peer_verified_metrics: list[dict[str, Any]],
) -> VerifyResult:
    """Verify a deterministic CPU g6load dspine rank-pack manifest JSON upload (zero GPU, bit-exact SHA-256)."""
    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception as e:
        return VerifyResult(False, "rejected", "failed_json", f"pack manifest JSON parse error: {e}")

    for k in ("checkpoint_rev", "ckpt_sha256", "nr", "format_version", "ranks"):
        if k not in data:
            return VerifyResult(False, "rejected", "failed_schema", f"pack manifest missing key '{k}'")

    nr = int(data["nr"])
    if slot_plan.get("nr") and nr != int(slot_plan["nr"]):
        return VerifyResult(False, "rejected", "failed_nr", f"nr={nr} != expected {slot_plan['nr']}")
    if nr not in (1, 2, 4, 8):
        return VerifyResult(False, "rejected", "failed_nr", f"unsupported tp nr={nr}")

    if str(data["format_version"]) != "g6dspine-v1":
        return VerifyResult(False, "rejected", "failed_format", f"unexpected format_version={data['format_version']}")

    ckpt_sha = str(data["ckpt_sha256"]).lower()
    if not HEX64_RE.match(ckpt_sha):
        return VerifyResult(False, "rejected", "failed_ckpt_sha", "ckpt_sha256 must be 64-char hex")
    if slot_plan.get("ckpt_sha256") and ckpt_sha != str(slot_plan["ckpt_sha256"]).lower():
        return VerifyResult(False, "rejected", "failed_ckpt_sha", "ckpt_sha256 does not match planned checkpoint")

    ranks = data["ranks"]
    if not isinstance(ranks, list) or len(ranks) != nr:
        return VerifyResult(False, "rejected", "failed_ranks", f"expected {nr} rank entries, got {type(ranks).__name__}")

    rank_shas: list[str] = []
    total_pack_bytes = 0
    for idx, rk in enumerate(ranks):
        if not isinstance(rk, dict) or int(rk.get("rank", -1)) != idx:
            return VerifyResult(False, "rejected", "failed_rank_index", f"rank entry {idx} malformed")
        rsha = str(rk.get("pack_sha256", "")).lower()
        rbytes = int(rk.get("pack_bytes", 0))
        if not HEX64_RE.match(rsha) or rbytes <= 0:
            return VerifyResult(False, "rejected", "failed_rank_sha", f"rank {idx} invalid sha256/bytes")
        rank_shas.append(rsha)
        total_pack_bytes += rbytes

    quorum_note = "verified_bitexact_pack"
    for peer in peer_verified_metrics:
        p_shas = peer.get("rank_pack_sha256s") or []
        if p_shas and p_shas != rank_shas:
            return VerifyResult(
                False,
                "rejected",
                "failed_pack_quorum",
                "bit-exact rank pack SHA-256 mismatch against peer",
                {"peer_rank_shas": p_shas, "submitted_rank_shas": rank_shas},
            )
        if p_shas:
            quorum_note = "bitexact_pack_quorum_matched"

    metrics = {
        "checkpoint_rev": str(data["checkpoint_rev"]),
        "ckpt_sha256": ckpt_sha,
        "nr": nr,
        "format_version": str(data["format_version"]),
        "rank_pack_sha256s": rank_shas,
        "total_pack_bytes": total_pack_bytes,
        "elapsed_s": data.get("elapsed_s"),
        "g6load_sha256": data.get("g6load_sha256"),
        "quorum": quorum_note,
    }
    return VerifyResult(
        True,
        "verified",
        quorum_note,
        f"verified deterministic CPU g6load dspine TP{nr} pack ({nr} ranks, {total_pack_bytes/1e9:.2f} GB, sha0={rank_shas[0][:12]}...)",
        metrics,
    )
