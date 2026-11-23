#!/usr/bin/env python3
"""Owner-side teacher-forced spot-rescoring gate for drafter-label uploads (LABEL-VERIFY-FIX.md).

- Polls GET /api/admin/pending-verifications for job_kind == "drafter-label" (defaulting to
  loopback http://127.0.0.1:8742 + X-Origin-Auth so the admin bearer never transits the public proxy).
- Seeds an unguessable PRNG via HMAC-SHA256(gate_secret, f"{job}:{slot_id}:{upload_sha256}").
- Samples P=8 distinct prompts x M=10 sorted offsets (always including t=0) = N=80 teacher-forced probes.
- Calls reference GLM-5.3 /v1/completions with max_tokens=1, temperature=0.0, logprobs=5,
  return_token_ids=true, return_tokens_as_token_ids=true (NEVER prompt_logprobs — zero serving stalls).
- Pauses cleanly whenever an anchored ^- (DSP|DRS) WINDOW-START is open in DSpine PROGRESS.md.
- Passes iff top1_agree >= 0.850 (>=68/80), top5_agree >= 0.960 (>=77/80), and every sampled
  prompt has prompt_top1 >= 0.50 (>=5/10). Flags canary status (>=0.920 top-1 and >=0.990 top-5).
- Posts verdict to POST /api/admin/verifications/{job}/{slot_id} and optionally harvests verified
  rows (with provenance tags) into data/exports/drs_labels.jsonl.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import hmac
import json
import random
import re
import signal
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import DraftSiteClient, DraftSiteError  # noqa: E402

STOP = False
_WIN_RE = re.compile(r"^- (?:DSP|DRS) WINDOW-(START|END)\b")

MIN_TOP1 = 0.850
MIN_TOP5 = 0.960
MIN_PROMPT_TOP1 = 0.50
CANARY_MIN_TOP1 = 0.920
CANARY_MIN_TOP5 = 0.990


def _on_sig(signum, frame):
    global STOP
    STOP = True


signal.signal(signal.SIGTERM, _on_sig)
signal.signal(signal.SIGINT, _on_sig)


def log(msg: str) -> None:
    print(f"[label-gate {time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def a14_window_is_open(dspine_progress: Path | None) -> bool:
    if dspine_progress is None or not dspine_progress.is_file():
        return False
    open_w = False
    try:
        for line in dspine_progress.read_text(encoding="utf-8", errors="ignore").splitlines()[-250:]:
            m = _WIN_RE.match(line)
            if m:
                open_w = m.group(1) == "START"
    except Exception:
        return False
    return open_w


def wait_for_lane_ready(endpoint: str, dspine_progress: Path | None, quiet: bool = False) -> str:
    logged = False
    while not STOP:
        if a14_window_is_open(dspine_progress):
            if not logged and not quiet:
                log("A1-A4 lane window active in PROGRESS.md; pausing gate cleanly until WINDOW-END...")
                logged = True
            time.sleep(15)
            continue
        logged = False
        try:
            with urllib.request.urlopen(f"{endpoint}/v1/models", timeout=10) as r:
                d = json.loads(r.read().decode("utf-8"))
                return str(d["data"][0]["id"])
        except Exception as e:
            if not quiet:
                log(f"Reference lane {endpoint} not ready ({e}); retrying in 15s...")
            time.sleep(15)
    return "glm-5.3"


def select_probe_plan(
    prompt_rows: dict[str, dict[str, Any]],
    label_rows: dict[str, dict[str, Any]],
    *,
    hmac_secret: str,
    job: str,
    slot_id: str,
    upload_sha256: str,
    n_prompts: int = 8,
    n_offsets: int = 10,
    max_probe_prefix: int = 4096,
) -> list[tuple[str, list[int], list[int], list[int]]]:
    """Deterministically sample unguessable (pid, prompt_ids, gen_ids, offsets) probes."""
    digest = hmac.new(
        hmac_secret.encode("utf-8"),
        f"{job}:{slot_id}:{upload_sha256}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    rng = random.Random(int.from_bytes(digest, "big"))

    preferred: list[str] = []
    secondary: list[str] = []
    fallback: list[str] = []
    for pid in sorted(prompt_rows.keys()):
        if pid not in label_rows:
            continue
        pids = prompt_rows[pid].get("prompt_ids") or []
        gids = label_rows[pid].get("gen_ids") or []
        if not pids or not gids:
            continue
        if len(pids) <= 1024 and len(gids) >= 32:
            preferred.append(pid)
        elif len(pids) < max_probe_prefix and len(gids) >= 16:
            secondary.append(pid)
        else:
            fallback.append(pid)

    chosen_pids: list[str] = []
    for pool in (preferred, secondary, fallback):
        need = n_prompts - len(chosen_pids)
        if need <= 0:
            break
        if len(pool) <= need:
            chosen_pids.extend(pool)
        else:
            chosen_pids.extend(rng.sample(pool, need))

    plan: list[tuple[str, list[int], list[int], list[int]]] = []
    for pid in chosen_pids:
        pids = [int(t) for t in prompt_rows[pid]["prompt_ids"]]
        gids = [int(t) for t in label_rows[pid]["gen_ids"]]
        max_off = len(gids) - 1
        if len(pids) < max_probe_prefix:
            max_off = min(max_off, max(0, max_probe_prefix - len(pids)))
        usable_len = max_off + 1
        m = min(n_offsets, usable_len)
        if m <= 1:
            offsets = [0]
        elif usable_len <= m:
            offsets = list(range(usable_len))
        else:
            # Always include t=0, plus m-1 stratified offsets across 1..max_off
            offsets = [0]
            span = usable_len - 1
            for k in range(m - 1):
                lo = 1 + (k * span) // (m - 1)
                hi = 1 + ((k + 1) * span) // (m - 1) - 1
                hi = max(lo, min(max_off, hi))
                offsets.append(rng.randint(lo, hi))
            offsets = sorted(set(offsets))
            # Top up if any stratum collided
            pool_rest = [o for o in range(1, usable_len) if o not in offsets]
            while len(offsets) < m and pool_rest:
                pick = rng.choice(pool_rest)
                pool_rest.remove(pick)
                offsets.append(pick)
            offsets.sort()
        plan.append((pid, pids, gids, offsets))
    return plan


def parse_probe_top5(resp: dict[str, Any]) -> tuple[int, set[int]]:
    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError("reference probe returned 0 choices")
    ch = choices[0]
    tids = ch.get("token_ids")
    if not isinstance(tids, list) or not tids:
        raise RuntimeError("reference probe omitted choices[0].token_ids")
    top1_id = int(tids[0])
    top5_ids: set[int] = {top1_id}
    lp = ch.get("logprobs") or {}
    top_lps = lp.get("top_logprobs") or []
    if top_lps and isinstance(top_lps[0], dict):
        for k in top_lps[0].keys():
            if isinstance(k, str) and k.startswith("token_id:"):
                try:
                    top5_ids.add(int(k.split(":", 1)[1]))
                except ValueError:
                    pass
    return top1_id, top5_ids


def probe_single_token(
    endpoint: str,
    model_id: str,
    prefix_ids: list[int],
    dspine_progress: Path | None,
    timeout: int = 120,
) -> tuple[int, set[int]]:
    payload = json.dumps(
        {
            "model": model_id,
            "prompt": prefix_ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "logprobs": 5,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "skip_special_tokens": False,
        }
    ).encode("utf-8")

    last_err: Exception | None = None
    for attempt in range(1, 10):
        if STOP:
            raise RuntimeError("label gate stopping on signal")
        if a14_window_is_open(dspine_progress):
            wait_for_lane_ready(endpoint, dspine_progress)
            if STOP:
                raise RuntimeError("label gate stopping on signal")
        try:
            req = urllib.request.Request(
                f"{endpoint}/v1/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read().decode("utf-8"))
            return parse_probe_top5(resp)
        except Exception as e:
            last_err = e
            if a14_window_is_open(dspine_progress):
                wait_for_lane_ready(endpoint, dspine_progress)
                continue
            time.sleep(min(20, 2 * attempt))
    raise RuntimeError(f"reference probe failed after retries: {last_err}")


def score_label_shard(
    prompt_file: Path,
    label_gz_file: Path,
    *,
    endpoint: str,
    model_id: str,
    hmac_secret: str,
    job: str,
    slot_id: str,
    upload_sha256: str,
    dspine_progress: Path | None = None,
    n_prompts: int = 8,
    n_offsets: int = 10,
    prompt_concurrency: int = 2,
) -> tuple[bool, str, dict[str, Any], list[dict[str, Any]]]:
    prompt_rows: dict[str, dict[str, Any]] = {}
    for ln in prompt_file.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            prompt_rows[str(r["id"])] = r

    with label_gz_file.open("rb") as f:
        head = f.read(2)
    raw = gzip.decompress(label_gz_file.read_bytes()) if head == b"\x1f\x8b" else label_gz_file.read_bytes()
    label_rows: dict[str, dict[str, Any]] = {}
    for ln in raw.decode("utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            label_rows[str(r["id"])] = r

    plan = select_probe_plan(
        prompt_rows,
        label_rows,
        hmac_secret=hmac_secret,
        job=job,
        slot_id=slot_id,
        upload_sha256=upload_sha256,
        n_prompts=n_prompts,
        n_offsets=n_offsets,
    )
    if not plan:
        return False, "no eligible prompts with prompt_ids and gen_ids for teacher-forced verification", {"n_probes": 0}, []

    t0 = time.time()

    def _probe_prompt(entry: tuple[str, list[int], list[int], list[int]]) -> tuple[str, int, int, int]:
        pid, pids, gids, offsets = entry
        p_hit1 = 0
        p_hit5 = 0
        # Ascending offset order exploits reference lane --reuse-prefix KV caching
        for off in offsets:
            expected = int(gids[off])
            prefix = pids + gids[:off]
            top1_id, top5_ids = probe_single_token(endpoint, model_id, prefix, dspine_progress)
            if expected == top1_id:
                p_hit1 += 1
            if expected in top5_ids:
                p_hit5 += 1
        return pid, len(offsets), p_hit1, p_hit5

    workers = max(1, min(int(prompt_concurrency), len(plan)))
    prompt_outcomes: list[tuple[str, int, int, int]] = []
    if workers == 1:
        for idx_p, entry in enumerate(plan, 1):
            out = _probe_prompt(entry)
            prompt_outcomes.append(out)
            log(f"  {slot_id}: prompt {idx_p}/{len(plan)} ({out[0]}) hit1={out[2]}/{out[1]} hit5={out[3]}/{out[1]}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_probe_prompt, entry) for entry in plan]
            for idx_p, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                out = fut.result()
                prompt_outcomes.append(out)
                log(f"  {slot_id}: prompt {idx_p}/{len(plan)} ({out[0]}) hit1={out[2]}/{out[1]} hit5={out[3]}/{out[1]}")

    total_hit1 = 0
    total_hit5 = 0
    total_probes = 0
    per_prompt_top1: list[float] = []
    per_prompt_summary: list[dict[str, Any]] = []
    for pid, m, p_hit1, p_hit5 in prompt_outcomes:
        total_hit1 += p_hit1
        total_hit5 += p_hit5
        total_probes += m
        p_rate = p_hit1 / max(1, m)
        per_prompt_top1.append(p_rate)
        per_prompt_summary.append({"id": pid, "probes": m, "hit1": p_hit1, "hit5": p_hit5, "top1": round(p_rate, 4)})

    elapsed = round(time.time() - t0, 2)
    top1_agree = total_hit1 / max(1, total_probes)
    top5_agree = total_hit5 / max(1, total_probes)
    min_p_top1 = min(per_prompt_top1) if per_prompt_top1 else 0.0

    passed = (
        top1_agree >= MIN_TOP1
        and top5_agree >= MIN_TOP5
        and min_p_top1 >= MIN_PROMPT_TOP1
    )
    canary_pass = top1_agree >= CANARY_MIN_TOP1 and top5_agree >= CANARY_MIN_TOP5

    gate_metrics = {
        "n_prompts": len(plan),
        "n_probes": total_probes,
        "hit1": total_hit1,
        "hit5": total_hit5,
        "top1_agree": round(top1_agree, 5),
        "top5_agree": round(top5_agree, 5),
        "min_prompt_top1": round(min_p_top1, 5),
        "canary_pass": canary_pass,
        "elapsed_s": elapsed,
        "reference_model": model_id,
    }
    if passed:
        canary_note = "" if canary_pass else " [CANARY WARNING: below 0.92/0.99 target]"
        reason = (
            f"PASS label spot gate: top1={top1_agree:.4f} ({total_hit1}/{total_probes}), "
            f"top5={top5_agree:.4f} ({total_hit5}/{total_probes}), min_prompt_top1={min_p_top1:.3f} "
            f"in {elapsed}s{canary_note}"
        )
    else:
        fails = []
        if top1_agree < MIN_TOP1:
            fails.append(f"top1={top1_agree:.4f}<{MIN_TOP1}")
        if top5_agree < MIN_TOP5:
            fails.append(f"top5={top5_agree:.4f}<{MIN_TOP5}")
        if min_p_top1 < MIN_PROMPT_TOP1:
            fails.append(f"min_prompt_top1={min_p_top1:.3f}<{MIN_PROMPT_TOP1}")
        reason = f"REJECT label spot gate ({'; '.join(fails)}) in {elapsed}s"

    # Build export records in DSpine train_labels.jsonl schema + external provenance tags
    export_recs: list[dict[str, Any]] = []
    if passed:
        for pid, pr in prompt_rows.items():
            lr = label_rows.get(pid)
            if not lr:
                continue
            pids = pr["prompt_ids"]
            gids = lr["gen_ids"]
            export_recs.append(
                {
                    "id": pid,
                    "category": pr.get("category", "code"),
                    "source": pr.get("source", "permissive"),
                    "model_id": lr.get("model_served") or model_id,
                    "engine_tag": lr.get("engine_tag"),
                    "engine_image_digest": lr.get("engine_image_digest"),
                    "engine_signature": lr.get("engine_signature"),
                    "quant_profile": lr.get("quant_profile"),
                    "concurrency": lr.get("concurrency"),
                    "token_id_source": lr.get("token_id_source"),
                    "prompt_len": len(pids),
                    "gen_len": len(gids),
                    "total_len": len(pids) + len(gids),
                    "finish_reason": lr.get("finish_reason", "stop"),
                    "gen_sha256": lr.get("gen_sha256"),
                    "prompt_ids": pids,
                    "gen_ids": gids,
                }
            )
    return passed, reason, gate_metrics, export_recs


def _harvest_rows(export_out: Path, slot_id: str, uploaded_by: str, export_recs: list[dict[str, Any]]) -> int:
    export_out.parent.mkdir(parents=True, exist_ok=True)
    marker = export_out.parent / ".harvested_slots.json"
    done_slots: dict[str, Any] = {}
    if marker.is_file():
        try:
            done_slots = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            done_slots = {}
    if slot_id in done_slots:
        return 0
    with export_out.open("a", encoding="utf-8") as ef:
        for rec in export_recs:
            rec_out = dict(rec)
            rec_out["contributor_login"] = uploaded_by
            rec_out["slot_id"] = slot_id
            ef.write(json.dumps(rec_out, separators=(",", ":")) + "\n")
        ef.flush()
    done_slots[slot_id] = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": len(export_recs), "by": uploaded_by}
    marker.write_text(json.dumps(done_slots, indent=2) + "\n", encoding="utf-8")
    return len(export_recs)


def process_pending_once(
    cli: DraftSiteClient,
    *,
    endpoint: str,
    hmac_secret: str,
    dspine_progress: Path | None,
    export_out: Path | None,
    n_prompts: int,
    n_offsets: int,
    prompt_concurrency: int = 2,
) -> int:
    pending = [x for x in cli.admin_get("/api/admin/pending-verifications") if x.get("job_kind") == "drafter-label"]
    if not pending:
        return 0
    model_id = wait_for_lane_ready(endpoint, dspine_progress)
    if STOP:
        return 0

    handled = 0
    for item in pending:
        if STOP:
            break
        job = str(item["job"])
        slot_id = str(item["slot_id"])
        art_id = str(item["artifact_id"])
        up_sha = str(item["sha256"])
        uploaded_by = str(item.get("uploaded_by") or "unknown")
        inputs = item.get("inputs") or []
        prompt_inp = next((x for x in inputs if x.get("name") in ("prompt_shard", "prompts") and x.get("artifact_id")), None)
        if prompt_inp is None:
            log(f"Skipping {job}/{slot_id}: missing prompt_shard input metadata")
            continue

        log(f"Verifying {job}/{slot_id} (by {uploaded_by}, sha={up_sha[:12]}...) via {endpoint}...")
        with tempfile.TemporaryDirectory(prefix="drs_lgate_") as td:
            td_p = Path(td)
            lbl_path = cli.download_artifact(art_id, td_p / "upload.labels.jsonl.gz", up_sha)
            prm_path = cli.download_artifact(prompt_inp["artifact_id"], td_p / "prompts.jsonl", prompt_inp.get("sha256"))
            passed, reason, gate_metrics, export_recs = score_label_shard(
                prm_path,
                lbl_path,
                endpoint=endpoint,
                model_id=model_id,
                hmac_secret=hmac_secret,
                job=job,
                slot_id=slot_id,
                upload_sha256=up_sha,
                dspine_progress=dspine_progress,
                n_prompts=n_prompts,
                n_offsets=n_offsets,
                prompt_concurrency=prompt_concurrency,
            )

        vstatus = "verified_label_spot_gate" if passed else "rejected_label_spot_gate"
        vresp = cli.admin_post(
            f"/api/admin/verifications/{job}/{slot_id}",
            {
                "passed": passed,
                "verify_status": vstatus,
                "reason": reason,
                "gate_metrics": gate_metrics,
                "reopen_on_reject": True,
            },
        )
        log(f"Verdict {job}/{slot_id}: {reason} -> server state={vresp.get('state')}")
        if passed and export_out and export_recs:
            n_exp = _harvest_rows(export_out, slot_id, uploaded_by, export_recs)
            if n_exp:
                log(f"Harvested {n_exp} verified external rows for {slot_id} -> {export_out}")
        handled += 1
    return handled


def main() -> None:
    ap = argparse.ArgumentParser(description="Owner teacher-forced spot-rescoring gate for drafter-label uploads")
    ap.add_argument("--url", default="http://127.0.0.1:8742", help="Coordinator URL (loopback recommended for admin API)")
    ap.add_argument("--admin-key-file", default="/tank/projects/draft-site/secrets/admin_api_key.txt")
    ap.add_argument("--origin-auth-file", default="/tank/projects/draft-site/secrets/origin_auth.secret")
    ap.add_argument("--endpoint", default="http://127.0.0.1:18090", help="Owner reference GLM-5.3 /v1/completions endpoint")
    ap.add_argument("--dspine-progress", default="/tank/projects/glm53-dspine/PROGRESS.md")
    ap.add_argument("--export-out", default="/tank/projects/draft-site/data/exports/drs_labels.jsonl")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--n-offsets", type=int, default=10)
    ap.add_argument("--prompt-concurrency", type=int, default=2, help="Parallel prompts probed (offsets per prompt stay ascending)")
    ap.add_argument("--poll-interval", type=int, default=30)
    ap.add_argument("--once", action="store_true", help="Process pending queue once and exit")
    ap.add_argument("--verify-local-labels", default=None, help="Offline smoke mode: path to .labels.jsonl.gz")
    ap.add_argument("--verify-local-prompts", default=None, help="Offline smoke mode: path to prompts.jsonl")
    args = ap.parse_args()

    admin_key = Path(args.admin_key_file).read_text(encoding="utf-8").strip()
    origin_auth = None
    if args.origin_auth_file and Path(args.origin_auth_file).is_file():
        origin_auth = Path(args.origin_auth_file).read_text(encoding="utf-8").strip()
    prog_p = Path(args.dspine_progress) if args.dspine_progress else None
    endpoint = args.endpoint.rstrip("/")

    if args.verify_local_labels:
        if not args.verify_local_prompts:
            raise SystemExit("--verify-local-prompts required with --verify-local-labels")
        model_id = wait_for_lane_ready(endpoint, prog_p)
        lbl_p = Path(args.verify_local_labels)
        sha = hashlib.sha256(lbl_p.read_bytes()).hexdigest()
        passed, reason, gm, _ = score_label_shard(
            Path(args.verify_local_prompts),
            lbl_p,
            endpoint=endpoint,
            model_id=model_id,
            hmac_secret=admin_key,
            job="local-smoke",
            slot_id=lbl_p.stem,
            upload_sha256=sha,
            dspine_progress=prog_p,
            n_prompts=args.n_prompts,
            n_offsets=args.n_offsets,
            prompt_concurrency=args.prompt_concurrency,
        )
        print(json.dumps({"passed": passed, "reason": reason, "gate_metrics": gm}, indent=2))
        raise SystemExit(0 if passed else 2)

    cli = DraftSiteClient(args.url, admin_key, origin_auth=origin_auth)
    me = cli.me()
    if not me.get("admin"):
        raise SystemExit("label_verify_gate requires admin credential")
    log(f"Started owner label verification gate as {me['login']} against {args.url} (ref={endpoint})")
    exp_p = Path(args.export_out) if args.export_out else None

    while not STOP:
        try:
            n = process_pending_once(
                cli,
                endpoint=endpoint,
                hmac_secret=admin_key,
                dspine_progress=prog_p,
                export_out=exp_p,
                n_prompts=args.n_prompts,
                n_offsets=args.n_offsets,
                prompt_concurrency=args.prompt_concurrency,
            )
            if args.once:
                log(f"Processed {n} pending label slot(s); exiting (--once).")
                break
        except DraftSiteError as e:
            log(f"API error (HTTP {e.status}): {e.body}")
            if args.once:
                raise
        except Exception as e:
            log(f"Gate loop error: {e}")
            if args.once:
                raise
        if not STOP:
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
