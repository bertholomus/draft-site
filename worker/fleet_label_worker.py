#!/usr/bin/env python3
"""Persistent concurrent (C=8) drafter-label fleet worker & verified-label harvester for draft.bertholomus.com.

- Claims production 200-prompt shards from glm53-dspine-labels-prod via https://draft.bertholomus.com
- Generates exact token-ID greedy labels (/v1/completions with prompt_ids + return_token_ids=True) at C=8 (~156 tok/s)
- Respects A1-A4 lane windows cleanly (anchored ^- (DSP|DRS) WINDOW-START/END; window waits NEVER consume retries)
- Persists every completed prompt to a per-slot disk cache so zero completed prompts are ever discarded
- Uploads verified .labels.jsonl.gz via https://draft.bertholomus.com
- Appends verified rows in exact DSpine train_labels.jsonl schema to --export-out
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import re
import signal
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import DraftSiteClient, DraftSiteError  # noqa: E402

STOP = False
_WIN_RE = re.compile(r"^- (?:DSP|DRS) WINDOW-(START|END)\b")


def _on_sig(signum, frame):
    global STOP
    STOP = True


signal.signal(signal.SIGTERM, _on_sig)
signal.signal(signal.SIGINT, _on_sig)


def log(msg: str) -> None:
    print(f"[drs-label {time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def a14_window_is_open(dspine_progress: Path) -> bool:
    if not dspine_progress.is_file():
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


def endpoint_is_a14(endpoint: str) -> bool:
    return "127.0.0.1:18090" in endpoint or "localhost:18090" in endpoint or ":8100" in endpoint


def wait_for_lane_ready(endpoint: str, dspine_progress: Path | None, quiet: bool = False) -> str:
    """Wait indefinitely (until STOP) until any A1-A4 lane window ends and /v1/models responds 200."""
    logged_win = False
    while not STOP:
        if dspine_progress and endpoint_is_a14(endpoint) and a14_window_is_open(dspine_progress):
            if not logged_win and not quiet:
                log("A1-A4 lane window active in DSpine PROGRESS; pausing cleanly until WINDOW-END...")
                logged_win = True
            time.sleep(15)
            continue
        logged_win = False
        try:
            with urllib.request.urlopen(f"{endpoint}/v1/models", timeout=10) as r:
                d = json.loads(r.read().decode("utf-8"))
                mid = d["data"][0]["id"]
                return mid
        except Exception as e:
            if not quiet:
                log(f"Lane {endpoint} not ready ({e}); retrying in 15s...")
            time.sleep(15)
    return "glm-5.3"


def run_one_slot(
    cli: DraftSiteClient,
    claim: dict[str, Any],
    endpoint: str,
    model_id: str,
    concurrency: int,
    engine_tag: str,
    dspine_progress: Path | None,
    export_path: Path | None,
    cache_dir: Path,
) -> dict[str, Any]:
    plan = claim["plan"]
    slot_id = str(claim["slot_id"])
    inp = next(x for x in claim["inputs"] if x["name"] in ("prompt_shard", "prompts"))

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{slot_id}.cache.jsonl"

    with tempfile.TemporaryDirectory(prefix="drs_lbl_") as td:
        td_p = Path(td)
        pf = cli.download_artifact(inp["url"], td_p / "prompts.jsonl", inp.get("sha256"))
        prompts = [json.loads(ln) for ln in pf.read_text(encoding="utf-8").splitlines() if ln.strip()]

        # Load any previously persisted prompt results for this slot (zero lost work on restart/window)
        cached: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        if cache_file.is_file():
            for ln in cache_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not ln.strip():
                    continue
                try:
                    rec = json.loads(ln)
                    if "result" in rec and "export" in rec and rec["result"].get("id"):
                        cached[str(rec["result"]["id"])] = (rec["result"], rec["export"])
                except Exception:
                    pass

        log(
            f"Slot {slot_id}: loaded {len(prompts)} prompts (cached={len(cached)}, C={concurrency}, endpoint={endpoint})"
        )

        t0 = time.time()
        results: list[dict[str, Any] | None] = [None] * len(prompts)
        export_recs: list[dict[str, Any] | None] = [None] * len(prompts)
        done_cnt = 0
        gen_toks = 0
        session_toks = 0
        paused_s = 0.0
        lock = threading.Lock()
        cache_lock = threading.Lock()
        pause_gate = threading.Lock()

        # Pre-populate from disk cache
        todo: list[tuple[int, dict[str, Any]]] = []
        for i, pr in enumerate(prompts):
            pid = str(pr["id"])
            if pid in cached:
                res_r, exp_r = cached[pid]
                results[i] = res_r
                export_recs[i] = exp_r
                done_cnt += 1
                gen_toks += len(res_r.get("gen_ids") or [])
            else:
                todo.append((i, pr))

        def maybe_wait_window() -> None:
            nonlocal paused_s
            if not (dspine_progress and endpoint_is_a14(endpoint)):
                return
            if not a14_window_is_open(dspine_progress):
                return
            with pause_gate:
                if not a14_window_is_open(dspine_progress):
                    return
                t_p0 = time.time()
                wait_for_lane_ready(endpoint, dspine_progress, quiet=False)
                dt_p = time.time() - t_p0
                with lock:
                    paused_s += dt_p
                if dt_p > 5.0:
                    log(f"Slot {slot_id}: resumed after {dt_p:.1f}s lane window pause")

        def do_prompt(idx: int, pr: dict[str, Any]) -> None:
            nonlocal done_cnt, gen_toks, session_toks
            pid = str(pr["id"])
            pids = pr["prompt_ids"]
            psha = pr.get("prompt_sha256") or hashlib.sha256(json.dumps(pids, separators=(",", ":")).encode()).hexdigest()
            max_tok = int(
                pr.get("max_tokens")
                or (plan.get("max_tokens_long", 1024) if len(pids) > 4096 else plan.get("max_tokens_short", 1536))
            )

            body = json.dumps(
                {
                    "model": model_id,
                    "prompt": pids,
                    "max_tokens": max_tok,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "return_token_ids": True,
                    "skip_special_tokens": False,
                }
            ).encode("utf-8")

            resp = None
            last_err: Exception | None = None
            real_failures = 0
            max_real_failures = 24
            t_req = time.time()

            while real_failures < max_real_failures:
                if STOP:
                    raise RuntimeError("worker stopping")
                # Window wait NEVER counts against real_failures
                maybe_wait_window()
                if STOP:
                    raise RuntimeError("worker stopping")
                try:
                    t_req = time.time()
                    req = urllib.request.Request(
                        f"{endpoint}/v1/completions",
                        data=body,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=600) as r:
                        resp = json.loads(r.read().decode("utf-8"))
                    break
                except Exception as e:
                    last_err = e
                    # If a lane window just opened or the lane is restarting, wait without burning retry budget
                    if dspine_progress and endpoint_is_a14(endpoint) and a14_window_is_open(dspine_progress):
                        maybe_wait_window()
                        continue
                    is_conn_or_5xx = isinstance(e, (urllib.error.URLError, ConnectionError, TimeoutError, OSError)) or (
                        isinstance(e, urllib.error.HTTPError) and e.code in (429, 500, 502, 503, 504)
                    )
                    if is_conn_or_5xx:
                        # Check if lane is actually down/restarting
                        wait_for_lane_ready(endpoint, dspine_progress, quiet=True)
                    real_failures += 1
                    if real_failures in (1, 5, 10, 20):
                        log(f"  {slot_id} prompt {pid} transient err ({real_failures}/{max_real_failures}): {e}")
                    time.sleep(min(30, 2 * real_failures))

            if resp is None:
                raise RuntimeError(f"failed prompt {pid} after {real_failures} retries (last_err={last_err})")

            dt = round(time.time() - t_req, 3)
            ch = resp["choices"][0]
            gids = ch.get("token_ids") or []
            if not gids:
                gids = [154829]  # EOS fallback if empty stop
            gsha = hashlib.sha256(json.dumps(gids, separators=(",", ":")).encode("utf-8")).hexdigest()
            fr = str(ch.get("finish_reason") or "stop")

            res_obj = {
                "id": pid,
                "prompt_sha256": psha,
                "gen_ids": gids,
                "finish_reason": fr,
                "engine_tag": engine_tag,
                "gen_sha256": gsha,
            }
            exp_obj = {
                "id": pid,
                "category": pr.get("category", "code"),
                "source": pr.get("source", "permissive"),
                "model_id": model_id,
                "prompt_len": len(pids),
                "gen_len": len(gids),
                "total_len": len(pids) + len(gids),
                "elapsed_s": dt,
                "finish_reason": fr,
                "usage": resp.get("usage"),
                "gen_sha256": gsha,
                "prompt_ids": pids,
                "gen_ids": gids,
            }
            results[idx] = res_obj
            export_recs[idx] = exp_obj

            # Persist immediately so this prompt is never lost even if slot is interrupted
            with cache_lock:
                with cache_file.open("a", encoding="utf-8") as cf:
                    cf.write(json.dumps({"result": res_obj, "export": exp_obj}, separators=(",", ":")) + "\n")
                    cf.flush()

            with lock:
                done_cnt += 1
                gen_toks += len(gids)
                session_toks += len(gids)
                if done_cnt % 25 == 0 or done_cnt == len(prompts):
                    active_s = max(time.time() - t0 - paused_s, 1e-3)
                    rate = session_toks / active_s
                    log(
                        f"  {slot_id}: {done_cnt}/{len(prompts)} prompts ({gen_toks} tok, active_agg={rate:.1f} tok/s, paused={paused_s:.0f}s)"
                    )

        if todo:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = [ex.submit(do_prompt, i, pr) for i, pr in todo]
                # Collect all futures even if one errors, so in-flight completions still persist to disk cache!
                first_err: Exception | None = None
                for fut in concurrent.futures.as_completed(futs):
                    try:
                        fut.result()
                    except Exception as e:
                        if first_err is None:
                            first_err = e
                if first_err is not None:
                    raise first_err

        out_gz = td_p / f"{slot_id}.labels.jsonl.gz"
        raw_jsonl = "\n".join(json.dumps(r, separators=(",", ":")) for r in results if r is not None) + "\n"
        out_gz.write_bytes(gzip.compress(raw_jsonl.encode("utf-8"), compresslevel=6))

        elapsed = round(time.time() - t0, 2)
        active_elapsed = round(max(elapsed - paused_s, 1e-3), 2)
        agg_tps = round(session_toks / max(active_elapsed, 1e-3), 2)
        resp = cli.upload_artifact(
            claim,
            out_gz,
            manifest={
                "slot_id": slot_id,
                "n_rows": len(prompts),
                "total_gen_tokens": gen_toks,
                "concurrency": concurrency,
                "agg_tok_s": agg_tps,
                "elapsed_s": elapsed,
                "active_elapsed_s": active_elapsed,
            },
        )
        log(
            f"Slot {slot_id} VERIFIED ({out_gz.stat().st_size} B gz, {gen_toks} tok in {active_elapsed}s active / {elapsed}s wall = {agg_tps} tok/s) -> {resp.get('state')}"
        )

        # Export exact DSpine-compatible schema rows for downstream DSpine ingestion
        if export_path and resp.get("state") == "verified":
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with export_path.open("a", encoding="utf-8") as ef:
                for rec in export_recs:
                    if rec is not None:
                        ef.write(json.dumps(rec, separators=(",", ":")) + "\n")
                ef.flush()
            try:
                cache_file.unlink(missing_ok=True)
            except Exception:
                pass
        return resp


def main() -> None:
    ap = argparse.ArgumentParser(description="Persistent C8 drafter-label fleet worker")
    ap.add_argument("--url", default="https://draft.bertholomus.com")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--login", required=True)
    ap.add_argument("--job", default="glm53-dspine-labels-prod")
    ap.add_argument("--runner", default="drs-worker-v1.1.0")
    ap.add_argument("--endpoint", default="http://127.0.0.1:18090")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--engine-tag", default="gr8-glm53-c8")
    ap.add_argument("--dspine-progress", default="/tank/projects/glm53-dspine/PROGRESS.md")
    ap.add_argument("--export-out", default="/tank/projects/draft-site/data/exports/drs_labels.jsonl")
    ap.add_argument("--cache-dir", default="/tank/projects/draft-site/data/worker_cache")
    ap.add_argument("--max-slots", type=int, default=0, help="0 = run forever until stopped or no open slots")
    ap.add_argument("--poll-interval", type=int, default=30)
    args = ap.parse_args()

    keys = json.loads(Path(args.key_file).read_text(encoding="utf-8"))
    cli = DraftSiteClient(args.url, keys[args.login])
    me = cli.me()
    log(f"Authenticated as {me['login']} (verified={me['verified_slots']}) job={args.job} C={args.concurrency}")

    prog_p = Path(args.dspine_progress) if args.dspine_progress else None
    exp_p = Path(args.export_out) if args.export_out else None
    cache_p = Path(args.cache_dir)
    completed = 0

    while not STOP:
        model_id = wait_for_lane_ready(args.endpoint.rstrip("/"), prog_p)
        if STOP:
            break
        try:
            claim = cli.claim(runner=args.runner, job=args.job, host_tag=f"{socket.gethostname()}-c{args.concurrency}")
        except DraftSiteError as e:
            if e.status == 409:
                log(f"No open slots on {args.job}; sleeping {args.poll_interval}s...")
                time.sleep(args.poll_interval)
                continue
            log(f"Claim error ({e.status}): {e.body}; sleeping 20s...")
            time.sleep(20)
            continue

        try:
            run_one_slot(
                cli,
                claim,
                args.endpoint.rstrip("/"),
                model_id,
                args.concurrency,
                args.engine_tag,
                prog_p,
                exp_p,
                cache_p,
            )
            completed += 1
            if args.max_slots > 0 and completed >= args.max_slots:
                log(f"Reached max_slots={args.max_slots}; exiting cleanly.")
                break
        except Exception as e:
            log(f"Slot {claim['slot_id']} error: {e}; releasing claim (cached progress preserved on disk)...")
            try:
                cli.release(claim["job"], claim["slot_id"])
            except Exception:
                pass
            time.sleep(15)


if __name__ == "__main__":
    main()
