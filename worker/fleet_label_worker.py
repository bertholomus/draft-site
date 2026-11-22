#!/usr/bin/env python3
"""Persistent concurrent (C=8) drafter-label fleet worker & verified-label harvester for draft.bertholomus.com.

- Claims production 200-prompt shards from glm53-dspine-labels-prod via https://draft.bertholomus.com
- Generates exact token-ID greedy labels (/v1/completions with prompt_ids + return_token_ids=True) at C=8 (~156 tok/s)
- Respects A1-A4 lane windows (pauses cleanly when DSP WINDOW-START without matching WINDOW-END is active)
- Uploads verified .labels.jsonl.gz via https://draft.bertholomus.com
- Optional --harvest-out exports all verified label shards into exact DSpine train_labels.jsonl schema
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
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
        for line in dspine_progress.read_text(encoding="utf-8", errors="ignore").splitlines()[-120:]:
            if "WINDOW-START" in line:
                open_w = True
            elif "WINDOW-END" in line:
                open_w = False
    except Exception:
        return False
    return open_w


def wait_for_lane_ready(endpoint: str, dspine_progress: Path | None) -> str:
    """Wait until any A1-A4 lane window ends and /v1/models responds 200."""
    while not STOP:
        if dspine_progress and a14_window_is_open(dspine_progress):
            log("A1-A4 lane window active in DSpine PROGRESS; waiting 20s...")
            time.sleep(20)
            continue
        try:
            with urllib.request.urlopen(f"{endpoint}/v1/models", timeout=10) as r:
                d = json.loads(r.read().decode("utf-8"))
                mid = d["data"][0]["id"]
                return mid
        except Exception as e:
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
) -> dict[str, Any]:
    plan = claim["plan"]
    inp = next(x for x in claim["inputs"] if x["name"] in ("prompt_shard", "prompts"))
    with tempfile.TemporaryDirectory(prefix="drs_lbl_") as td:
        td_p = Path(td)
        pf = cli.download_artifact(inp["url"], td_p / "prompts.jsonl", inp.get("sha256"))
        prompts = [json.loads(ln) for ln in pf.read_text(encoding="utf-8").splitlines() if ln.strip()]
        log(f"Slot {claim['slot_id']}: loaded {len(prompts)} prompts (C={concurrency}, endpoint={endpoint})")

        t0 = time.time()
        results: list[dict[str, Any] | None] = [None] * len(prompts)
        export_recs: list[dict[str, Any] | None] = [None] * len(prompts)
        done_cnt = 0
        gen_toks = 0
        lock = threading.Lock()

        def do_prompt(idx: int, pr: dict[str, Any]) -> None:
            nonlocal done_cnt, gen_toks
            pid = str(pr["id"])
            pids = pr["prompt_ids"]
            psha = pr.get("prompt_sha256") or hashlib.sha256(json.dumps(pids, separators=(",", ":")).encode()).hexdigest()
            max_tok = int(pr.get("max_tokens") or (plan.get("max_tokens_long", 1024) if len(pids) > 4096 else plan.get("max_tokens_short", 1536)))

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
            t_req = time.time()
            for attempt in range(12):
                if STOP:
                    raise RuntimeError("worker stopping")
                if dspine_progress and a14_window_is_open(dspine_progress):
                    time.sleep(20)
                    continue
                try:
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
                    time.sleep(min(30, 3 * (attempt + 1)))
            if resp is None:
                raise RuntimeError(f"failed prompt {pid} after retries")
            dt = round(time.time() - t_req, 3)
            ch = resp["choices"][0]
            gids = ch.get("token_ids") or []
            if not gids:
                gids = [154829]  # EOS fallback if empty stop
            gsha = hashlib.sha256(json.dumps(gids, separators=(",", ":")).encode("utf-8")).hexdigest()
            fr = str(ch.get("finish_reason") or "stop")

            results[idx] = {
                "id": pid,
                "prompt_sha256": psha,
                "gen_ids": gids,
                "finish_reason": fr,
                "engine_tag": engine_tag,
                "gen_sha256": gsha,
            }
            export_recs[idx] = {
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
            with lock:
                done_cnt += 1
                gen_toks += len(gids)
                if done_cnt % 25 == 0 or done_cnt == len(prompts):
                    rate = gen_toks / max(time.time() - t0, 1e-3)
                    log(f"  {claim['slot_id']}: {done_cnt}/{len(prompts)} prompts ({gen_toks} tok, agg={rate:.1f} tok/s)")

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(do_prompt, i, pr) for i, pr in enumerate(prompts)]
            for fut in futs:
                fut.result()

        out_gz = td_p / f"{claim['slot_id']}.labels.jsonl.gz"
        raw_jsonl = "\n".join(json.dumps(r, separators=(",", ":")) for r in results if r is not None) + "\n"
        out_gz.write_bytes(gzip.compress(raw_jsonl.encode("utf-8"), compresslevel=6))

        elapsed = round(time.time() - t0, 2)
        agg_tps = round(gen_toks / max(elapsed, 1e-3), 2)
        resp = cli.upload_artifact(
            claim,
            out_gz,
            manifest={
                "slot_id": claim["slot_id"],
                "n_rows": len(prompts),
                "total_gen_tokens": gen_toks,
                "concurrency": concurrency,
                "agg_tok_s": agg_tps,
                "elapsed_s": elapsed,
            },
        )
        log(
            f"Slot {claim['slot_id']} VERIFIED ({out_gz.stat().st_size} B gz, {gen_toks} tok in {elapsed}s = {agg_tps} tok/s) -> {resp.get('state')}"
        )

        # Export exact DSpine-compatible schema rows for downstream DSpine ingestion (strictly under draft-site!)
        if export_path and resp.get("state") == "verified":
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with export_path.open("a", encoding="utf-8") as ef:
                for rec in export_recs:
                    if rec is not None:
                        ef.write(json.dumps(rec, separators=(",", ":")) + "\n")
                ef.flush()
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
    ap.add_argument("--max-slots", type=int, default=0, help="0 = run forever until stopped or no open slots")
    ap.add_argument("--poll-interval", type=int, default=30)
    args = ap.parse_args()

    keys = json.loads(Path(args.key_file).read_text(encoding="utf-8"))
    cli = DraftSiteClient(args.url, keys[args.login])
    me = cli.me()
    log(f"Authenticated as {me['login']} (verified={me['verified_slots']}) job={args.job} C={args.concurrency}")

    prog_p = Path(args.dspine_progress) if args.dspine_progress else None
    exp_p = Path(args.export_out) if args.export_out else None
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
            run_one_slot(cli, claim, args.endpoint.rstrip("/"), model_id, args.concurrency, args.engine_tag, prog_p, exp_p)
            completed += 1
            if args.max_slots > 0 and completed >= args.max_slots:
                log(f"Reached max_slots={args.max_slots}; exiting cleanly.")
                break
        except Exception as e:
            log(f"Slot {claim['slot_id']} error: {e}; releasing claim...")
            try:
                cli.release(claim["job"], claim["slot_id"])
            except Exception:
                pass
            time.sleep(15)


if __name__ == "__main__":
    main()
