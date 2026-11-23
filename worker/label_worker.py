#!/usr/bin/env python3
"""Standalone external drafter-label contributor worker for draft.bertholomus.com.

Protocol:
- Claims open drafter-label shards from https://draft.bertholomus.com.
- Sends pre-tokenized prompt_ids to an OpenAI-compatible /v1/completions endpoint
  with temperature=0.0, top_p=1.0, return_token_ids=true, skip_special_tokens=false.
- Token ID extraction (FAIL CLOSED — zero silent fallbacks):
    Mode A (preferred): exact integer choices[0].token_ids returned by the engine.
    Mode B (fallback):  ONLY if --tokenizer-dir is explicitly supplied AND its
                        tokenizer.json matches the pinned GLM-5.3 SHA-256.
  If neither yields valid in-range integer token IDs, the worker fails hard and
  releases its claim cleanly.
- Records full provenance per row: engine_tag, engine_image_digest, engine_signature,
  quant_profile, concurrency, token_id_source, and model_served.
- Persists completed prompts to a local per-slot disk cache keyed by (slot_id, shard_sha, model_id).
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
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import DraftSiteClient, DraftSiteError, sha256_file  # noqa: E402

RUNNER_DEFAULT = "drs-label-ext-v1.0.0"
VOCAB_SIZE = 154_880
PINNED_GLM53_TOKENIZER_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
STOP = False


def _on_sig(signum, frame):
    global STOP
    STOP = True


signal.signal(signal.SIGTERM, _on_sig)
signal.signal(signal.SIGINT, _on_sig)


def log(msg: str) -> None:
    print(f"[label-worker {time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def load_secret_from_file_or_env(key_file: str | None, env_var: str) -> str | None:
    if key_file:
        val = Path(key_file).read_text(encoding="utf-8").strip()
        if not val:
            raise ValueError(f"key file is empty: {key_file}")
        return val
    val = os.environ.get(env_var, "").strip()
    return val or None


class PinnedTokenizer:
    """Hash-verified GLM-5.3 tokenizer wrapper (Mode B fallback only)."""

    def __init__(self, tokenizer_dir: Path, expected_sha256: str = PINNED_GLM53_TOKENIZER_SHA256):
        tok_json = Path(tokenizer_dir) / "tokenizer.json"
        if not tok_json.is_file():
            raise RuntimeError(f"pinned tokenizer missing tokenizer.json in {tokenizer_dir}")
        actual_sha = sha256_file(tok_json)
        if actual_sha != expected_sha256.lower():
            raise RuntimeError(
                f"tokenizer.json SHA-256 mismatch: got {actual_sha}, expected pinned {expected_sha256}"
            )
        self.sha256 = actual_sha
        self._mode: str
        self._tok: Any
        try:
            from tokenizers import Tokenizer

            self._tok = Tokenizer.from_file(str(tok_json))
            self._mode = "tokenizers"
        except ImportError:
            try:
                from transformers import AutoTokenizer

                self._tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
                self._mode = "transformers"
            except ImportError as e:
                raise RuntimeError(
                    "Mode B tokenizer fallback requires either 'tokenizers' or 'transformers' installed"
                ) from e

    def encode_no_special(self, text: str) -> list[int]:
        if self._mode == "tokenizers":
            ids = self._tok.encode(text, add_special_tokens=False).ids
        else:
            ids = self._tok.encode(text, add_special_tokens=False)
        return [int(t) for t in ids]


def validate_token_ids(gids: Any, context: str) -> list[int]:
    if not isinstance(gids, list) or len(gids) == 0:
        raise RuntimeError(f"missing or empty token_ids ({context}); silent fallbacks are forbidden")
    out: list[int] = []
    for t in gids:
        if type(t) is not int or t < 0 or t >= VOCAB_SIZE:
            raise RuntimeError(f"invalid token id {t!r} ({context})")
        out.append(t)
    return out


def extract_gen_ids(choice: dict[str, Any], tokenizer: PinnedTokenizer | None, prompt_id: str) -> tuple[list[int], str]:
    """Extract exact token IDs from a /v1/completions choice; fail hard if unavailable."""
    raw_ids = choice.get("token_ids")
    if raw_ids is not None:
        return validate_token_ids(raw_ids, f"engine token_ids for prompt {prompt_id}"), "engine"

    if tokenizer is None:
        raise RuntimeError(
            f"engine omitted choices[0].token_ids for prompt {prompt_id} and no --tokenizer-dir was provided; "
            "refusing to fabricate token IDs"
        )
    text = choice.get("text")
    if not isinstance(text, str) or text == "":
        raise RuntimeError(f"engine returned neither token_ids nor non-empty text for prompt {prompt_id}")
    encoded = tokenizer.encode_no_special(text)
    return validate_token_ids(encoded, f"pinned tokenizer encode for prompt {prompt_id}"), "pinned_tokenizer"


def _llm_headers(llm_api_key: str | None) -> dict[str, str]:
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if llm_api_key:
        h["Authorization"] = f"Bearer {llm_api_key}"
    return h


def detect_model_id(endpoint: str, llm_api_key: str | None, timeout: int = 15) -> str:
    req = urllib.request.Request(f"{endpoint}/v1/models", headers=_llm_headers(llm_api_key), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    models = data.get("data") or []
    if not models or not models[0].get("id"):
        raise RuntimeError(f"{endpoint}/v1/models returned no model id")
    return str(models[0]["id"])


def probe_endpoint_capabilities(
    endpoint: str,
    model_id: str,
    llm_api_key: str | None,
    tokenizer: PinnedTokenizer | None,
) -> str:
    """Send a 1-token preflight request before claiming any slot to verify token_ids support."""
    body = json.dumps(
        {
            "model": model_id,
            "prompt": [154822],
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "return_token_ids": True,
            "skip_special_tokens": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}/v1/completions",
        data=body,
        headers=_llm_headers(llm_api_key),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read().decode("utf-8"))
    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError("preflight /v1/completions returned 0 choices")
    _, source = extract_gen_ids(choices[0], tokenizer, "preflight")
    return source


def run_one_slot(
    cli: DraftSiteClient,
    claim: dict[str, Any],
    *,
    endpoint: str,
    llm_api_key: str | None,
    model_id: str,
    concurrency: int,
    engine_tag: str,
    engine_image_digest: str,
    engine_signature: str,
    quant_profile: str,
    tokenizer: PinnedTokenizer | None,
    cache_dir: Path,
    request_timeout: int = 600,
) -> dict[str, Any]:
    plan = claim["plan"]
    slot_id = str(claim["slot_id"])
    inp = next((x for x in claim["inputs"] if x["name"] in ("prompt_shard", "prompts")), None)
    if inp is None:
        raise RuntimeError(f"slot {slot_id} claim missing prompt_shard input")

    shard_sha = str(inp.get("sha256") or "")
    safe_model = re.sub(r"[^a-zA-Z0-9._-]", "_", model_id)[:48]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{slot_id}_{shard_sha[:12]}_{safe_model}.cache.jsonl"

    with tempfile.TemporaryDirectory(prefix="lbl_slot_") as td:
        td_p = Path(td)
        pf = cli.download_artifact(inp.get("artifact_id") or inp["url"], td_p / "prompts.jsonl", shard_sha)
        prompts = [json.loads(ln) for ln in pf.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not prompts:
            raise RuntimeError(f"slot {slot_id} prompt shard is empty")

        cached: dict[str, dict[str, Any]] = {}
        if cache_file.is_file():
            for ln in cache_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not ln.strip():
                    continue
                try:
                    rec = json.loads(ln)
                    if isinstance(rec, dict) and rec.get("id") and rec.get("gen_ids"):
                        cached[str(rec["id"])] = rec
                except Exception:
                    pass

        log(f"Slot {slot_id}: loaded {len(prompts)} prompts (cached={len(cached)}, C={concurrency}, model={model_id})")
        t0 = time.time()
        results: list[dict[str, Any] | None] = [None] * len(prompts)
        done_cnt = 0
        gen_toks = 0
        session_toks = 0
        lock = threading.Lock()
        cache_lock = threading.Lock()

        todo: list[tuple[int, dict[str, Any]]] = []
        for i, pr in enumerate(prompts):
            pid = str(pr["id"])
            if "prompt_ids" not in pr or not isinstance(pr["prompt_ids"], list) or not pr["prompt_ids"]:
                raise RuntimeError(f"prompt '{pid}' missing pre-tokenized prompt_ids")
            if pid in cached:
                results[i] = cached[pid]
                done_cnt += 1
                gen_toks += len(cached[pid]["gen_ids"])
            else:
                todo.append((i, pr))

        def do_prompt(idx: int, pr: dict[str, Any]) -> None:
            nonlocal done_cnt, gen_toks, session_toks
            pid = str(pr["id"])
            pids = pr["prompt_ids"]
            psha = pr.get("prompt_sha256") or hashlib.sha256(
                json.dumps(pids, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            max_tok = int(
                pr.get("max_tokens")
                or (plan.get("max_tokens_long", 1024) if len(pids) > 4096 else plan.get("max_tokens_short", 1536))
            )
            payload = json.dumps(
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
            for attempt in range(1, 13):
                if STOP:
                    raise RuntimeError("worker stopping on signal")
                try:
                    req = urllib.request.Request(
                        f"{endpoint}/v1/completions",
                        data=payload,
                        headers=_llm_headers(llm_api_key),
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=request_timeout) as r:
                        resp = json.loads(r.read().decode("utf-8"))
                    break
                except urllib.error.HTTPError as e:
                    last_err = e
                    if e.code in (400, 401, 403, 404, 422):
                        raise RuntimeError(f"non-retryable HTTP {e.code} on prompt {pid}") from e
                    time.sleep(min(30, 2 * attempt))
                except Exception as e:
                    last_err = e
                    time.sleep(min(30, 2 * attempt))

            if resp is None:
                raise RuntimeError(f"failed prompt {pid} after retries: {last_err}")

            choices = resp.get("choices") or []
            if not choices:
                raise RuntimeError(f"engine returned 0 choices for prompt {pid}")
            ch = choices[0]
            gids, token_source = extract_gen_ids(ch, tokenizer, pid)
            fr = str(ch.get("finish_reason") or "stop")
            gsha = hashlib.sha256(json.dumps(gids, separators=(",", ":")).encode("utf-8")).hexdigest()

            res_obj = {
                "id": pid,
                "prompt_sha256": psha,
                "gen_ids": gids,
                "finish_reason": fr,
                "engine_tag": engine_tag,
                "engine_image_digest": engine_image_digest,
                "engine_signature": engine_signature,
                "quant_profile": quant_profile,
                "concurrency": concurrency,
                "token_id_source": token_source,
                "model_served": str(resp.get("model") or model_id),
                "gen_sha256": gsha,
            }
            results[idx] = res_obj

            with cache_lock:
                with cache_file.open("a", encoding="utf-8") as cf:
                    cf.write(json.dumps(res_obj, separators=(",", ":")) + "\n")
                    cf.flush()

            with lock:
                done_cnt += 1
                gen_toks += len(gids)
                session_toks += len(gids)
                if done_cnt % 25 == 0 or done_cnt == len(prompts):
                    dt = max(time.time() - t0, 1e-3)
                    log(f"  {slot_id}: {done_cnt}/{len(prompts)} prompts ({gen_toks} tok, {session_toks / dt:.1f} tok/s)")

        if todo:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = [ex.submit(do_prompt, i, pr) for i, pr in todo]
                first_err: Exception | None = None
                for fut in concurrent.futures.as_completed(futs):
                    try:
                        fut.result()
                    except Exception as e:
                        if first_err is None:
                            first_err = e
                if first_err is not None:
                    raise first_err

        completed_rows = [r for r in results if r is not None]
        if len(completed_rows) != len(prompts):
            raise RuntimeError(f"incomplete shard: {len(completed_rows)}/{len(prompts)} prompts completed")

        out_gz = td_p / f"{slot_id}.labels.jsonl.gz"
        raw_jsonl = "\n".join(json.dumps(r, separators=(",", ":")) for r in completed_rows) + "\n"
        out_gz.write_bytes(gzip.compress(raw_jsonl.encode("utf-8"), compresslevel=6))

        elapsed = round(time.time() - t0, 2)
        agg_tps = round(session_toks / max(elapsed, 1e-3), 2)
        resp = cli.upload_artifact(
            claim,
            out_gz,
            manifest={
                "slot_id": slot_id,
                "n_rows": len(completed_rows),
                "total_gen_tokens": gen_toks,
                "engine_tag": engine_tag,
                "engine_image_digest": engine_image_digest,
                "engine_signature": engine_signature,
                "quant_profile": quant_profile,
                "concurrency": concurrency,
                "agg_tok_s": agg_tps,
                "elapsed_s": elapsed,
            },
        )
        log(
            f"Slot {slot_id} uploaded ({out_gz.stat().st_size} B gz, {gen_toks} tok in {elapsed}s = {agg_tps} tok/s) "
            f"-> state={resp.get('state')} verify_status={resp.get('verify_status')}"
        )
        if resp.get("accepted"):
            cache_file.unlink(missing_ok=True)
        return resp


def main() -> None:
    ap = argparse.ArgumentParser(description="Standalone external GLM-5.3 label contributor worker")
    ap.add_argument("--url", default="https://draft.bertholomus.com", help="Coordinator base URL")
    ap.add_argument("--key-file", "--api-key-file", dest="key_file", default=None, help="Path to single-line API key file (or set DRAFT_API_KEY)")
    ap.add_argument("--llm-endpoint", required=True, help="OpenAI-compatible /v1/completions base URL (e.g. http://127.0.0.1:8100)")
    ap.add_argument("--llm-api-key-file", default=None, help="Optional single-line LLM bearer key file (or set LLM_API_KEY)")
    ap.add_argument("--llm-model", default=None, help="Model ID (auto-detected from /v1/models if omitted)")
    ap.add_argument("--engine-tag", required=True, help="Engine label (e.g. kindling-gr8-bit280 or kindling-tessera-fp8)")
    ap.add_argument("--engine-image-digest", required=True, help="Container digest (sha256:...) or build identifier recorded per row")
    ap.add_argument("--engine-signature", default="unknown", help="Engine signature / commit / runtime configuration string")
    ap.add_argument("--quant-profile", required=True, help="Quantization profile (e.g. bit280, bf16, fp8)")
    ap.add_argument("--concurrency", type=int, default=4, help="Concurrent completion requests (default 4)")
    ap.add_argument("--cache-dir", default="./label_cache", help="Local per-slot resume cache directory")
    ap.add_argument("--tokenizer-dir", default=None, help="Optional GLM-5.3 tokenizer dir (required ONLY if engine omits token_ids; SHA-256 verified)")
    ap.add_argument("--job", default="glm53-dspine-labels-prod", help="Target drafter-label job")
    ap.add_argument("--slot", default=None, help="Optional specific slot_id to claim")
    ap.add_argument("--runner", default=RUNNER_DEFAULT, help="Registered runner ID")
    ap.add_argument("--max-slots", type=int, default=0, help="Stop after N completed slots (0 = run until stopped)")
    ap.add_argument("--poll-interval", type=int, default=30, help="Seconds to sleep when no slots are open")
    ap.add_argument("--skip-preflight", action="store_true", help="Skip 1-token endpoint capability probe")
    args = ap.parse_args()

    if args.concurrency <= 0 or args.concurrency > 64:
        raise SystemExit("--concurrency must be between 1 and 64")

    api_key = load_secret_from_file_or_env(args.key_file, "DRAFT_API_KEY")
    if not api_key:
        raise SystemExit("API key required via --key-file or DRAFT_API_KEY environment variable")
    llm_api_key = load_secret_from_file_or_env(args.llm_api_key_file, "LLM_API_KEY")

    tokenizer: PinnedTokenizer | None = None
    if args.tokenizer_dir:
        tokenizer = PinnedTokenizer(Path(args.tokenizer_dir))
        log(f"Verified pinned GLM-5.3 tokenizer.json (sha256={tokenizer.sha256[:16]}...)")

    endpoint = args.llm_endpoint.rstrip("/")
    model_id = args.llm_model or detect_model_id(endpoint, llm_api_key)
    log(f"Using LLM endpoint {endpoint} model={model_id}")

    if not args.skip_preflight:
        src = probe_endpoint_capabilities(endpoint, model_id, llm_api_key, tokenizer)
        log(f"Preflight OK (token_id_source={src})")

    cli = DraftSiteClient(args.url, api_key)
    me = cli.me()
    log(
        f"Authenticated as {me['login']} (verified={me['verified_slots']}, "
        f"allowed_kinds={me.get('allowed_kinds')}, claim_cap={me.get('claim_cap')})"
    )

    cache_p = Path(args.cache_dir)
    completed = 0
    while not STOP:
        try:
            claim = cli.claim(runner=args.runner, job=args.job, slot_id=args.slot)
        except DraftSiteError as e:
            if e.status == 409:
                if args.slot:
                    raise SystemExit(f"requested slot {args.slot} not open: {e.body}")
                log(f"No open slots on {args.job}; sleeping {args.poll_interval}s...")
                time.sleep(args.poll_interval)
                continue
            log(f"Claim error (HTTP {e.status}): {e.body}")
            if e.status in (401, 403):
                raise SystemExit(1)
            time.sleep(20)
            continue

        try:
            run_one_slot(
                cli,
                claim,
                endpoint=endpoint,
                llm_api_key=llm_api_key,
                model_id=model_id,
                concurrency=args.concurrency,
                engine_tag=args.engine_tag,
                engine_image_digest=args.engine_image_digest,
                engine_signature=args.engine_signature,
                quant_profile=args.quant_profile,
                tokenizer=tokenizer,
                cache_dir=cache_p,
            )
            completed += 1
            if args.max_slots > 0 and completed >= args.max_slots:
                log(f"Completed max_slots={args.max_slots}; exiting cleanly.")
                break
            if args.slot:
                break
        except Exception as e:
            log(f"Slot {claim['slot_id']} failed: {e}; releasing claim...")
            try:
                cli.release(claim["job"], claim["slot_id"])
            except Exception:
                pass
            if args.slot or args.max_slots == 1:
                raise SystemExit(1)
            time.sleep(15)


if __name__ == "__main__":
    main()
