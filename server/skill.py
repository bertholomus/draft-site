"""Campfire-format /skill.md generator for draft.bertholomus.com drafter-label workers."""
from __future__ import annotations


def render_skill_md(public_url: str) -> str:
    base = public_url.rstrip("/")
    return f"""---
name: drafter-label
description: Subscribe a machine running a GLM-5.3 serving endpoint to volunteer drafter-label work from {base} (Campfire-compatible contract). Check which drafter-label projects have open work, verify the local /v1/completions endpoint, run one test slot, and keep the label worker running while slots are open.
---

# drafter-label: subscribe a serving endpoint to greedy label work

{base} hands out `drafter-label` work in `(job, slot_id)` shards of pre-tokenized, permissive public
prompts (using Kindling AI Campfire's project, slot, claim, and key contract so future cutover to
https://campfire.kindlingai.dev is a redirect). A worker claims one shard at a time, queries a local
OpenAI-compatible `/v1/completions` server with greedy decoding (`temperature: 0`, `top_p: 1`), uploads
a gzipped JSONL of exact token IDs, and waits while nothing is open. Every external upload passes a
structural/SHA check and then an owner-side teacher-forced spot check before turning `verified`.

## What the user must provide

- An API key. A human logs in with GitHub at {base}, waits for an owner to approve the account, and
  makes a key under **API Keys** on the site. You cannot make one. Keep it in a file with mode `600`.
  Send it only to {base}, as the header `Authorization: Bearer KEY` (together with `X-Quant: 1`).
  Never print it.
- A local OpenAI-compatible serving endpoint answering `/v1/completions` for the project's target
  model (`gr8`, vLLM, SGLang, or a text-only server paired with the pinned `tokenizer.json`). Your
  endpoint stays local and never needs to be reachable from the internet.
- Engine provenance flags: `--engine-tag`, `--engine-image-digest`, `--engine-signature`, and
  `--quant-profile`. Ask the user when a value cannot be detected; do not invent digests.

## 1. See what work exists

~~~bash
curl -s -H "X-Quant: 1" "{base}/api/jobs?kind=drafter-label"
~~~

A JSON array with one entry per active `drafter-label` project. The fields that matter:

- `job`, `title`, `kind`: the project's slug, title, and kind (`drafter-label`).
- `open`: slots that can be claimed now. Work is available when `open` is above 0.
- `closed`, `not_ready`: slots that will open later (`closed` = inputs staged, claims paused).
- `uploaded`, `verified`, `pending_verify`: completed slots (`pending_verify` awaits the owner spot check).
- `weights`: `{{hf, revision, id}}`, the target model release the serving endpoint should serve.

Poll this endpoint every 5 to 15 minutes, not faster.

## 2. Check your account and local serving endpoint

~~~bash
curl -s -H "X-Quant: 1" -H "Authorization: Bearer $(cat ~/.config/draft-site/key)" \\
  "{base}/api/me"
~~~

Proceed only when `approved: true`, `quarantined: false`, `"drafter-label"` is in `allowed_kinds`,
and `active_claims < claim_cap`. Do not echo the Authorization header.

Check whether the local serving endpoint returns integer `choices[0].token_ids`:

~~~bash
LLM=http://127.0.0.1:8100
curl -sf "$LLM/v1/models"
curl -s "$LLM/v1/completions" -H 'Content-Type: application/json' -d '{{
  "model":"<model-id>","prompt":[154822],"max_tokens":1,"temperature":0,
  "return_token_ids":true,"skip_special_tokens":false}}'
~~~

- If integer `choices[0].token_ids` is present: run in token-id mode (Python 3.10+ stdlib only).
- If omitted: download the model's `tokenizer.json`, verify its SHA-256 equals the pinned hash in
  `LABELS.md`, install `tokenizers`, and pass `--tokenizer-dir <dir>`.

## 3. Run one test slot, then subscribe

Run one test slot first (`--max-slots 1`) and verify `verified_slots` increases with zero strikes
before starting a continuous run:

~~~bash
python3 worker/label_worker.py \\
  --url {base} \\
  --key-file ~/.config/draft-site/key \\
  --llm-endpoint "$LLM" \\
  --engine-tag kindling-gr8-bit280 \\
  --engine-image-digest "ghcr.io/kindlingai/gr8@sha256:<digest>" \\
  --engine-signature "<engine-commit>" \\
  --quant-profile bit280 \\
  --concurrency 4 \\
  --cache-dir ./label_cache \\
  --max-slots 1
~~~

Once the first slot is `verified`, omit `--max-slots 1` (or set `--max-slots 0`) and run the worker
under a service manager (`systemd --user` or Docker). Completed prompts are cached per slot in
`--cache-dir`, and `SIGINT` / `SIGTERM` releases the active claim cleanly.

## 4. Keep it healthy

- Poll `GET /api/me` every 5 to 10 minutes after a test slot (not faster):
  - `verified_slots` increased -> pass.
  - `strikes` increased -> **stop immediately** and report to the user. Each rejected upload adds 1
    strike; 3 strikes quarantine the account until an owner reviews it.
- Active claim cap: external accounts hold at most 2 active `drafter-label` claims (6-hour TTL).
  List held claims with `GET /api/claims?kind=drafter-label` and release one with
  `DELETE /api/claims/JOB/SLOT_ID` (or all at once with `DELETE /api/claims`), both with
  `X-Quant: 1` and `Authorization: Bearer KEY`.
- "No open slots": normal when all shards are claimed or done; the worker sleeps `--poll-interval`
  seconds and retries.

## Unsubscribe

Stop the worker (`SIGTERM` releases its active claim), or release any remaining claim with
`DELETE /api/claims` as above.

## As a scheduled check

Every 15 minutes: fetch step 1's list (`?kind=drafter-label`). Start the worker if `open > 0` and it
is not running. Report any strike increase or 401/403 exit to the user rather than retrying blindly.
"""
