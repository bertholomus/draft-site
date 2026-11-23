# AGENTS.md — operating guide for AI agents

You are an AI agent helping a human contributor label data for **draft.bertholomus.com**. This file is your
complete procedure. Humans: read [CONTRIBUTING.md](./CONTRIBUTING.md) instead (same content, explained).
Protocol reference: [LABELS.md](./LABELS.md).

## Goal

Run `worker/label_worker.py` against the contributor's own local GLM-5.3 server so it claims `drafter-label`
slots on job `glm53-dspine-labels-prod`, generates greedy labels, uploads them, and the uploads reach
`verified`. One slot = 200 pre-tokenized prompts.

## Hard rules (do not break)

1. **Never print, log, commit, or paste the contributor key** (`drs_wrk_...`). Read it only from a file with
   mode `0600`. Never put it in an issue, a commit, a shell history line you show, or a chat message.
2. **Do not modify** `worker/label_worker.py` or `worker/client.py` logic: not prompts, sampling, token
   handling, retry, or upload code. Greedy decoding is required (`temperature 0.0`, `top_p 1.0`). Changing it
   produces rejected uploads and strikes.
3. **Never fabricate or post-process labels.** The engine's exact token ids are the product.
4. **First run is always `--max-slots 1`.** Do not start a continuous run until that slot is `verified`.
   Repeat the one-slot test after ANY change to engine, image, weights, quant profile, or server config.
5. **Stop on any rejection.** Do not retry blindly: 3 rejected uploads quarantine the account.
6. The account can only claim `drafter-label` slots, max **2** active claims. Do not try other job kinds,
   admin endpoints (`/api/admin/*`), or more claims; they fail and look like abuse.
7. Do not expose the contributor's GLM-5.3 server to the internet. The worker reaches it locally.
8. Every HTTP request to the site needs header `X-Quant: 1` (the worker sends it; add it to any manual curl).

## Facts

| Item | Value |
|---|---|
| Site | `https://draft.bertholomus.com` |
| Job | `glm53-dspine-labels-prod` (worker default) |
| Runner id | `drs-label-ext-v1.0.0` (worker default) |
| Model | `zai-org/GLM-5.3`, vocab 154,880 |
| Recommended weights | `kindlingai/glm-5.3-exlr8-k2-k3-k4` @ `9e25c28a2e8808d8b2599d780521fd6af24e7558`, profile `bit280` |
| Engines | gr8 (recommended, default port 8100), vLLM, SGLang: token-id mode. tessera-serve / text-only: tokenizer mode |
| Pinned tokenizer.json SHA-256 | `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` |
| Request | `POST {llm}/v1/completions`, `prompt` = list of token ids, `max_tokens` 1536 (1024 if prompt > 4096 tokens), `temperature 0`, `top_p 1`, `return_token_ids true`, `skip_special_tokens false` |
| Claim lifetime | 6 hours, then the slot returns to the pool |
| Pass bar | 80 secret probes: top-1 >= 68/80, top-5 >= 77/80, each sampled prompt >= 5/10. New engine/quant first slot: top-1 >= 92%, top-5 >= 99% |
| Strikes | each rejected upload = 1; 3 = quarantine |
| Python | 3.10+, stdlib only (plus `tokenizers` in tokenizer mode) |

## Procedure

### Step 1 — Preconditions (check, don't assume)

```bash
python3 --version                                   # >= 3.10
test -f ~/.config/draft-site/key && stat -c '%a' ~/.config/draft-site/key   # expect 600
LLM=http://127.0.0.1:8100                           # ask the human if their server uses another URL
curl -sf $LLM/v1/models                             # must list a GLM-5.3 model id
```

If the key file is missing, ask the human to create it themselves:
`mkdir -p ~/.config/draft-site && printf '%s\n' '<key>' > ~/.config/draft-site/key && chmod 600 ~/.config/draft-site/key`.
Do not ask them to paste the key to you.

### Step 2 — Account check

```bash
curl -s -H "X-Quant: 1" -H "Authorization: Bearer $(cat ~/.config/draft-site/key)" \
  https://draft.bertholomus.com/api/me
```

Proceed only if: `approved: true`, `quarantined: false`, `"drafter-label"` in `allowed_kinds`,
`active_claims < claim_cap`. Report `strikes` to the human. Do not echo the command's Authorization header.

### Step 3 — Engine token-id capability

```bash
curl -s $LLM/v1/completions -H 'Content-Type: application/json' -d '{
  "model":"<model-id>","prompt":[154822],"max_tokens":1,"temperature":0,
  "return_token_ids":true,"skip_special_tokens":false}'
```

- Response has integer `choices[0].token_ids` -> **token-id mode** (no extra flags).
- No `token_ids` -> **tokenizer mode**: obtain GLM-5.3 `tokenizer.json`, verify
  `sha256sum tokenizer.json` equals the pinned hash, `pip install tokenizers`, pass `--tokenizer-dir <dir>`.

### Step 4 — Engine provenance values

| Flag | How to get it |
|---|---|
| `--engine-image-digest` | `docker inspect --format '{{index .RepoDigests 0}}' <image>` -> `...@sha256:...`; if not containerized, a build id |
| `--engine-signature` | engine build commit (gr8: the image's source commit) |
| `--engine-tag` | e.g. `kindling-gr8-bit280` |
| `--quant-profile` | `bit280` / `bf16` / `fp8` / ... matching the loaded weights |

Ask the human when you can't determine a value. Do not invent digests.

### Step 5 — One test slot

```bash
python3 worker/label_worker.py \
  --key-file ~/.config/draft-site/key \
  --llm-endpoint "$LLM" \
  --engine-tag <tag> --engine-image-digest <digest> --engine-signature <sig> \
  --quant-profile <profile> --concurrency 4 --cache-dir ./label_cache \
  --max-slots 1            # + --tokenizer-dir <dir> in tokenizer mode
```

Success log sequence: `Using LLM endpoint ... model=...` -> `Preflight OK (token_id_source=...)` -> claim ->
progress -> upload -> `Completed max_slots=1; exiting cleanly.` Exit code 0.

Expect about 190k-200k generated tokens per slot: 27-64 minutes of active time on a 4-node GB10 gr8 server at
concurrency 8 in our runs; slower hardware takes longer.
Interrupting (SIGINT/SIGTERM) releases the claim; re-running the same command resumes from `--cache-dir`.

### Step 6 — Wait for verification

Poll every 5-10 minutes (not faster):

```bash
curl -s -H "X-Quant: 1" -H "Authorization: Bearer $(cat ~/.config/draft-site/key)" \
  https://draft.bertholomus.com/api/me | python3 -c 'import json,sys;d=json.load(sys.stdin);print({k:d.get(k) for k in ("verified_slots","strikes","active_claims","quarantined")})'
```

- `verified_slots` went up -> pass. Go to step 7.
- `strikes` went up -> rejected. **Stop.** Report to the human with engine tag, quant profile and worker log
  tail. Check weights/quant/greedy settings. Run another single test slot only after a concrete fix.

### Step 7 — Continuous run

Same command without `--max-slots 1`. Prefer a supervised service (see CONTRIBUTING.md section 8 for a
`systemd --user` unit). Keep `--concurrency` within what the server sustains (1-64).

## Error map

| Output | Action |
|---|---|
| `API key required via --key-file or DRAFT_API_KEY` | Fix key path; key file must exist |
| `Claim error (HTTP 401)` / `(HTTP 403)` then exit | Stop. Key invalid/revoked or quarantined. Tell the human to contact Albert |
| `No open slots on ...; sleeping Ns...` | Normal. Leave it running |
| `requested slot ... not open` | Drop `--slot`; let the worker pick |
| `non-retryable HTTP 4xx on prompt ...` | Server rejected a request (often context length). Server must allow prompt + 1,536 tokens |
| `pinned tokenizer missing tokenizer.json` / SHA mismatch | Wrong tokenizer dir/file |
| `--concurrency must be between 1 and 64` | Fix flag |
| `Slot ... failed: ...; releasing claim...` | Read the error; fix; re-run with `--max-slots 1` |
| `missing required header X-Quant: 1` (manual curl) | Add `-H "X-Quant: 1"` |

## Reporting

Report to the human: mode (token-id or tokenizer), engine tag, quant profile, slots completed, current
`verified_slots` and `strikes`, and any error lines. Bugs go to an issue in `bertholomus/draft-site` with log
lines and engine tag, **never the key**.
