# External Label Contributor Guide (`draft.bertholomus.com`)

This guide covers running the standalone `drafter-label` worker for `glm53-dspine-labels-prod`. External contributor accounts are scoped strictly to **`drafter-label`** slots on public permissive prompts (Apache-2.0 / MIT / PSF).

## 1. Target Model & Pinned Tokenizer
- **Base model:** `zai-org/GLM-5.3` (154,880 vocab)
- **Reference quant release:** `kindlingai/glm-5.3-exlr8-k2-k3-k4` at commit `9e25c28a2e8808d8b2599d780521fd6af24e7558` (recommended profile: `bit280`; unquantized BF16 / FP8 / other `exlr8` profiles `bit200..bit400` are also eligible pending the canary check)
- **Pinned `tokenizer.json` SHA-256:** `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` (required only if your serving engine does not return `choices[0].token_ids`)
- **Runner ID:** `drs-label-ext-v1.0.0`

## 2. Inference Protocol & Row Schema
Input shards contain pre-tokenized, already chat-templated `prompt_ids` (no raw chat text). Do **not** call `/v1/chat/completions`. Send each row to `/v1/completions`:
```json
{
  "model": "<model-id>",
  "prompt": [154822, 154824, ...],
  "max_tokens": 1536,
  "temperature": 0.0,
  "top_p": 1.0,
  "return_token_ids": true,
  "skip_special_tokens": false
}
```
- **Mode A (preferred — `gr8`, vLLM, SGLang):** Use exact integer `choices[0].token_ids` directly.
- **Mode B (text-only engines — e.g. `tessera-serve` without `return_token_ids`):** Pass `--tokenizer-dir <dir>` containing the pinned `tokenizer.json` above. The worker verifies its SHA-256 on startup and encodes `choices[0].text` with `add_special_tokens=False`.
- **Fail closed:** The worker has **zero silent fallbacks** (no UTF-8 byte fallback, no fabricated `[1]` / EOS tokens). Missing or empty token IDs abort the slot and release the claim cleanly.

Each line of the uploaded `.labels.jsonl.gz` records:
```json
{
  "id": "tr4_prose_...",
  "prompt_sha256": "<64-hex>",
  "gen_ids": [1234, ...],
  "finish_reason": "stop",
  "engine_tag": "kindling-gr8-bit280",
  "engine_image_digest": "ghcr.io/kindlingai/gr8@sha256:...",
  "engine_signature": "gr8-sha-...",
  "quant_profile": "bit280",
  "concurrency": 4,
  "token_id_source": "engine",
  "model_served": "glm-5.3",
  "gen_sha256": "<sha256 of compact JSON gen_ids>"
}
```

## 3. Quick Start (2 Files, Stdlib Only for Mode A)
1. **Create your own worker API key:** Open https://draft.bertholomus.com, click **Log in with GitHub** (read-only identity; new accounts wait for owner approval), open the **API Keys** tab, and click **Create API Key** (full `drs_wrk_...` key is shown once; up to 2 live keys per account; rotate or revoke anytime).
2. Download `worker/client.py` and `worker/label_worker.py` (Python 3.10+; install `tokenizers` only if using Mode B), save your key with mode `0600` at `~/.config/draft-site/key`, and run **1 canary slot first**:
```bash
mkdir -p ~/.config/draft-site
printf '%s\n' 'drs_wrk_PASTE_YOUR_KEY_HERE' > ~/.config/draft-site/key
chmod 600 ~/.config/draft-site/key
python3 worker/label_worker.py \
  --url https://draft.bertholomus.com \
  --key-file ~/.config/draft-site/key \
  --llm-endpoint http://127.0.0.1:8100 \
  --engine-tag kindling-gr8-bit280 \
  --engine-image-digest ghcr.io/kindlingai/gr8@sha256:<digest> \
  --engine-signature <build-commit-or-signature> \
  --quant-profile bit280 \
  --concurrency 4 \
  --cache-dir ./label_cache \
  --max-slots 1
```
Pass `--tokenizer-dir /path/to/glm53-tokenizer` only if your engine omits `token_ids`. Completed prompts are persisted per slot in `--cache-dir` so interruptions lose zero work; `SIGTERM`/`SIGINT` cleanly releases the claim. After your canary slot passes owner verification, omit `--max-slots 1` (or set `--max-slots 0`) to run continuously.

## 4. Verification, Quotas & Downstream Use
- **Owner-side teacher-forced gate:** Accepted uploads stay at `state="uploaded"` (`pending_label_gate`) until an owner gate probes an unguessable HMAC-sampled `8 prompts × 10 offsets = 80 tokens` on the reference GLM-5.3 lane.
  - **Per-slot pass bar:** `top1 >= 0.850` (`>=68/80`), `top5 >= 0.960` (`>=77/80`), and every sampled prompt `>= 0.50` (`>=5/10`).
  - **New-engine canary bar:** `top1 >= 0.920` and `top5 >= 0.990` on the first 200-prompt shard before bulk runs.
- **Account controls:** Non-owner accounts may claim `drafter-label` only, hold at most **2 active claims**, hold up to **2 live API keys** (self-service create/rotate/revoke via GitHub login), download only input shards bound to their active claims, and are **auto-quarantined at 3 strikes**.
- **Training note:** External labels are ingested downstream with teacher-forced target relabeling and prefix masking (`L2`/`L4` regime) so minor cross-engine greedy drift does not harm drafter quality.
