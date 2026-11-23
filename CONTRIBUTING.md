# Contributing labels to draft.bertholomus.com

Thank you for helping. This guide explains what the site is, what you give it, and how to run the worker,
from a fresh machine to a steady run. If you use an AI coding agent, point it at [AGENTS.md](./AGENTS.md):
that file is the same procedure written as exact steps and checks for an agent.

Reference material lives in [LABELS.md](./LABELS.md) (protocol, row schema, pass bars). This file is the
walkthrough.

---

## 1. What this is, in plain words

- **DSpine** is a small *drafter* model for **GLM-5.3**. A drafter guesses the next several tokens; the big
  model checks the guesses in one pass. Good guesses make GLM-5.3 generate text faster with **no change to its
  output**.
- A drafter learns from examples of what GLM-5.3 itself writes. We call those examples **labels**.
- Making labels takes GPU time: you run GLM-5.3 on a fixed set of prompts and record its exact output tokens.
  The prompts come only from permissively licensed public text (Apache-2.0 / MIT / PSF).
- **This site hands out that work in pieces.** You claim a piece, your own GLM-5.3 server labels it, the
  worker uploads the result, and the site checks it.
- Everything you upload is checked against our own GLM-5.3 before it is used. Honest labels from a correct
  engine pass. Wrong or made-up labels fail.

Current bulk job: `glm53-dspine-labels-prod`, 386 pieces ("slots") of 200 prompts each (about 77,000 prompts).

## 2. What you need

| Need | Detail |
|---|---|
| A GLM-5.3 server you run | Any OpenAI-compatible server that answers `/v1/completions`. **gr8** is the recommended engine; vLLM and SGLang also work. tessera-serve works in text mode (see 5.3). |
| The right weights | `zai-org/GLM-5.3`. Recommended: `kindlingai/glm-5.3-exlr8-k2-k3-k4` at commit `9e25c28a2e8808d8b2599d780521fd6af24e7558`, profile `bit280`. BF16, FP8 and other exlr8 profiles (`bit200`..`bit400`) are eligible but must pass the new-engine check (section 7). |
| Python | 3.10 or newer. The worker uses only the standard library, unless you need text mode (then `pip install tokenizers`). |
| Two files from this repo | `worker/label_worker.py` and `worker/client.py`. Nothing else in the repo is needed to contribute. |
| A contributor key | One line, starts with `drs_wrk_`. Albert sends it to you privately. It can **only** submit labels. |
| Network | Outbound HTTPS to `draft.bertholomus.com`. Your GLM-5.3 server never needs to be reachable from the internet; the worker talks to it locally. |

## 3. One-time setup

```bash
# 1. Get the worker (repo access comes with your GitHub invite)
git clone https://github.com/bertholomus/draft-site.git
cd draft-site

# 2. Store your key. One line, nothing else, readable only by you.
mkdir -p ~/.config/draft-site
printf '%s\n' 'drs_wrk_PASTE_YOUR_KEY_HERE' > ~/.config/draft-site/key
chmod 600 ~/.config/draft-site/key

# 3. Check your key works (shows your account, claim cap and strikes)
curl -s -H "X-Quant: 1" -H "Authorization: Bearer $(cat ~/.config/draft-site/key)" \
  https://draft.bertholomus.com/api/me
```

Expected from step 3: JSON with `"approved": true`, `"quarantined": false`,
`"allowed_kinds": ["drafter-label"]`, `"claim_cap": 2`, `"strikes": 0`.

**Every request to the site needs the header `X-Quant: 1`.** The worker adds it for you.

## 4. Collect your engine details

The worker records which engine made each label. Gather these once:

```bash
# Your server's base URL (gr8 default port is 8100)
LLM=http://127.0.0.1:8100
curl -s $LLM/v1/models            # must list your GLM-5.3 model id

# Container image digest of your engine (gr8 example)
docker inspect --format '{{index .RepoDigests 0}}' ghcr.io/kindlingai/gr8:<tag>
# -> ghcr.io/kindlingai/gr8@sha256:...

# Engine signature: the gr8 build commit (or any string that pins your build + config)
```

| Worker flag | What to put |
|---|---|
| `--engine-tag` | Short label, e.g. `kindling-gr8-bit280`, `kindling-tessera-fp8` |
| `--engine-image-digest` | The `...@sha256:...` from above, or a build id if you don't run a container |
| `--engine-signature` | gr8 commit / build string |
| `--quant-profile` | `bit280`, `bf16`, `fp8`, ... |

## 5. First run: one test piece

Always start with exactly one slot (`--max-slots 1`). It proves your setup before you spend hours of GPU time.

```bash
python3 worker/label_worker.py \
  --url https://draft.bertholomus.com \
  --key-file ~/.config/draft-site/key \
  --llm-endpoint http://127.0.0.1:8100 \
  --engine-tag kindling-gr8-bit280 \
  --engine-image-digest ghcr.io/kindlingai/gr8@sha256:<digest> \
  --engine-signature <gr8-commit> \
  --quant-profile bit280 \
  --concurrency 4 \
  --cache-dir ./label_cache \
  --max-slots 1
```

What you should see, in order:

1. `Using LLM endpoint ... model=...` (model id auto-detected from `/v1/models`; override with `--llm-model`)
2. `Preflight OK (token_id_source=engine)` – a 1-token test request worked
3. A claim on one slot, then progress through its 200 prompts
4. Upload, then `Completed max_slots=1; exiting cleanly.`

### 5.1 How long a slot takes
A slot is 200 prompts with up to 1,536 new tokens each (1,024 for prompts over 4,096 tokens), about 190,000
to 200,000 generated tokens in total. On our own 4-node GB10 gr8 server at 8 parallel requests, recent slots
took 27 to 64 minutes of active time (48 to 119 tokens/s). Your time depends on your hardware and
`--concurrency` (1-64, default 4).

### 5.2 Interruptions lose nothing
Finished prompts are saved in `--cache-dir`. Ctrl-C or `SIGTERM` releases the claim cleanly. Re-run the same
command to continue. A claim you abandon expires after 6 hours and goes back to the pool.

### 5.3 Text-only engines (tessera-serve and others)
If your server cannot return token ids (`choices[0].token_ids`), download the GLM-5.3 `tokenizer.json`, put it
in a folder, `pip install tokenizers`, and add `--tokenizer-dir /path/to/that/folder`. The worker refuses to
start unless that file's SHA-256 is exactly
`19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`.

## 6. Check your result

Your upload waits at `pending_label_gate` until the owner-side check runs, then turns `verified` or
`rejected`.

```bash
curl -s -H "X-Quant: 1" -H "Authorization: Bearer $(cat ~/.config/draft-site/key)" \
  https://draft.bertholomus.com/api/me        # verified_slots, strikes, active_claims
```

You can also watch the grid at https://draft.bertholomus.com.

## 7. How the check works (and why honest labels always pass)

- The site picks **80 secret positions** in your upload (8 prompts x 10 positions, chosen with a key you
  can't see) and asks our own GLM-5.3 what token it would produce at each one, given your earlier tokens.
- **Pass bar per slot:** at least 68/80 exact top-1 matches, at least 77/80 inside the top 5, and every sampled
  prompt at least 5/10.
- **First slot from a new engine or quant:** stricter, top-1 >= 92% and top-5 >= 99%.
- Small differences between engines and quants are expected; the bar leaves room for them, and training
  re-checks every label against GLM-5.3 anyway.
- Made-up or corrupted labels score near zero. In our own red-team test a fully corrupted upload scored 0/80
  and was rejected and quarantined.

## 8. Run continuously

After your first slot shows `verified`, drop `--max-slots 1` (or use `--max-slots 0`). The worker keeps
claiming slots, sleeping `--poll-interval` seconds (default 30) when none are open. Example `systemd --user`
unit:

```ini
# ~/.config/systemd/user/drs-label.service
[Unit]
Description=draft.bertholomus.com label worker

[Service]
WorkingDirectory=%h/draft-site
ExecStart=/usr/bin/python3 worker/label_worker.py --url https://draft.bertholomus.com \
  --key-file %h/.config/draft-site/key --llm-endpoint http://127.0.0.1:8100 \
  --engine-tag kindling-gr8-bit280 --engine-image-digest ghcr.io/kindlingai/gr8@sha256:<digest> \
  --engine-signature <gr8-commit> --quant-profile bit280 --concurrency 4 --cache-dir %h/draft-site/label_cache
Restart=on-failure
RestartSec=60

[Install]
WantedBy=default.target
```

`systemctl --user daemon-reload && systemctl --user enable --now drs-label`

## 9. Rules

- Your account can claim **`drafter-label` slots only**, at most **2 at a time**.
- Each rejected upload is a **strike**. **3 strikes** quarantine the account until Albert reviews it. Run one
  test slot after any engine, quant or config change to avoid strikes.
- Don't edit the worker to change prompts, sampling or token handling. Greedy decoding (`temperature 0`,
  `top_p 1`) is required.
- Keep your key private. If it leaks, tell Albert; the key can be replaced in one step and the old one stops
  working at once.

## 10. Troubleshooting

| You see | Meaning | Fix |
|---|---|---|
| `API key required via --key-file or DRAFT_API_KEY` | No key found | Check `--key-file` path, or `export DRAFT_API_KEY=...` |
| `Claim error (HTTP 401)` / `(HTTP 403)` and exit | Key wrong, revoked, or account quarantined | Run the `/api/me` check; contact Albert |
| `missing required header X-Quant: 1` | Calling the API by hand without the header | Add `-H "X-Quant: 1"` |
| `No open slots on ...; sleeping` | All slots are claimed or done | Nothing to do; it retries |
| Preflight fails / no `token_ids` | Your engine doesn't return token ids | Use text mode (section 5.3) or turn on `return_token_ids` in your engine |
| `tokenizer.json` SHA-256 mismatch | Wrong tokenizer file | Use the exact GLM-5.3 `tokenizer.json` |
| `non-retryable HTTP 4xx on prompt ...` | Your server refused a request (often context length) | Make sure your server allows the prompt length plus 1,536 new tokens |
| `--concurrency must be between 1 and 64` | Bad flag | Use 1-64 |
| Upload `rejected` | Labels didn't match GLM-5.3 closely enough | Check weights, quant profile, greedy settings; run one test slot; ask us |

## 11. Questions, bugs, ideas

Open an issue in this repo, or message Albert. Include the worker log lines around the problem and your
`--engine-tag`. **Never paste your key into an issue.**

## 12. Credit and license

The code is Apache-2.0 ([LICENSE](./LICENSE)); attributions are in [NOTICE](./NOTICE). The site is modelled on
KindlingAI's exlr8 quant site. Contributors are credited by login on the site.
