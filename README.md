# draft.bertholomus.com — Community Speculative Drafter Pilot

Closed-pilot community drafter coordination site modelled on [Kindling AI's exlr8 quant site](https://quant.kindlingai.dev/), implementing the empirical pipeline from the CDC deep dive (`PIPELINE.md`):

- **`drafter-label`** — Same-model GLM-5.3 greedy responses on public permissive prompts (`Apache-2.0`/`MIT`/`PSF`), verified by schema + `prompt_sha256` + downstream teacher-forced agreement (`>=92%` top-1, `>=99.5%` top-5).
- **`drafter-train`** — Fused train-where-you-capture DiLoCo / SparseLoCo / Model Soups (`0 B` tap upload, `59–548 MB` safe zero-pickle `.safetensors` deltas), verified by Phase A safe-parse + Phase B Owner GPU Held-Out Replay (`delta_ce8 < -0.005`) + FLTrust Cosine (`> 0.05`) + L2 Norm Band (`[0.3x, 2.5x]`).
- **`drafter-eval`** — Deterministic cross-profile offline MXFP8 replay (`<50 KB` JSON) with duplicate-slot `pred_sha256` quorum verification.
- **`drafter-capture`** — Optional standalone 4-tap (`L33/L47/L61/L75`) INT8 + zstd safe `.safetensors` archive (`~3.8×` smaller, `0.00%` `tau` loss).

## Layout
- `server/` — FastAPI + SQLite WAL coordinator & web UI (runs non-root in Docker on FrankenNAS).
- `proxy/` — Minimal PHP 8.3 streaming HTTPS reverse proxy + `.htaccess` for Hostinger `draft.bertholomus.com`.
- `worker/` — Standalone volunteer worker CLI (`draft_worker.py`), owner GPU gate & aggregator (`owner_gate.py`), pilot seeder (`seed_pilot.py`), and worker `Dockerfile`.
- `docker/` — Hardened non-root server `Dockerfile` and `compose.yaml`.
- `tests/` — Automated API, auth, chunked upload, readiness gating, and rejection tests.

## Attribution & License
Target release: `kindlingai/glm-5.3-exlr8-k2-k3-k4@9e25c28a` (Kindling AI). Warm-start preserves Apache-2.0 `NOTICE` (incoai DFlash2, Tencent DSpine, Z.ai GLM-5.3). No private traffic; no CC BY-NC-ND tensors.
