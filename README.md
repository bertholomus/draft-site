# draft.bertholomus.com — GLM-5.3 Drafter Label Coordinator

Hardened coordination server and external label worker for GLM-5.3 DSpine B=16 drafter data generation, modelled on [Kindling AI's exlr8 quant site](https://quant.kindlingai.dev/).

## Start here

- **Want to contribute labels?** Read [CONTRIBUTING.md](./CONTRIBUTING.md): what the site is, what you need, GitHub sign-in & self-service API keys, setup, first test run, checks, rules and troubleshooting.
- **Using an AI coding agent?** Point it at [AGENTS.md](./AGENTS.md): the same procedure as exact steps, checks and hard rules.
- **Protocol reference:** [LABELS.md](./LABELS.md): request format, row schema, pass bars.

## Trust Model & Scope
- **GitHub OAuth & self-service keys:** Contributors click **Log in with GitHub** (read-only identity, `allow_signup=false`, allow-list gated) and manage up to **2 live worker API keys** (`drs_wrk_...`, shown once) directly in the **API Keys** panel (create, rotate, revoke, CSRF-protected). Unknown GitHub accounts land on a waiting-for-approval page with zero keys or claims until approved by an owner.
- **External contributors (`drafter-label` only):** Non-owner accounts can claim only `drafter-label` slots, subject to tier checks, a per-login active claim cap (`2`), claim-bound artifact downloads, and automatic quarantine at `3` strikes.
- **Owner-side label verification:** Every non-trusted label upload passes Phase-A structural & SHA-256 verification in an isolated bounded worker subprocess (`server/verify_worker.py`) and waits at `state="uploaded"` (`verify_status="pending_label_gate"`). An owner-side daemon (`worker/label_verify_gate.py`) then teacher-forces an unguessable HMAC-sampled set of `8 × 10 = 80` token probes against the reference GLM-5.3 lane (`>=85%` top-1, `>=96%` top-5, `>=50%` per prompt; canary target `>=92%` top-1 / `>=99%` top-5) before marking the slot `verified`.
- **Owner-only kinds:** `drafter-train`, `drafter-eval`, `drafter-capture`, and `drafter-pack` are restricted to owner logins.
- **External onboarding:** See [LABELS.md](./LABELS.md) for the standalone external label worker (`worker/label_worker.py` + `worker/client.py`), request schema, provenance requirements, and pinned tokenizer hash.

## Layout
- `server/` — Hardened FastAPI + SQLite (`journal_mode=DELETE`) coordinator, GitHub OAuth & session store, self-service API keys, isolated Phase-A verifier (`verify_worker.py`), and CSP-hardened web UI (`ui.py`, `static/app.js`).
- `worker/label_worker.py` + `worker/client.py` — Clean standalone external label contributor worker (zero owner paths/hosts/IPs; fail-closed token IDs; origin-pinned auth; mandatory download SHA-256).
- `worker/label_verify_gate.py` — Owner-side HMAC-sampled teacher-forced label verification daemon.
- `docker/` — Digest-pinned non-root server container (`Dockerfile`, `compose.yaml`).
- `proxy/` — Reverse proxy scripts for the public frontend.
- `tests/` — Pytest test suite (`tests/test_server.py`, `tests/test_ghlogin.py`, `tests/test_label_gate.py`, `tests/test_worker_and_client.py`) and live red-team verification (`tests/redteam_public.py`).

## Running Tests
```bash
pytest -q tests/
```

## License & Attribution
Licensed under the Apache License, Version 2.0 ([LICENSE](./LICENSE)). See [NOTICE](./NOTICE) for full attributions including Kindling AI (Matt Mastracci `@mmastrac`, Chuck `@ck2084`, Steve `@coffeedev`), Z.ai (GLM-5.3), Tencent (DSpine), and Inco AI (`GLM-5.3-DFlash2@8f02251d`).
