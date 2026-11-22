"""Configuration and security limits for draft.bertholomus.com."""
from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DRAFT_DATA_DIR", "/data")).resolve()
SECRETS_DIR = Path(os.environ.get("DRAFT_SECRETS_DIR", "/secrets")).resolve()
DB_PATH = DATA_DIR / "db" / "draft_site.sqlite3"
INPUTS_DIR = DATA_DIR / "inputs"
STAGING_DIR = DATA_DIR / "uploads" / "staging"
VERIFIED_DIR = DATA_DIR / "uploads" / "verified"
QUARANTINE_DIR = DATA_DIR / "uploads" / "quarantine"
EVALS_DIR = DATA_DIR / "evals"

PUBLIC_URL = os.environ.get("DRAFT_PUBLIC_URL", "https://draft.bertholomus.com").rstrip("/")
ORIGIN_HOST = os.environ.get("DRAFT_ORIGIN_HOST", "draft-origin.albertlee.co").lower()
REQUIRE_ORIGIN_AUTH = os.environ.get("DRAFT_REQUIRE_ORIGIN_AUTH", "1") == "1"

# Strict upload limits (Guardrail 5)
MAX_DIRECT_UPLOAD_BYTES = 12 * 1024 * 1024  # 12 MiB
MAX_CHUNK_BYTES = 8 * 1024 * 1024           # 8 MiB per chunk (under both Hostinger PHP & CF 100 MiB limits)
MAX_DECOMPRESSED_LABEL_BYTES = 64 * 1024 * 1024  # zip-bomb guard for gzipped JSONL
MAX_UPLOAD_BYTES_BY_KIND = {
    "drafter-eval": 2 * 1024 * 1024,          # 2 MiB (<50 KB expected)
    "drafter-label": 32 * 1024 * 1024,        # 32 MiB (~1.2 MB gz expected)
    "drafter-capture": 256 * 1024 * 1024,     # 256 MiB for pilot 4-tap INT8 shard pack
    "drafter-train": 2_800 * 1024 * 1024,     # 2.8 GB (59 MB S1 / 548 MB SparseLoCo / 2.44 GB INT8 dense)
}

DEFAULT_CLAIM_TTL_SEC = 6 * 3600  # 6 hours per PIPELINE.md
VOCAB_SIZE = 154_880

# Verification thresholds from CDC PIPELINE.md §5.1 (measured in Pilot 2 & 3)
LABEL_MIN_TOP1_AGREE = 0.920
LABEL_MIN_TOP5_AGREE = 0.995
EVAL_DUP_MAX_TAU_DIFF = 0.005
TRAIN_MAX_SELFCHECK_ERR = 1e-6
TRAIN_MIN_COS_VS_ROOT = 0.05
TRAIN_MAX_DELTA_CE8 = -0.005
TRAIN_NORM_RATIO_MIN = 0.30
TRAIN_NORM_RATIO_MAX = 2.50

def load_origin_auth_secret() -> str:
    p = SECRETS_DIR / "origin_auth.secret"
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    return os.environ.get("DRAFT_ORIGIN_AUTH_SECRET", "").strip()

def load_bootstrap_admin_key() -> str:
    p = SECRETS_DIR / "admin_api_key.txt"
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    return os.environ.get("DRAFT_ADMIN_API_KEY", "").strip()

def load_bootstrap_worker_keys() -> dict[str, str]:
    import json
    p = SECRETS_DIR / "pilot_worker_keys.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}
