"""Configuration and security limits for draft.bertholomus.com."""
from __future__ import annotations

import json
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
ORIGIN_HOST = os.environ.get("DRAFT_ORIGIN_HOST", "").lower()
REQUIRE_ORIGIN_AUTH = os.environ.get("DRAFT_REQUIRE_ORIGIN_AUTH", "1") == "1"
ALLOW_LOOPBACK_NO_ORIGIN_AUTH = os.environ.get("DRAFT_ALLOW_LOOPBACK_NO_ORIGIN_AUTH", "1") == "1"
ADMIN_LOCAL_ONLY = os.environ.get("DRAFT_ADMIN_LOCAL_ONLY", "0") == "1"
BOOTSTRAP_WORKER_KEYS = os.environ.get("DRAFT_BOOTSTRAP_WORKER_KEYS", "0") == "1"

# Strict upload & verification limits (S6)
MAX_DIRECT_UPLOAD_BYTES = 12 * 1024 * 1024  # 12 MiB
MIN_CHUNK_BYTES = int(os.environ.get("DRAFT_MIN_CHUNK_BYTES", str(256 * 1024)))  # 256 KiB floor
MAX_CHUNK_BYTES = 8 * 1024 * 1024  # 8 MiB per chunk
MAX_CHUNKS_PER_SESSION = int(os.environ.get("DRAFT_MAX_CHUNKS_PER_SESSION", "512"))
MAX_OPEN_UPLOAD_SESSIONS_PER_LOGIN = int(os.environ.get("DRAFT_MAX_OPEN_SESSIONS", "4"))
MAX_DECOMPRESSED_LABEL_BYTES = 64 * 1024 * 1024  # zip-bomb guard for gzipped JSONL
VERIFY_SUBPROCESS_TIMEOUT_SEC = int(os.environ.get("DRAFT_VERIFY_TIMEOUT_SEC", "120"))
VERIFY_SUBPROCESS_MEM_LIMIT_BYTES = int(os.environ.get("DRAFT_VERIFY_MEM_BYTES", str(1024 * 1024 * 1024)))
MAX_CONCURRENT_VERIFIES = int(os.environ.get("DRAFT_MAX_CONCURRENT_VERIFIES", "2"))
STAGING_ORPHAN_TTL_SEC = int(os.environ.get("DRAFT_STAGING_ORPHAN_TTL_SEC", "3600"))

MAX_UPLOAD_BYTES_BY_KIND = {
    "drafter-eval": 2 * 1024 * 1024,          # 2 MiB (<50 KB expected)
    "drafter-pack": 2 * 1024 * 1024,          # 2 MiB (<50 KB deterministic TP rank SHA-256 manifest)
    "drafter-label": 32 * 1024 * 1024,        # 32 MiB (~1.2 MB gz expected)
    "drafter-capture": 256 * 1024 * 1024,     # 256 MiB for pilot 4-tap INT8 shard pack
    "drafter-train": 2_800 * 1024 * 1024,     # 2.8 GB (59 MB S1 / 548 MB SparseLoCo / 2.44 GB INT8 dense)
}

DEFAULT_CLAIM_TTL_SEC = 6 * 3600  # 6 hours
VOCAB_SIZE = 154_880
PINNED_GLM53_TOKENIZER_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"

# Access control, tiers, claim caps, strikes (Fable 4.8 & S7)
KIND_TO_TIER = {
    "drafter-label": "tier-label",
    "drafter-train": "tier-train",
    "drafter-eval": "tier-eval",
    "drafter-capture": "tier-capture",
    "drafter-pack": "tier-pack",
}
NON_OWNER_ALLOWED_KINDS = frozenset(
    x.strip() for x in os.environ.get("DRAFT_NON_OWNER_KINDS", "drafter-label").split(",") if x.strip()
)
OWNER_LOGINS = frozenset(
    x.strip()
    for x in os.environ.get(
        "DRAFT_OWNER_LOGINS",
        "admin,bertholomus-w1,bertholomus-w2,bertholomus-w3,bertholomus-w4,"
        "bertholomus-w5,bertholomus-w6,bertholomus-w7,bertholomus-w8,bertholomus-qa",
    ).split(",")
    if x.strip()
)
MAX_ACTIVE_CLAIMS_PER_LOGIN = int(os.environ.get("DRAFT_MAX_CLAIMS_PER_LOGIN", "2"))
MAX_ACTIVE_CLAIMS_OWNER = int(os.environ.get("DRAFT_MAX_CLAIMS_OWNER", "8"))
MAX_STRIKES_BEFORE_QUARANTINE = int(os.environ.get("DRAFT_MAX_STRIKES", "3"))
CLAIM_RATE_LIMIT_PER_MIN = int(os.environ.get("DRAFT_CLAIM_RATE_PER_MIN", "60"))

# Owner-side label spot-rescoring gate (LABEL-VERIFY-FIX.md)
LABEL_GATE_TRUSTED_LOGINS = frozenset(
    x.strip() for x in os.environ.get("DRAFT_LABEL_TRUSTED_LOGINS", "bertholomus-w1").split(",") if x.strip()
)
LABEL_SPOT_MIN_TOP1 = 0.850        # >= 68/80 probes
LABEL_SPOT_MIN_TOP5 = 0.960        # >= 77/80 probes
LABEL_SPOT_MIN_PROMPT_TOP1 = 0.50  # >= 5/10 per sampled prompt
LABEL_CANARY_MIN_TOP1 = 0.920      # canary bar for new external engine
LABEL_CANARY_MIN_TOP5 = 0.990

# Verification thresholds for owner-only eval/capture/train paths
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
    """Opt-in legacy bootstrap only (disabled by default; minted keys live hashed in SQLite)."""
    if not BOOTSTRAP_WORKER_KEYS:
        return {}
    p = SECRETS_DIR / "pilot_worker_keys.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}
