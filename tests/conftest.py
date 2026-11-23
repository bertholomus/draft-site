"""Pytest fixtures for isolated, zero-leak draft-site tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

ORIGIN_SECRET = "test_origin_secret_hex_1234567890abcdef"
ADMIN_KEY = "drs_adm_test_admin_key_0123456789abcdef"


@pytest.fixture()
def site_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    data_dir = tmp_path / "data"
    sec_dir = tmp_path / "secrets"
    data_dir.mkdir(parents=True, exist_ok=True)
    sec_dir.mkdir(parents=True, exist_ok=True)
    (sec_dir / "origin_auth.secret").write_text(ORIGIN_SECRET, encoding="utf-8")
    (sec_dir / "admin_api_key.txt").write_text(ADMIN_KEY, encoding="utf-8")

    monkeypatch.setenv("DRAFT_DATA_DIR", str(data_dir))
    monkeypatch.setenv("DRAFT_SECRETS_DIR", str(sec_dir))
    monkeypatch.setenv("DRAFT_PUBLIC_URL", "https://draft.bertholomus.com")
    monkeypatch.setenv("DRAFT_REQUIRE_ORIGIN_AUTH", "1")
    monkeypatch.setenv("DRAFT_BOOTSTRAP_WORKER_KEYS", "0")

    from server import app as srv_app, config, db

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "SECRETS_DIR", sec_dir)
    monkeypatch.setattr(config, "DB_PATH", data_dir / "db" / "draft_site.sqlite3")
    monkeypatch.setattr(config, "INPUTS_DIR", data_dir / "inputs")
    monkeypatch.setattr(config, "STAGING_DIR", data_dir / "uploads" / "staging")
    monkeypatch.setattr(config, "VERIFIED_DIR", data_dir / "uploads" / "verified")
    monkeypatch.setattr(config, "QUARANTINE_DIR", data_dir / "uploads" / "quarantine")
    monkeypatch.setattr(config, "EVALS_DIR", data_dir / "evals")
    monkeypatch.setattr(config, "REQUIRE_ORIGIN_AUTH", True)
    monkeypatch.setattr(config, "BOOTSTRAP_WORKER_KEYS", False)
    monkeypatch.setattr(config, "MAX_ACTIVE_CLAIMS_PER_LOGIN", 2)
    monkeypatch.setattr(config, "MAX_STRIKES_BEFORE_QUARANTINE", 3)
    monkeypatch.setattr(config, "MIN_CHUNK_BYTES", 256 * 1024)
    monkeypatch.setattr(config, "MAX_CHUNKS_PER_SESSION", 512)
    monkeypatch.setattr(config, "LABEL_GATE_TRUSTED_LOGINS", frozenset({"bertholomus-w1"}))
    monkeypatch.setattr(config, "NON_OWNER_ALLOWED_KINDS", frozenset({"drafter-label"}))
    monkeypatch.setattr(config, "CLAIM_RATE_LIMIT_PER_MIN", 200)

    srv_app._CLAIM_RATE_BUCKETS.clear()
    srv_app._VERIFY_SEMAPHORE = None
    db.init_db()

    origin_hdr = {"X-Origin-Auth": ORIGIN_SECRET}
    adm_hdr = {"X-Origin-Auth": ORIGIN_SECRET, "X-Quant": "1", "Authorization": f"Bearer {ADMIN_KEY}"}

    with TestClient(srv_app.app) as client:
        def mint_user(
            login: str,
            *,
            tiers: list[str] | None = None,
            is_owner: bool = False,
            approved: bool = True,
            quarantined: bool = False,
        ) -> tuple[str, dict[str, str]]:
            payload: dict[str, Any] = {
                "login": login,
                "is_owner": is_owner,
                "approved": approved,
                "quarantined": quarantined,
            }
            if tiers is not None:
                payload["tiers"] = tiers
            r = client.post("/api/admin/contributors", headers=adm_hdr, json=payload)
            assert r.status_code == 200, r.text
            key = r.json()["api_key_shown_once"]
            return key, {"X-Origin-Auth": ORIGIN_SECRET, "X-Quant": "1", "Authorization": f"Bearer {key}"}

        yield {
            "client": client,
            "data_dir": data_dir,
            "secrets_dir": sec_dir,
            "origin_hdr": origin_hdr,
            "pub_hdr": {"X-Origin-Auth": ORIGIN_SECRET, "X-Quant": "1"},
            "adm_hdr": adm_hdr,
            "mint_user": mint_user,
        }
