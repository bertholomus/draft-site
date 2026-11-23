"""Tests for worker/client.py (S11), worker/label_worker.py (Goal 1 & 4), fallbacks, and repo hygiene."""
from __future__ import annotations

import gzip
import hashlib
import http.server
import json
import re
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))
import client as wrk_client  # noqa: E402
import fleet_label_worker as flw  # noqa: E402
import label_worker as lw  # noqa: E402


def test_client_origin_pinning_no_redirect_and_mandatory_sha_s11(tmp_path: Path):
    evil_seen_headers: list[dict[str, str]] = []

    class _EvilHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            evil_seen_headers.append({k.lower(): v for k, v in self.headers.items()})
            body = b"evil-payload"
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(body)

    class _OriginHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            if self.path == "/api/redirect":
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{evil_port}/steal")
                self.end_headers()
            elif self.path == "/api/artifacts/a1":
                data = b"verified-content-bytes"
                self.send_response(200)
                self.send_header("X-SHA256", hashlib.sha256(data).hexdigest())
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')

    evil_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _EvilHandler)
    evil_port = evil_srv.server_address[1]
    orig_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _OriginHandler)
    orig_port = orig_srv.server_address[1]
    threading.Thread(target=evil_srv.serve_forever, daemon=True).start()
    threading.Thread(target=orig_srv.serve_forever, daemon=True).start()

    try:
        cli = wrk_client.DraftSiteClient(f"http://127.0.0.1:{orig_port}", api_key="secret_worker_token_xyz")

        # 1. Cross-origin API request MUST be refused before touching network
        with pytest.raises(ValueError, match="cross-origin"):
            cli._request("GET", f"http://127.0.0.1:{evil_port}/steal")
        assert evil_seen_headers == []

        # 2. HTTP 302 redirect MUST be refused (never forwards Authorization header!)
        with pytest.raises(wrk_client.DraftSiteError) as exc_info:
            cli._request("GET", "/api/redirect")
        assert exc_info.value.status == 302
        assert evil_seen_headers == []

        # 3. Cross-origin download refused by default; even with allow_external=True, NEVER sends Authorization!
        evil_sha = hashlib.sha256(b"evil-payload").hexdigest()
        with pytest.raises(ValueError, match="cross-origin"):
            cli.download_artifact(f"http://127.0.0.1:{evil_port}/steal", tmp_path / "e1.bin", evil_sha)
        cli.download_artifact(f"http://127.0.0.1:{evil_port}/steal", tmp_path / "e2.bin", evil_sha, allow_external=True)
        assert len(evil_seen_headers) == 1
        assert "authorization" not in evil_seen_headers[0]

        # 4. Mandatory checksum on download_artifact
        with pytest.raises(ValueError, match="expected_sha256"):
            cli.download_artifact("a1", tmp_path / "a1.bin", None)
        with pytest.raises(ValueError, match="expected_sha256"):
            cli.download_artifact("a1", tmp_path / "a1.bin", "")

        # 5. Wrong checksum raises RuntimeError and cleans .part file
        with pytest.raises(RuntimeError):
            cli.download_artifact("a1", tmp_path / "a1_bad.bin", "0" * 64)
        assert not (tmp_path / "a1_bad.bin").exists()
        assert not (tmp_path / "a1_bad.bin.part").exists()

        # 6. Exact checksum succeeds
        good_sha = hashlib.sha256(b"verified-content-bytes").hexdigest()
        out = cli.download_artifact("a1", tmp_path / "a1_ok.bin", good_sha)
        assert out.read_bytes() == b"verified-content-bytes"
    finally:
        evil_srv.shutdown()
        evil_srv.server_close()
        orig_srv.shutdown()
        orig_srv.server_close()


def test_workers_fail_hard_zero_silent_fallbacks(tmp_path: Path):
    # label_worker.py: missing token_ids without tokenizer MUST fail hard
    with pytest.raises(RuntimeError):
        lw.extract_gen_ids({"text": "hello world"}, None, "p0")
    with pytest.raises(RuntimeError):
        lw.extract_gen_ids({"token_ids": []}, None, "p0")
    with pytest.raises(RuntimeError):
        lw.extract_gen_ids({"token_ids": [True, 5]}, None, "p0")
    with pytest.raises(RuntimeError):
        lw.extract_gen_ids({"token_ids": [999999]}, None, "p0")

    # PinnedTokenizer fails closed on missing or hash-mismatched tokenizer.json
    with pytest.raises(RuntimeError, match="missing tokenizer.json"):
        lw.PinnedTokenizer(tmp_path)
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        lw.PinnedTokenizer(tmp_path)

    # fleet_label_worker.py: missing/empty token_ids MUST fail hard (no [154829] fallback)
    with pytest.raises(RuntimeError):
        flw.extract_strict_token_ids({"text": "hello"}, "p0")
    with pytest.raises(RuntimeError):
        flw.extract_strict_token_ids({"token_ids": []}, "p0")
    assert flw.extract_strict_token_ids({"token_ids": [10, 20, 30]}, "p0") == [10, 20, 30]

    # Source check: zero byte-token / [1] / [154829] fallbacks left in any worker file
    for fname in ("label_worker.py", "fleet_label_worker.py", "draft_worker.py"):
        src = (ROOT / "worker" / fname).read_text(encoding="utf-8")
        assert "int(b) + 100" not in src, f"byte fallback still present in {fname}"
        assert "gids = [154829]" not in src, f"[154829] fallback still present in {fname}"
        assert "gen_ids = [1]" not in src, f"[1] fallback still present in {fname}"


def test_label_worker_records_full_provenance_per_row(tmp_path: Path):
    class _MockLLMHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            assert isinstance(req["prompt"], list) and req["return_token_ids"] is True
            body = json.dumps(
                {
                    "model": "glm-5.3",
                    "choices": [{"token_ids": [11, 22, 33, 44, 55], "finish_reason": "stop"}],
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    llm_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MockLLMHandler)
    llm_url = f"http://127.0.0.1:{llm_srv.server_address[1]}"
    threading.Thread(target=llm_srv.serve_forever, daemon=True).start()

    pids = [154822, 154824, 99, 100]
    psha = hashlib.sha256(json.dumps(pids, separators=(",", ":")).encode("utf-8")).hexdigest()
    shard_file = tmp_path / "shard.jsonl"
    shard_file.write_text(
        json.dumps({"id": "p0", "prompt_ids": pids, "prompt_sha256": psha, "max_tokens": 32}) + "\n",
        encoding="utf-8",
    )
    shard_sha = wrk_client.sha256_file(shard_file)
    uploaded_rows: list[dict] = []

    class _FakeCli:
        def download_artifact(self, url_or_id, dest, expected_sha256):
            assert expected_sha256 == shard_sha
            Path(dest).write_bytes(shard_file.read_bytes())
            return Path(dest)

        def upload_artifact(self, claim, file_path, manifest=None):
            raw = gzip.decompress(Path(file_path).read_bytes()).decode("utf-8")
            for ln in raw.splitlines():
                if ln.strip():
                    uploaded_rows.append(json.loads(ln))
            return {"accepted": True, "state": "uploaded", "verify_status": "pending_label_gate"}

    try:
        claim = {
            "job": "labels-prod",
            "slot_id": "s0",
            "plan": {"max_tokens_short": 32},
            "inputs": [{"name": "prompt_shard", "url": "https://draft.bertholomus.com/api/artifacts/inp0", "sha256": shard_sha}],
        }
        resp = lw.run_one_slot(
            _FakeCli(),  # type: ignore[arg-type]
            claim,
            endpoint=llm_url,
            llm_api_key=None,
            model_id="glm-5.3",
            concurrency=2,
            engine_tag="kindling-gr8-bit280",
            engine_image_digest="ghcr.io/kindlingai/gr8@sha256:" + "c" * 64,
            engine_signature="gr8-sha-abc123",
            quant_profile="bit280",
            tokenizer=None,
            cache_dir=tmp_path / "cache",
        )
        assert resp["state"] == "uploaded"
        assert len(uploaded_rows) == 1
        row = uploaded_rows[0]
        assert row["gen_ids"] == [11, 22, 33, 44, 55]
        assert row["engine_tag"] == "kindling-gr8-bit280"
        assert row["engine_image_digest"] == "ghcr.io/kindlingai/gr8@sha256:" + "c" * 64
        assert row["engine_signature"] == "gr8-sha-abc123"
        assert row["quant_profile"] == "bit280"
        assert row["concurrency"] == 2
        assert row["token_id_source"] == "engine"
    finally:
        llm_srv.shutdown()
        llm_srv.server_close()


def test_external_worker_and_docs_hygiene_goal4():
    forbidden = re.compile(
        r"/tank/|frankennas|ascent-\d+|mastersim|192\.168\.|100\.108\.|draft-origin|albertlee|\.hermes|18090|18780|bertholomus-w|u320837031|PROGRESS\.md",
        re.IGNORECASE,
    )
    for rel in ("worker/label_worker.py", "worker/client.py", "LABELS.md", "LICENSE"):
        txt = (ROOT / rel).read_text(encoding="utf-8")
        hits = forbidden.findall(txt)
        assert not hits, f"private identifier leak in {rel}: {hits}"
