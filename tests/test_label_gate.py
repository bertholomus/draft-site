"""Unit & integration tests for worker/label_verify_gate.py (S1 owner teacher-forced gate)."""
from __future__ import annotations

import gzip
import hashlib
import http.server
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))
import label_verify_gate as lvg  # noqa: E402


def _det_next_token(prefix: list[int]) -> int:
    """Deterministic pseudo-GLM-5.3 next-token oracle used by the mock /v1/completions server."""
    h = hashlib.sha256(json.dumps(prefix, separators=(",", ":")).encode("utf-8")).digest()
    return 100 + (int.from_bytes(h[:4], "big") % 50_000)


class _MockRefHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "glm-5.3"}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n).decode("utf-8"))
        assert "prompt_logprobs" not in req, "prompt_logprobs MUST NOT be sent to reference lane"
        assert req["max_tokens"] == 1
        assert req["logprobs"] == 5
        assert req["return_token_ids"] is True
        assert req["return_tokens_as_token_ids"] is True
        prefix = req["prompt"]
        top1 = _det_next_token(prefix)
        top5 = {f"token_id:{top1 + k}": -0.1 * k for k in range(5)}
        resp = {
            "model": "glm-5.3",
            "choices": [
                {
                    "token_ids": [top1],
                    "text": "",
                    "finish_reason": "length",
                    "logprobs": {"top_logprobs": [top5]},
                }
            ],
        }
        body = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


def _start_mock_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MockRefHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{port}"


def _build_synthetic_shard(tmp_path: Path, n_prompts: int = 16, gen_len: int = 40, poison_frac: float = 0.0):
    prompt_lines = []
    label_lines = []
    n_poison = int(round(n_prompts * poison_frac))
    for i in range(n_prompts):
        pids = [154822, 154824, 500 + i, 600 + i, 700 + i]
        psha = hashlib.sha256(json.dumps(pids, separators=(",", ":")).encode("utf-8")).hexdigest()
        gids: list[int] = []
        prefix = list(pids)
        for step in range(gen_len):
            tok = _det_next_token(prefix)
            if i < n_poison:
                tok = (tok + 7777) % 150_000
            gids.append(tok)
            prefix.append(tok)
        gsha = hashlib.sha256(json.dumps(gids, separators=(",", ":")).encode("utf-8")).hexdigest()
        prompt_lines.append(
            json.dumps({"id": f"pr_{i:03d}", "prompt_ids": pids, "prompt_sha256": psha, "category": "code"}, separators=(",", ":"))
        )
        label_lines.append(
            json.dumps(
                {
                    "id": f"pr_{i:03d}",
                    "prompt_sha256": psha,
                    "gen_ids": gids,
                    "finish_reason": "stop",
                    "engine_tag": "mock-gr8",
                    "engine_image_digest": "sha256:" + "b" * 64,
                    "engine_signature": "sig-mock",
                    "quant_profile": "bit280",
                    "concurrency": 4,
                    "gen_sha256": gsha,
                },
                separators=(",", ":"),
            )
        )
    pf = tmp_path / "prompts.jsonl"
    lf = tmp_path / "labels.jsonl.gz"
    pf.write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
    lf.write_bytes(gzip.compress(("\n".join(label_lines) + "\n").encode("utf-8")))
    return pf, lf


def test_hmac_sampling_deterministic_unguessable_and_starts_at_zero():
    prows = {f"p{i:02d}": {"prompt_ids": [1, 2, 3, 4]} for i in range(20)}
    lrows = {f"p{i:02d}": {"gen_ids": list(range(100, 150))} for i in range(20)}

    p1 = lvg.select_probe_plan(prows, lrows, hmac_secret="sec-a", job="j", slot_id="s0", upload_sha256="aa" * 32)
    p2 = lvg.select_probe_plan(prows, lrows, hmac_secret="sec-a", job="j", slot_id="s0", upload_sha256="aa" * 32)
    p3 = lvg.select_probe_plan(prows, lrows, hmac_secret="sec-b", job="j", slot_id="s0", upload_sha256="aa" * 32)
    assert p1 == p2
    assert [x[0] for x in p1] != [x[0] for x in p3]
    assert len(p1) == 8
    for pid, pids, gids, offsets in p1:
        assert len(offsets) == 10
        assert offsets[0] == 0
        assert offsets == sorted(set(offsets))


def test_a14_window_detection(tmp_path: Path):
    prog = tmp_path / "PROGRESS.md"
    prog.write_text("- DSP NOTE prose mentioning DSP WINDOW-START in the middle of a sentence\n", encoding="utf-8")
    assert lvg.a14_window_is_open(prog) is False

    prog.write_text(
        "- DSP WINDOW-START gr8-speed test 2026-11-22T23:47:58Z\n",
        encoding="utf-8",
    )
    assert lvg.a14_window_is_open(prog) is True

    prog.write_text(
        "- DSP WINDOW-START gr8-speed test 2026-11-22T23:47:58Z\n"
        "- DSP WINDOW-END gr8-speed test rc=0 2026-11-23T00:20:00Z\n",
        encoding="utf-8",
    )
    assert lvg.a14_window_is_open(prog) is False


def test_score_label_shard_honest_passes_and_corrupted_fails(tmp_path: Path):
    srv, endpoint = _start_mock_server()
    try:
        pf_ok, lf_ok = _build_synthetic_shard(tmp_path / "ok" if False else tmp_path, n_prompts=12, gen_len=40, poison_frac=0.0)
        passed, reason, gm, exp = lvg.score_label_shard(
            pf_ok,
            lf_ok,
            endpoint=endpoint,
            model_id="glm-5.3",
            hmac_secret="test-secret",
            job="labels-prod",
            slot_id="s0",
            upload_sha256="11" * 32,
        )
        assert passed is True, reason
        assert gm["n_prompts"] == 8 and gm["n_probes"] == 80
        assert gm["top1_agree"] == 1.0 and gm["top5_agree"] == 1.0 and gm["canary_pass"] is True
        assert len(exp) == 12 and exp[0]["engine_tag"] == "mock-gr8"

        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        pf_bad, lf_bad = _build_synthetic_shard(bad_dir, n_prompts=12, gen_len=40, poison_frac=0.5)
        passed_b, reason_b, gm_b, exp_b = lvg.score_label_shard(
            pf_bad,
            lf_bad,
            endpoint=endpoint,
            model_id="glm-5.3",
            hmac_secret="test-secret",
            job="labels-prod",
            slot_id="s0",
            upload_sha256="22" * 32,
        )
        assert passed_b is False, reason_b
        assert gm_b["min_prompt_top1"] < 0.50
        assert exp_b == []
    finally:
        srv.shutdown()
        srv.server_close()
