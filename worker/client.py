"""Authenticated HTTP client for draft.bertholomus.com (quant.kindlingai.dev API style)."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

USER_AGENT = "draft-worker/1.0 (GLM-5.3 DSpine Community Pilot)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class DraftSiteError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body
        try:
            self.json = json.loads(body)
        except Exception:
            self.json = None


class DraftSiteClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: int = 240):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip() if api_key else None
        self.timeout = timeout

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        h = {"X-Quant": "1", "User-Agent": USER_AGENT, "Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if extra:
            h.update(extra)
        return h

    def _request(
        self,
        method: str,
        path_or_url: str,
        *,
        json_body: Any = None,
        raw_body: bytes | None = None,
        headers: dict[str, str] | None = None,
        retries: int = 3,
    ) -> Any:
        url = path_or_url if path_or_url.startswith("http") else (self.base_url + path_or_url)
        hdrs = self._headers(headers)
        data = raw_body
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")

        last_err: Exception | None = None
        for attempt in range(retries):
            req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    if not raw:
                        return {}
                    return json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                if e.code in (502, 503, 504) and attempt + 1 < retries:
                    time.sleep(2 * (attempt + 1))
                    last_err = DraftSiteError(e.code, body)
                    continue
                raise DraftSiteError(e.code, body) from e
            except Exception as e:
                last_err = e
                if attempt + 1 < retries:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        raise last_err  # pragma: no cover

    # --- Public & contributor reads ---
    def jobs(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/jobs")

    def grid(self, job: str) -> dict[str, Any]:
        return self._request("GET", f"/api/grid?job={urllib.parse.quote(job)}")

    def runners(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/runners")

    def contributors(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/contributors")

    def me(self) -> dict[str, Any]:
        return self._request("GET", "/api/me")

    def my_claims(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/claims")

    # --- Claim / release ---
    def claim(
        self,
        *,
        runner: str,
        job: str | None = None,
        slot_id: str | None = None,
        profile: str | None = None,
        round_num: int | None = None,
        capabilities: list[str] | None = None,
        host_tag: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "runner": runner,
            "job_version": 1,
            "request": "req_" + secrets.token_hex(12),
        }
        if job:
            body["job"] = job
        if slot_id:
            body["slot_id"] = slot_id
        if profile:
            body["profile"] = profile
        if round_num is not None:
            body["round"] = round_num
        if capabilities:
            body["capabilities"] = capabilities
        if host_tag:
            body["host_tag"] = host_tag
        return self._request("POST", "/api/claims", json_body=body)

    def release(self, job: str, slot_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/claims/{urllib.parse.quote(job)}/{urllib.parse.quote(slot_id)}")

    # --- Streaming artifact download with SHA-256 verification ---
    def download_artifact(self, url_or_id: str, dest: Path, expected_sha256: str | None = None) -> Path:
        url = url_or_id if url_or_id.startswith("http") else f"{self.base_url}/api/artifacts/{url_or_id}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file() and expected_sha256 and sha256_file(dest) == expected_sha256.lower():
            return dest
        tmp = dest.with_suffix(dest.suffix + ".part")
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        h = hashlib.sha256()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp, tmp.open("wb") as out:
            while True:
                buf = resp.read(1024 * 1024)
                if not buf:
                    break
                h.update(buf)
                out.write(buf)
        got = h.hexdigest()
        if expected_sha256 and got != expected_sha256.lower():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"download SHA-256 mismatch for {url}: got {got}, expected {expected_sha256}")
        os.replace(tmp, dest)
        return dest

    # --- Direct & chunked verified upload ---
    def upload_artifact(
        self,
        claim: dict[str, Any],
        file_path: Path,
        manifest: dict[str, Any] | None = None,
        progress_cb=None,
    ) -> dict[str, Any]:
        file_path = Path(file_path)
        total_bytes = file_path.stat().st_size
        total_sha = sha256_file(file_path)
        up_info = claim["upload"]
        max_direct = int(up_info.get("max_direct_bytes", 10 * 1024 * 1024))
        chunk_bytes = min(int(up_info.get("chunk_bytes", 8 * 1024 * 1024)), 8 * 1024 * 1024)

        if total_bytes <= max_direct:
            raw = file_path.read_bytes()
            direct_url = up_info["direct_url"] + "?" + urllib.parse.urlencode({"filename": file_path.name})
            hdrs = {
                "Content-Type": "application/octet-stream",
                "X-Content-SHA256": total_sha,
                "X-Manifest-JSON": json.dumps(manifest or {}, separators=(",", ":")),
            }
            return self._request("POST", direct_url, raw_body=raw, headers=hdrs)

        # Resumable chunked upload
        init_resp = self._request(
            "POST",
            up_info["init_url"],
            json_body={
                "claim_id": claim["claim_id"],
                "filename": file_path.name,
                "total_bytes": total_bytes,
                "total_sha256": total_sha,
                "chunk_bytes": chunk_bytes,
                "manifest": manifest or {},
            },
        )
        upload_id = init_resp["upload_id"]
        total_chunks = int(init_resp["total_chunks"])
        chunk_tmpl = init_resp["chunk_url_template"]
        complete_url = init_resp["complete_url"]

        from concurrent.futures import ThreadPoolExecutor

        def _send_chunk(idx: int, chunk: bytes) -> tuple[int, int]:
            csha = hashlib.sha256(chunk).hexdigest()
            url = chunk_tmpl.replace("{index}", str(idx))
            self._request(
                "PUT",
                url,
                raw_body=chunk,
                headers={"Content-Type": "application/octet-stream", "X-Chunk-SHA256": csha},
                retries=5,
            )
            return idx, len(chunk)

        done_count = 0
        with file_path.open("rb") as f, ThreadPoolExecutor(max_workers=4) as ex:
            in_flight = []
            for idx in range(total_chunks):
                chunk = f.read(chunk_bytes)
                in_flight.append(ex.submit(_send_chunk, idx, chunk))
                if len(in_flight) >= 8:
                    _, nb = in_flight.pop(0).result()
                    done_count += 1
                    if progress_cb:
                        progress_cb(done_count, total_chunks, nb)
            for fut in in_flight:
                _, nb = fut.result()
                done_count += 1
                if progress_cb:
                    progress_cb(done_count, total_chunks, nb)

        return self._request("POST", complete_url, raw_body=b"", retries=3)

    # --- Admin helpers ---
    def admin_post(self, path: str, body: dict[str, Any]) -> Any:
        return self._request("POST", path, json_body=body)

    def admin_get(self, path: str) -> Any:
        return self._request("GET", path)
