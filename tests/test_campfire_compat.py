"""Campfire compatibility & Guardrail-1 public-surface hygiene tests for draft.bertholomus.com."""
from __future__ import annotations

import re
from typing import Any


def test_campfire_skill_md_and_project_route(site_env: dict[str, Any]) -> None:
    client = site_env["client"]
    adm = site_env["adm_hdr"]
    pub = site_env["pub_hdr"]
    orig = site_env["origin_hdr"]

    # S4 origin auth enforced on /skill.md and /jobs/<job>
    assert client.get("/skill.md").status_code == 403
    assert client.get("/jobs/lbl-proj").status_code == 403

    r_sk = client.get("/skill.md", headers=orig)
    assert r_sk.status_code == 200
    assert "text/markdown" in r_sk.headers.get("content-type", "")
    assert r_sk.text.startswith("---\nname: drafter-label\n")
    assert "## What the user must provide" in r_sk.text
    assert "## 1. See what work exists" in r_sk.text
    assert "DELETE /api/claims" in r_sk.text

    # Unknown project returns 404
    r_404 = client.get("/jobs/does-not-exist", headers=orig)
    assert r_404.status_code == 404

    # Seed project and verify /jobs/<job> renders strict CSP with data-initial-job
    assert client.post(
        "/api/admin/jobs",
        headers=adm,
        json={
            "job": "lbl-proj",
            "title": "Label Project",
            "job_kind": "drafter-label",
            "slots": [
                {"slot_id": "s0", "row_key": "r0", "col_key": "c0", "state": "open"},
                {"slot_id": "s1", "row_key": "r0", "col_key": "c1", "state": "closed"},
            ],
        },
    ).status_code == 200

    r_job = client.get("/jobs/lbl-proj", headers=orig)
    assert r_job.status_code == 200
    assert 'data-initial-job="lbl-proj"' in r_job.text
    assert "script-src 'self'" in r_job.headers.get("content-security-policy", "")
    assert "onclick=" not in r_job.text

    # Guardrail 1 hygiene across public pages (/, /jobs/<job>, /static/app.js, /skill.md)
    forbidden = re.compile(
        r"v5-s2-w5|s500|sparseloco|codebridge|L33/47/61/75|dflash2|dspine|5\.81|4\.47",
        re.IGNORECASE,
    )
    for path in ("/", "/jobs/lbl-proj", "/static/app.js", "/skill.md"):
        body = client.get(path, headers=orig).text
        hits = forbidden.findall(body)
        assert not hits, f"unreleased drafter details leaked on public route {path}: {hits}"

    # Verify /api/jobs emits Campfire-aligned fields + ?kind= filter + closed count
    jobs = client.get("/api/jobs", headers=pub).json()
    assert len(jobs) == 1
    j = jobs[0]
    for k in (
        "job",
        "title",
        "kind",
        "job_kind",
        "model",
        "version",
        "layers",
        "experts",
        "slots",
        "uploaded",
        "verified",
        "pending_verify",
        "claimed",
        "disabled",
        "open",
        "closed",
        "not_ready",
        "harvested",
        "ready_layers",
        "done_layers",
        "contributors",
        "last_upload",
        "weights",
        "codebooks",
    ):
        assert k in j, f"/api/jobs missing Campfire field '{k}'"
    assert j["kind"] == "drafter-label"
    assert j["job_kind"] == "drafter-label"
    assert j["open"] == 1 and j["closed"] == 1
    assert j["weights"]["id"] is None
    assert j["codebooks"] == []

    assert len(client.get("/api/jobs?kind=drafter-label", headers=pub).json()) == 1
    assert client.get("/api/jobs?kind=encode", headers=pub).json() == []


def test_campfire_closed_state_kind_claims_and_release_all(site_env: dict[str, Any]) -> None:
    client = site_env["client"]
    adm = site_env["adm_hdr"]
    pub = site_env["pub_hdr"]

    assert client.post("/api/admin/runners", headers=adm, json={"commit_id": "drs-label-ext-v1.0.0", "kind": "drafter-label"}).status_code == 200
    assert client.post("/api/admin/runners", headers=adm, json={"commit_id": "other-eval", "kind": "drafter-eval"}).status_code == 200
    assert client.post(
        "/api/admin/jobs",
        headers=adm,
        json={
            "job": "lbl-c",
            "title": "Label Campfire Compat",
            "job_kind": "drafter-label",
            "slots": [
                {"slot_id": "s0", "row_key": "r0", "col_key": "c0", "state": "open"},
                {"slot_id": "s1", "row_key": "r0", "col_key": "c1", "state": "open"},
                {"slot_id": "s2", "row_key": "r0", "col_key": "c2", "state": "open"},
            ],
        },
    ).status_code == 200

    # Admin can mark s2 as Campfire 'closed' (inputs up, claims off)
    r_cl = client.post("/api/admin/slots/lbl-c/s2/ready", headers=adm, json={"state": "closed"})
    assert r_cl.status_code == 200 and r_cl.json()["state"] == "closed"

    # /api/runners?kind= filters properly
    r_lbl = client.get("/api/runners?kind=drafter-label", headers=pub).json()
    assert [r["commit_id"] for r in r_lbl] == ["drs-label-ext-v1.0.0"]

    _, ext_hdr = site_env["mint_user"]("ext-cf", tiers=["tier-label"], is_owner=False)

    # /api/me includes Campfire `role`
    me = client.get("/api/me", headers=ext_hdr).json()
    assert me["role"] == "contributor"

    # Encode claim against drafter site is rejected cleanly (400)
    r_enc = client.post(
        "/api/claims",
        headers=ext_hdr,
        json={"kind": "encode", "runner": "drs-label-ext-v1.0.0", "request": "req_enc"},
    )
    assert r_enc.status_code == 400

    # Claim s0 and s1 with explicit kind='drafter-label'
    c0 = client.post(
        "/api/claims",
        headers=ext_hdr,
        json={"kind": "drafter-label", "runner": "drs-label-ext-v1.0.0", "job": "lbl-c", "request": "req_0"},
    )
    assert c0.status_code == 201
    c0j = c0.json()
    assert c0j["kind"] == "drafter-label"
    assert c0j["weights"]["id"] is None

    c1 = client.post(
        "/api/claims",
        headers=ext_hdr,
        json={"kind": "drafter-label", "runner": "drs-label-ext-v1.0.0", "job": "lbl-c", "request": "req_1"},
    )
    assert c1.status_code == 201

    # Closed slot s2 cannot be claimed
    r_s2 = client.post(
        "/api/claims",
        headers=ext_hdr,
        json={"kind": "drafter-label", "runner": "drs-label-ext-v1.0.0", "job": "lbl-c", "slot_id": "s2", "request": "req_2"},
    )
    assert r_s2.status_code in (409, 429)

    # GET /api/claims?kind=drafter-label returns both; ?kind=encode returns []
    assert len(client.get("/api/claims?kind=drafter-label", headers=ext_hdr).json()) == 2
    assert client.get("/api/claims?kind=encode", headers=ext_hdr).json() == []

    # Campfire DELETE /api/claims releases all held claims at once
    r_del_all = client.delete("/api/claims", headers=ext_hdr)
    assert r_del_all.status_code == 200
    assert r_del_all.json()["released"] == 2
    assert client.get("/api/claims", headers=ext_hdr).json() == []

    # /api/grid includes Campfire kind & weights
    g = client.get("/api/grid?job=lbl-c", headers=pub).json()
    assert g["kind"] == "drafter-label"
    assert g["weights"]["id"] is None
