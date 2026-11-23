"""Web UI for draft.bertholomus.com — hardened CSP (no inline scripts) and accurate security docs."""
from __future__ import annotations


def render_index_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>draft.bertholomus.com · GLM-5.3 Drafter Label Coordinator</title>
<meta name="description" content="Trusted-contributor label coordinator for GLM-5.3 DSpine B=16 drafter training.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root { color-scheme: dark;
  --ink: #16130F; --paper: #F2EDE4;
  --bg: var(--ink); --fg: var(--paper);
  --accent: #FF6B1A; --accent-soft: #FFC24A; --accent-text: #FF8139;
  --surface: #1E1914; --surface-2: #28211A; --line: #342E26; --muted: #A39A8C;
  --rim: inset 0 1px 0 rgb(255 246 227 / 0.05);
  --lift: 0 1px 0 rgb(0 0 0 / 0.35), 0 18px 40px -22px rgb(0 0 0 / 0.8);
  --code-bg: #0E0C09; --good: #4CD97B; --bad: #FF5C7A;
  --notready: #29241D; --open: #5E5040; --claimed: #FFC24A; --uploaded: #E5A030; --verified: #FF6B1A; --rejected: #FF5C7A; --disabled: #3B3229;
  --mine: #7CC4FF;
  --font-display: "Space Grotesk", ui-sans-serif, system-ui, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  --radius: 12px;
}
:root[data-theme=light] { color-scheme: light;
  --bg: var(--paper); --fg: var(--ink);
  --accent: #E8590C; --accent-soft: #FFB020; --accent-text: #BC3F0C;
  --surface: #FAF7F2; --surface-2: #EAE3D7; --line: #D9D0C2; --muted: #6B6358;
  --rim: inset 0 1px 0 rgb(255 255 255 / 0.7);
  --lift: 0 1px 2px rgb(22 19 15 / 0.06), 0 14px 32px -18px rgb(22 19 15 / 0.22);
  --code-bg: #16130F; --good: #16803C; --bad: #C81E4A;
  --notready: #EAE3D7; --open: #C9B391; --claimed: #FFB020; --uploaded: #D97706; --verified: #E8590C; --rejected: #C81E4A; --disabled: #DCCFB9;
  --mine: #0B6BCB;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.55 var(--font-display); -webkit-font-smoothing: antialiased;
}
a { color: var(--accent-text); text-decoration: none; }
a:hover { text-decoration: underline; }
.wrap { max-width: 1220px; margin: 0 auto; padding: 20px 20px 64px; }
header.top {
  display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
  padding-bottom: 18px; border-bottom: 1px solid var(--line); margin-bottom: 24px;
}
.brand { display: flex; align-items: center; gap: 12px; }
.logo {
  width: 34px; height: 34px; border-radius: 8px; background: var(--surface-2); border: 1px solid var(--line);
  display: grid; place-items: center; font: 700 15px var(--font-mono); color: var(--accent);
}
.brand h1 { margin: 0; font-size: 19px; letter-spacing: -0.02em; }
.brand .sub { font: 12px var(--font-mono); color: var(--muted); }
.top-actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.pill {
  display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 999px;
  border: 1px solid var(--line); background: var(--surface); font: 12px var(--font-mono); color: var(--muted);
}
.pill .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--good); }
.btn {
  border: 1px solid var(--line); background: var(--surface); color: var(--fg);
  padding: 6px 12px; border-radius: 8px; font: 500 13px var(--font-mono); cursor: pointer;
}
.btn:hover { border-color: var(--accent); color: var(--accent-text); }
.hero {
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
  box-shadow: var(--rim), var(--lift); padding: 22px 24px; margin-bottom: 22px;
}
.hero h2 { margin: 0 0 8px; font-size: 24px; letter-spacing: -0.02em; }
.hero p { margin: 0 0 12px; color: var(--muted); max-width: 900px; }
.hero-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 16px; }
.stage-card {
  background: var(--surface-2); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px;
}
.stage-card .k { font: 600 12px var(--font-mono); color: var(--accent-text); text-transform: uppercase; letter-spacing: 0.04em; }
.stage-card .t { font-weight: 600; margin: 3px 0 4px; }
.stage-card .d { font-size: 13px; color: var(--muted); }
.tabs { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 18px; }
.tab {
  border: 1px solid var(--line); background: var(--surface); color: var(--muted);
  padding: 8px 14px; border-radius: 9px; font: 500 13px var(--font-mono); cursor: pointer;
}
.tab.active { border-color: var(--accent); color: var(--fg); background: var(--surface-2); }
.panel {
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
  box-shadow: var(--rim), var(--lift); margin-bottom: 20px; overflow: hidden;
}
.panel > .head {
  display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
  padding: 13px 18px; border-bottom: 1px solid var(--line);
}
.panel > .head h3 { margin: 0; font-size: 16px; }
.panel > .head .meta { font: 12px var(--font-mono); color: var(--muted); }
.panel > .body { padding: 18px; }
.jobs-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(275px, 1fr)); gap: 14px; }
.job-card {
  background: var(--surface-2); border: 1px solid var(--line); border-radius: 10px; padding: 14px;
  cursor: pointer; transition: border-color 0.15s;
}
.job-card:hover, .job-card.selected { border-color: var(--accent); }
.job-card .kind { font: 600 11px var(--font-mono); color: var(--accent-text); text-transform: uppercase; }
.job-card .title { font-weight: 600; margin: 4px 0 6px; font-size: 15px; }
.bar { height: 8px; border-radius: 999px; background: var(--notready); overflow: hidden; display: flex; margin: 8px 0; }
.bar > span { display: block; height: 100%; }
.stats-row { display: flex; gap: 10px; flex-wrap: wrap; font: 12px var(--font-mono); color: var(--muted); }
.legend { display: flex; gap: 12px; flex-wrap: wrap; font: 12px var(--font-mono); color: var(--muted); margin-bottom: 14px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: middle; }
.slot-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(175px, 1fr)); gap: 10px; }
.slot-cell {
  border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: var(--surface-2);
  font: 12px var(--font-mono); position: relative;
}
.slot-cell.st-verified { border-left: 4px solid var(--verified); }
.slot-cell.st-uploaded { border-left: 4px solid var(--uploaded); }
.slot-cell.st-claimed { border-left: 4px solid var(--claimed); }
.slot-cell.st-open { border-left: 4px solid var(--open); }
.slot-cell.st-not_ready { border-left: 4px solid var(--notready); opacity: 0.72; }
.slot-cell.st-rejected { border-left: 4px solid var(--rejected); }
.slot-cell .sid { font-weight: 600; color: var(--fg); word-break: break-all; }
.slot-cell .st { text-transform: uppercase; font-size: 11px; margin-top: 3px; }
.slot-cell .meta2 { color: var(--muted); font-size: 11px; margin-top: 4px; line-height: 1.4; }
table { width: 100%; border-collapse: collapse; font: 13px var(--font-mono); }
th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 500; font-size: 12px; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.tag-ok { color: var(--good); font-weight: 600; }
.tag-bad { color: var(--bad); font-weight: 600; }
.tag-warn { color: var(--claimed); font-weight: 600; }
pre {
  background: var(--code-bg); color: #F2EDE4; border: 1px solid var(--line); border-radius: 8px;
  padding: 14px; overflow-x: auto; font: 12.5px/1.55 var(--font-mono); margin: 8px 0 0;
}
.cols-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 18px; }
footer { margin-top: 34px; padding-top: 18px; border-top: 1px solid var(--line); color: var(--muted); font: 12.5px var(--font-mono); }
</style>
<script src="/static/app.js" defer></script>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div class="brand">
      <div class="logo">DS</div>
      <div>
        <h1>draft.bertholomus.com</h1>
        <div class="sub">Trusted-Contributor Label Coordinator · GLM-5.3 DSpine B=16</div>
      </div>
    </div>
    <div class="top-actions">
      <span class="pill"><span class="dot"></span><span id="statusText">Trusted Labels · Hardened</span></span>
      <button class="btn" id="refreshBtn" type="button">Refresh</button>
      <button class="btn" id="themeBtn" type="button">Theme</button>
    </div>
  </header>

  <section class="hero">
    <h2>Owner-Verified Label Coordination for GLM-5.3 DSpine</h2>
    <p>
      Modelled on <a href="https://quant.kindlingai.dev/" target="_blank" rel="noreferrer">Kindling AI's exlr8 quant site</a>
      (jobs, slot grids, atomic claims, readiness gating, verified uploads, contributor credit), this coordinator
      onboards trusted external contributors for <strong><code>drafter-label</code> slots only</strong> on
      <code>kindlingai/glm-5.3-exlr8-k2-k3-k4@9e25c28a</code>. Every external label upload waits at
      <code>uploaded</code> until an owner-side gate teacher-forces a secret HMAC-sampled set of
      8&times;10 = 80 token probes against the reference GLM-5.3 lane. Training, evaluation, capture,
      and packing slots remain owner-only.
    </p>
    <div class="hero-grid">
      <div class="stage-card">
        <div class="k">External · drafter-label</div>
        <div class="t">Same-Model Greedy Responses</div>
        <div class="d">Pre-tokenized permissive prompts (<code>/v1/completions</code> with <code>prompt_ids</code>). Held at <code>uploaded</code> until owner spot-rescoring passes (&ge;85% top-1, &ge;96% top-5, &ge;50% per prompt; canary target &ge;92% / &ge;99%). Relabeled with prefix masking in training.</div>
      </div>
      <div class="stage-card">
        <div class="k">Owner-Only · drafter-train</div>
        <div class="t">Local Capture + SparseLoCo</div>
        <div class="d">Owner fleet only. Local scratch capture (0 B taps uploaded), safe <code>.safetensors</code> deltas gated by owner held-out replay before aggregation.</div>
      </div>
      <div class="stage-card">
        <div class="k">Owner-Only · drafter-eval</div>
        <div class="t">Offline Acceptance Matrix</div>
        <div class="d">Owner fleet only. Offline MXFP8 replay across the 5 evaluated anchor profiles (<code>bit200, bit240, bit280, bit322, bit400</code>).</div>
      </div>
      <div class="stage-card">
        <div class="k">Owner-Only · drafter-capture &amp; pack</div>
        <div class="t">Archive &amp; TP4 Pack</div>
        <div class="d">Owner fleet only. 4-tap (<code>L33/47/61/75</code>) INT8+zstd archive shards and deterministic CPU TP4 packing manifests.</div>
      </div>
    </div>
  </section>

  <div class="tabs">
    <button class="tab active" data-tab="board">Job Board &amp; Slot Grid</button>
    <button class="tab" data-tab="matrix">Eval Matrix &amp; Rounds</button>
    <button class="tab" data-tab="contributors">Contributors &amp; Audit</button>
    <button class="tab" data-tab="api">API &amp; Label Worker</button>
  </div>

  <!-- TAB 1: JOB BOARD & SLOT GRID -->
  <div id="tab-board">
    <section class="panel">
      <div class="head">
        <h3>Active Jobs</h3>
        <div class="meta" id="jobsMeta">Loading…</div>
      </div>
      <div class="body">
        <div class="jobs-grid" id="jobsGrid"></div>
      </div>
    </section>

    <section class="panel">
      <div class="head">
        <h3 id="gridTitle">Slot Grid</h3>
        <div class="meta" id="gridMeta">Select a job above</div>
      </div>
      <div class="body">
        <div class="legend">
          <span><i class="sw" style="background:var(--verified)"></i>verified</span>
          <span><i class="sw" style="background:var(--uploaded)"></i>uploaded (pending owner gate)</span>
          <span><i class="sw" style="background:var(--claimed)"></i>claimed</span>
          <span><i class="sw" style="background:var(--open)"></i>open</span>
          <span><i class="sw" style="background:var(--notready)"></i>not_ready (gated)</span>
          <span><i class="sw" style="background:var(--rejected)"></i>rejected</span>
        </div>
        <div class="slot-grid" id="slotGrid"></div>
      </div>
    </section>
  </div>

  <!-- TAB 2: EVAL MATRIX & TRAINING ROUNDS -->
  <div id="tab-matrix" style="display:none">
    <section class="panel">
      <div class="head">
        <h3>5-Profile Offline Acceptance Matrix (MXFP8 B16 / B8 &tau;)</h3>
        <div class="meta"><code>drafter-eval</code> owner-verified slots · target <code>kindlingai/glm-5.3-exlr8-k2-k3-k4@9e25c28a</code></div>
      </div>
      <div class="body" id="evalMatrixBody">Loading…</div>
    </section>

    <section class="panel">
      <div class="head">
        <h3>Training Rounds &amp; Offline Champion Reference</h3>
        <div class="meta"><code>drafter-train</code> aggregated results vs champion <code>v5-s2-w5@500</code></div>
      </div>
      <div class="body" id="roundsBody">Loading…</div>
    </section>
  </div>

  <!-- TAB 3: CONTRIBUTORS & AUDIT LOG -->
  <div id="tab-contributors" style="display:none">
    <div class="cols-2">
      <section class="panel">
        <div class="head">
          <h3>Contributor Credit</h3>
          <div class="meta">Verified slots only</div>
        </div>
        <div class="body" id="contribBody">Loading…</div>
      </section>
      <section class="panel">
        <div class="head">
          <h3>Verification &amp; Security Audit Log</h3>
          <div class="meta">Redacted public event feed</div>
        </div>
        <div class="body" id="auditBody">Loading…</div>
      </section>
    </div>
  </div>

  <!-- TAB 4: API & WORKER -->
  <div id="tab-api" style="display:none">
    <section class="panel">
      <div class="head">
        <h3>API Reference &amp; External Label Worker</h3>
        <div class="meta">All <code>/api/*</code> requests require <code>X-Quant: 1</code></div>
      </div>
      <div class="body">
        <table>
          <thead><tr><th>Endpoint</th><th>Method</th><th>Auth</th><th>Purpose</th></tr></thead>
          <tbody>
            <tr><td><code>/api/jobs</code></td><td>GET</td><td>Public</td><td>List jobs, slot counts, verified totals, release info</td></tr>
            <tr><td><code>/api/grid?job=&lt;job&gt;&amp;offset=0&amp;limit=500</code></td><td>GET</td><td>Public (+optional Bearer)</td><td>Paginated slot grid, readiness gating, eval matrix, rounds</td></tr>
            <tr><td><code>/api/runners</code> (alias <code>/api/encoders</code>)</td><td>GET</td><td>Public</td><td>Accepted worker runner IDs &amp; container digests</td></tr>
            <tr><td><code>/api/contributors</code></td><td>GET</td><td>Public</td><td>Verified contributor credit &amp; per-kind slot breakdown</td></tr>
            <tr><td><code>/api/audit</code></td><td>GET</td><td>Public</td><td>Redacted recent claim, verification, and rejection events</td></tr>
            <tr><td><code>/api/me</code></td><td>GET</td><td>Bearer key</td><td>Authenticated login, allowed kinds, claim cap, tiers, strikes</td></tr>
            <tr><td><code>/api/claims</code></td><td>GET / POST</td><td>Bearer key</td><td>List or claim open slots (non-owners: <code>drafter-label</code> only, max 2 active claims)</td></tr>
            <tr><td><code>/api/claims/&lt;job&gt;/&lt;slot_id&gt;</code></td><td>DELETE</td><td>Bearer key</td><td>Release an active claim cleanly</td></tr>
            <tr><td><code>/api/artifacts/&lt;artifact_id&gt;</code></td><td>GET</td><td>Bearer key</td><td>Download input shards bound strictly to caller's active claim (<code>X-SHA256</code>)</td></tr>
            <tr><td><code>/api/uploads/direct/&lt;claim_id&gt;</code></td><td>POST</td><td>Bearer key</td><td>Direct upload (&le;12 MiB, mandatory <code>X-Content-SHA256</code>, isolated Phase-A verification)</td></tr>
            <tr><td><code>/api/uploads/init</code> &rarr; <code>PUT .../chunk/{i}</code> &rarr; <code>POST .../complete</code></td><td>POST/PUT</td><td>Bearer key</td><td>Chunked upload (&ge;256 KiB chunk floor, exact chunk sizes, mandatory SHA-256)</td></tr>
            <tr><td><code>/api/admin/*</code></td><td>GET / POST</td><td>Admin Bearer</td><td>Operator credential for job/runner management &amp; owner verification verdicts</td></tr>
          </tbody>
        </table>
<pre># External label contributor quickstart (see LABELS.md for full spec)
export DRAFT_API_KEY="$(cat ~/.config/draft-site/key)"
python3 worker/label_worker.py \
  --url https://draft.bertholomus.com \
  --llm-endpoint http://127.0.0.1:8100 \
  --engine-tag kindling-gr8-bit280 \
  --engine-image-digest sha256:&lt;engine-image-digest&gt; \
  --engine-signature &lt;engine-build-or-signature&gt; \
  --quant-profile bit280 \
  --concurrency 4 \
  --max-slots 1</pre>
      </div>
    </section>
  </div>

  <footer>
    <div>
      <strong>Attribution &amp; Rights:</strong> Target model release
      <code>kindlingai/glm-5.3-exlr8-k2-k3-k4@9e25c28a</code> by <strong>Kindling AI</strong> (Matt Mastracci <code>@mmastrac</code>, Chuck <code>@ck2084</code>, Steve <code>@coffeedev</code>).
      Base model GLM-5.3 by Z.ai. DSpine architecture by Tencent (Apache-2.0); warm-start from
      <code>incoai/GLM-5.3-DFlash2@8f02251d</code> (Apache-2.0, <a href="/NOTICE">NOTICE</a> &amp; <a href="/LICENSE">LICENSE</a>).
      Prompts from public permissive sources (Apache-2.0 / MIT / PSF); zero private traffic; zero CC BY-NC-ND weights.
      Uploaded artifacts are untrusted, verified in isolated non-executing worker processes without pickle, stored outside web root, and never loaded onto serving lanes.
    </div>
  </footer>
</div>
</body>
</html>
"""
