"""Web UI for draft.bertholomus.com — Campfire-aligned vocabulary, hardened CSP (no inline scripts), GitHub OAuth login, self-service API keys, and owner admin view."""
from __future__ import annotations

import html

COMMON_STYLE = r"""
:root { color-scheme: dark;
  --ink: #16130F; --paper: #F2EDE4;
  --bg: var(--ink); --fg: var(--paper);
  --accent: #FF6B1A; --accent-soft: #FFC24A; --accent-text: #FF8139;
  --surface: #1E1914; --surface-2: #28211A; --line: #342E26; --muted: #A39A8C;
  --rim: inset 0 1px 0 rgb(255 246 227 / 0.05);
  --lift: 0 1px 0 rgb(0 0 0 / 0.35), 0 18px 40px -22px rgb(0 0 0 / 0.8);
  --code-bg: #0E0C09; --good: #4CD97B; --bad: #FF5C7A;
  --notready: #29241D; --closed: #473E33; --open: #5E5040; --claimed: #FFC24A; --uploaded: #E5A030; --verified: #FF6B1A; --rejected: #FF5C7A; --disabled: #3B3229;
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
  --notready: #EAE3D7; --closed: #D2C7B5; --open: #C9B391; --claimed: #FFB020; --uploaded: #D97706; --verified: #E8590C; --rejected: #C81E4A; --disabled: #DCCFB9;
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
.pill .dot-warn { width: 7px; height: 7px; border-radius: 50%; background: var(--claimed); }
.btn {
  display: inline-flex; align-items: center; gap: 6px;
  border: 1px solid var(--line); background: var(--surface); color: var(--fg);
  padding: 6px 12px; border-radius: 8px; font: 500 13px var(--font-mono); cursor: pointer;
  text-decoration: none;
}
.btn:hover { border-color: var(--accent); color: var(--accent-text); text-decoration: none; }
.btn:disabled { opacity: 0.45; cursor: not-allowed; }
.btn-accent { border-color: var(--accent); background: var(--accent); color: #16130F; font-weight: 600; }
.btn-accent:hover { opacity: 0.92; color: #16130F; border-color: var(--accent-soft); }
.btn-danger { border-color: var(--bad); color: var(--bad); }
.btn-danger:hover { background: var(--bad); color: #16130F; }
.btn-sm { padding: 4px 9px; font-size: 12px; }
.input {
  border: 1px solid var(--line); background: var(--code-bg); color: #F2EDE4;
  padding: 7px 10px; border-radius: 8px; font: 13px var(--font-mono); min-width: 220px;
}
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
.notice-box {
  background: var(--surface-2); border: 1px solid var(--line); border-radius: 10px;
  padding: 16px 18px; margin-bottom: 16px;
}
.key-reveal {
  background: var(--code-bg); color: #F2EDE4; border: 1px solid var(--accent); border-radius: 10px;
  padding: 16px 18px; margin-bottom: 18px;
}
.key-reveal .kr-title { color: var(--accent-soft); font: 600 13px var(--font-mono); margin-bottom: 8px; }
.key-reveal .kr-val {
  display: block; padding: 10px 12px; border-radius: 8px; background: #1B1712; border: 1px solid var(--line);
  font: 600 13px var(--font-mono); word-break: break-all; color: var(--good); margin-bottom: 10px;
}
.row-actions { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 999px; font: 600 11px var(--font-mono);
  border: 1px solid var(--line); background: var(--surface-2); color: var(--muted);
}
.badge-ok { border-color: var(--good); color: var(--good); }
.badge-warn { border-color: var(--claimed); color: var(--claimed); }
.badge-bad { border-color: var(--bad); color: var(--bad); }
.jobs-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(275px, 1fr)); gap: 14px; }
.job-card {
  background: var(--surface-2); border: 1px solid var(--line); border-radius: 10px; padding: 14px;
  cursor: pointer; transition: border-color 0.15s;
}
.job-card:hover, .job-card.selected { border-color: var(--accent); }
.job-card .kind { font: 600 11px var(--font-mono); color: var(--accent-text); text-transform: uppercase; display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.job-card .title { font-weight: 600; margin: 4px 0 6px; font-size: 15px; }
.bar { height: 8px; border-radius: 999px; background: var(--notready); overflow: hidden; display: flex; margin: 8px 0; }
.bar > span { display: block; height: 100%; }
.stats-row { display: flex; gap: 10px; flex-wrap: wrap; font: 12px var(--font-mono); color: var(--muted); }
.legend { display: flex; gap: 12px; flex-wrap: wrap; font: 12px var(--font-mono); color: var(--muted); margin-bottom: 14px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: middle; }
.slot-grid { display: flex; flex-direction: column; gap: 3px; overflow-x: auto; }
.slot-row { display: flex; align-items: center; gap: 8px; }
.slot-row .rlabel { flex: 0 0 15ch; font: 11px var(--font-mono); color: var(--muted); text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.slot-row .tiles { display: flex; flex-wrap: wrap; gap: 2px; }
.slot-cell { display: block; width: 13px; height: 13px; border-radius: 2px; background: var(--notready); cursor: pointer; }
.slot-cell.st-verified { background: var(--verified); }
.slot-cell.st-uploaded { background: var(--uploaded); }
.slot-cell.st-claimed { background: var(--claimed); }
.slot-cell.st-open { background: var(--open); }
.slot-cell.st-closed { background: var(--closed); }
.slot-cell.st-not_ready { background: var(--notready); outline: 1px solid var(--line); outline-offset: -1px; }
.slot-cell.st-disabled { background: var(--disabled); }
.slot-cell.st-rejected { background: var(--rejected); }
.slot-cell.mine { box-shadow: inset 0 0 0 1.5px var(--mine); }
.slot-cell:hover, .slot-cell.sel { outline: 2px solid var(--fg); outline-offset: 0; }
.slot-more { font: 11px var(--font-mono); color: var(--muted); margin-top: 6px; }
#slotSel { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: baseline; margin-top: 12px; font: 12px var(--font-mono); min-height: 18px; }
#slotSel .muted { color: var(--muted); }
#slotTip { position: fixed; pointer-events: none; display: none; white-space: nowrap; z-index: 40; padding: 4px 9px; border-radius: 6px;
  background: var(--surface); border: 1px solid var(--line); box-shadow: var(--lift); font: 12px var(--font-mono); color: var(--fg); }
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
"""


def render_index_html(*, initial_job: str | None = None) -> str:
    esc_init = html.escape(initial_job or "", quote=True)
    init_attr = f' data-initial-job="{esc_init}"' if initial_job else ""
    page_title = (
        f"{esc_init} · draft.bertholomus.com" if initial_job else "draft.bertholomus.com · Volunteer Drafter Label Projects"
    )
    return (
        """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>"""
        + page_title
        + """</title>
<meta name="description" content="Campfire-compatible volunteer drafter-label project coordinator for GLM-5.3.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>"""
        + COMMON_STYLE
        + """</style>
<script src="/static/app.js" defer></script>
</head>
<body data-page="index\""""
        + init_attr
        + """>
<div class="wrap">
  <header class="top">
    <div class="brand">
      <div class="logo">DR</div>
      <div>
        <h1><a href="/" style="color:inherit;text-decoration:none">draft.bertholomus.com</a></h1>
        <div class="sub">Campfire-Compatible Drafter Label Coordinator · GLM-5.3</div>
      </div>
    </div>
    <div class="top-actions">
      <a class="btn" href="/skill.md">/skill.md</a>
      <span class="pill"><span class="dot"></span><span id="statusText">Campfire-Aligned · Verified Labels</span></span>
      <span id="authActions"><a class="btn btn-accent" href="/auth/login">Log in with GitHub</a></span>
      <button class="btn" id="refreshBtn" type="button">Refresh</button>
      <button class="btn" id="themeBtn" type="button">Theme</button>
    </div>
  </header>

  <section class="hero">
    <h2>Volunteer Drafter Projects for GLM-5.3</h2>
    <p>
      Built to match <a href="https://campfire.kindlingai.dev/" target="_blank" rel="noreferrer">Campfire by Kindling AI</a>
      (projects, slot states, GitHub sign-in, self-service API keys, atomic claims, readiness gating, and
      <a href="/skill.md">/skill.md</a> agent onboarding) so contributors can transition cleanly when drafter projects
      are hosted directly on Campfire. Approved external contributors claim <strong><code>drafter-label</code> slots</strong>
      on <code>kindlingai/glm-5.3-exlr8-k2-k3-k4@9e25c28a</code>; every upload passes a fast structural/SHA-256 check
      and waits at <code>uploaded</code> until an owner-side gate runs an unguessable 8&times;10 = 80 teacher-forced
      spot check against the reference GLM-5.3 lane.
    </p>
    <div class="hero-grid">
      <div class="stage-card">
        <div class="k">Live · drafter-label</div>
        <div class="t">Same-Model Greedy Labels</div>
        <div class="d">Pre-tokenized permissive public prompts via local <code>/v1/completions</code>. Held at <code>uploaded</code> until owner spot-rescoring passes (&ge;85% top-1, &ge;96% top-5, &ge;50% per prompt; canary target &ge;92% / &ge;99%).</div>
      </div>
      <div class="stage-card">
        <div class="k">Planned · drafter-eval</div>
        <div class="t">Cross-Profile Eval Matrix</div>
        <div class="d">Small deterministic JSON eval summaries verified by 2-contributor distinct-login duplicate quorum.</div>
      </div>
      <div class="stage-card">
        <div class="k">Optional · drafter-capture</div>
        <div class="t">Compressed Tap Shards</div>
        <div class="d">Optional chunked <code>.safetensors</code> capture uploads gated by zero-pickle header, shape, and finiteness checks.</div>
      </div>
    </div>
  </section>

  <div class="tabs">
    <button class="tab active" data-tab="board" type="button">Projects &amp; Slot Grid</button>
    <button class="tab" data-tab="keys" type="button">API Keys</button>
    <button class="tab" data-tab="admin" id="adminTabBtn" type="button" style="display:none">Admin · Access Requests</button>
    <button class="tab" data-tab="contributors" type="button">Contributors &amp; Audit</button>
    <button class="tab" data-tab="api" type="button">API &amp; /skill.md</button>
  </div>

  <!-- TAB 1: PROJECTS & SLOT GRID -->
  <div id="tab-board">
    <section class="panel" id="drafterProgressPanel" style="display:none">
      <div class="head">
        <h3>Drafter Progress</h3>
        <div class="meta" id="drafterProgressMeta"></div>
      </div>
      <div class="body" id="drafterProgressBody"></div>
    </section>

    <section class="panel">
      <div class="head">
        <h3>Active Projects</h3>
        <div class="meta" id="jobsMeta">Loading…</div>
      </div>
      <div class="body">
        <div class="jobs-grid" id="jobsGrid"></div>
      </div>
    </section>

    <section class="panel">
      <div class="head">
        <h3 id="gridTitle">Slot Grid</h3>
        <div class="meta" id="gridMeta">Select a project above</div>
      </div>
      <div class="body">
        <div class="legend">
          <span><i class="sw" style="background:var(--verified)"></i>verified</span>
          <span><i class="sw" style="background:var(--uploaded)"></i>uploaded (pending owner gate)</span>
          <span><i class="sw" style="background:var(--claimed)"></i>claimed</span>
          <span><i class="sw" style="background:var(--open)"></i>open</span>
          <span><i class="sw" style="background:var(--closed)"></i>closed (inputs up, claims off)</span>
          <span><i class="sw" style="background:var(--disabled)"></i>disabled</span>
          <span><i class="sw" style="background:var(--notready)"></i>not ready</span>
          <span><i class="sw" style="background:var(--rejected)"></i>rejected</span>
        </div>
        <div class="slot-grid" id="slotGrid"></div>
        <div id="slotSel"></div>
        <div id="slotTip"></div>
      </div>
    </section>
  </div>

  <!-- TAB: SELF-SERVICE API KEYS -->
  <div id="tab-keys" style="display:none">
    <section class="panel">
      <div class="head">
        <h3>Self-Service Worker API Keys</h3>
        <div class="meta" id="keysMeta">Sign in with GitHub to manage keys</div>
      </div>
      <div class="body">
        <div id="keysAccountBar" class="notice-box">Loading session…</div>
        <div id="keyRevealBox" class="key-reveal" style="display:none">
          <div class="kr-title">New API key (shown once — save it now; it cannot be viewed again)</div>
          <code class="kr-val" id="keyRevealValue"></code>
          <div class="row-actions">
            <button class="btn btn-accent btn-sm" id="copyKeyBtn" type="button">Copy key</button>
            <button class="btn btn-sm" id="dismissKeyBtn" type="button">I saved it · Clear from screen</button>
          </div>
          <pre id="keySaveHint"></pre>
        </div>
        <div id="keysCreateBox" class="notice-box" style="display:none">
          <div class="row-actions">
            <input class="input" id="newKeyLabel" type="text" maxlength="64" placeholder="Key label (e.g. gr8-rig-1)">
            <button class="btn btn-accent" id="createKeyBtn" type="button">Create API Key</button>
            <span class="meta" id="keysCapHint"></span>
          </div>
        </div>
        <div id="keysError" class="tag-bad" style="margin-bottom:10px;display:none"></div>
        <div id="keysBody"></div>
      </div>
    </section>
  </div>

  <!-- TAB: OWNER ADMIN ACCESS REQUESTS -->
  <div id="tab-admin" style="display:none">
    <section class="panel">
      <div class="head">
        <h3>GitHub Access Requests &amp; Allow-List</h3>
        <div class="meta" id="adminMeta">Owner only</div>
      </div>
      <div class="body">
        <div id="adminError" class="tag-bad" style="margin-bottom:10px;display:none"></div>
        <div id="adminBody">Loading…</div>
      </div>
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
        <h3>Campfire-Compatible API &amp; Label Worker</h3>
        <div class="meta">All <code>/api/*</code> requests require <code>X-Quant: 1</code> · Agent setup: <a href="/skill.md">/skill.md</a></div>
      </div>
      <div class="body">
        <table>
          <thead><tr><th>Endpoint</th><th>Method</th><th>Auth</th><th>Purpose</th></tr></thead>
          <tbody>
            <tr><td><code>/skill.md</code></td><td>GET</td><td>Public</td><td>Campfire-format agent onboarding guide for <code>drafter-label</code> workers</td></tr>
            <tr><td><code>/jobs/&lt;job&gt;</code></td><td>GET</td><td>Public</td><td>Direct project view URL (matches Campfire's <code>/jobs/&lt;job&gt;</code> route)</td></tr>
            <tr><td><code>/auth/login</code> &rarr; <code>/auth/callback</code></td><td>GET</td><td>Browser</td><td>Log in with GitHub (read-only identity, allow-list gated)</td></tr>
            <tr><td><code>/api/keys</code>, <code>.../rotate</code>, <code>DELETE /api/keys/&lt;id&gt;</code></td><td>GET/POST/DELETE</td><td>Session + CSRF</td><td>Self-service worker API keys (max 2 live keys, shown once)</td></tr>
            <tr><td><code>/api/jobs[?kind=...]</code></td><td>GET</td><td>Public</td><td>List projects with Campfire-aligned fields (<code>kind</code>, <code>open</code>, <code>closed</code>, <code>claimed</code>, <code>uploaded</code>, <code>verified</code>, <code>weights</code>)</td></tr>
            <tr><td><code>/api/grid?job=&lt;job&gt;&amp;offset=0&amp;limit=500</code></td><td>GET</td><td>Public (+optional Bearer)</td><td>Paginated slot grid and readiness state</td></tr>
            <tr><td><code>/api/runners[?kind=...]</code> (alias <code>/api/encoders</code>)</td><td>GET</td><td>Public</td><td>Accepted worker runner IDs &amp; digests</td></tr>
            <tr><td><code>/api/contributors</code></td><td>GET</td><td>Public</td><td>Verified contributor credit &amp; per-kind slot breakdown</td></tr>
            <tr><td><code>/api/audit</code></td><td>GET</td><td>Public</td><td>Redacted recent claim, verification, and rejection events</td></tr>
            <tr><td><code>/api/me</code></td><td>GET</td><td>Bearer key</td><td>Authenticated login, role, allowed kinds, claim cap, tiers, strikes</td></tr>
            <tr><td><code>/api/claims[?kind=...]</code></td><td>GET / POST / DELETE</td><td>Bearer key</td><td>List, claim, or release all active claims (external: <code>drafter-label</code> only, max 2 active)</td></tr>
            <tr><td><code>/api/claims/&lt;job&gt;/&lt;slot_id&gt;</code></td><td>DELETE</td><td>Bearer key</td><td>Release one active claim cleanly</td></tr>
            <tr><td><code>/api/artifacts/&lt;artifact_id&gt;</code></td><td>GET</td><td>Bearer key</td><td>Download input shards bound strictly to caller's active claim (<code>X-SHA256</code>)</td></tr>
            <tr><td><code>/api/uploads/direct/&lt;claim_id&gt;</code></td><td>POST</td><td>Bearer key</td><td>Direct upload (&le;12 MiB, mandatory <code>X-Content-SHA256</code>, isolated Phase-A verification)</td></tr>
            <tr><td><code>/api/uploads/init</code> &rarr; <code>PUT .../chunk/{i}</code> &rarr; <code>POST .../complete</code></td><td>POST/PUT</td><td>Bearer key</td><td>Chunked upload (&ge;256 KiB chunk floor, exact chunk sizes, mandatory SHA-256)</td></tr>
            <tr><td><code>/api/admin/*</code></td><td>GET / POST</td><td>Admin Bearer</td><td>Operator credential for project/runner management, access approvals &amp; owner verification verdicts</td></tr>
          </tbody>
        </table>
<pre># 1. Click "Log in with GitHub" above, open the "API Keys" tab, and create a key.
# 2. Save it with mode 0600:
mkdir -p ~/.config/draft-site
printf '%s\n' 'drs_wrk_PASTE_YOUR_KEY_HERE' &gt; ~/.config/draft-site/key
chmod 600 ~/.config/draft-site/key

# 3. Run one test slot first (see /skill.md, CONTRIBUTING.md, and LABELS.md):
python3 worker/label_worker.py \
  --url https://draft.bertholomus.com \
  --key-file ~/.config/draft-site/key \
  --llm-endpoint http://127.0.0.1:8100 \
  --engine-tag kindling-gr8-bit280 \
  --engine-image-digest ghcr.io/kindlingai/gr8@sha256:&lt;digest&gt; \
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
      Base model GLM-5.3 by Z.ai. Coordinator &amp; label worker licensed under Apache-2.0 (<a href="/NOTICE">NOTICE</a> &amp; <a href="/LICENSE">LICENSE</a>).
      Prompts come from public permissive sources (Apache-2.0 / MIT / PSF); zero private traffic.
      Uploaded artifacts are untrusted, verified in isolated non-executing worker processes without pickle, stored outside web root, and never loaded onto serving lanes.
    </div>
  </footer>
</div>
</body>
</html>
"""
    )


def render_waiting_html(
    *,
    github_login: str,
    github_name: str | None,
    status: str,
    csrf_token: str,
) -> str:
    esc_login = html.escape(github_login, quote=True)
    esc_name = html.escape(github_name or "", quote=True)
    esc_csrf = html.escape(csrf_token, quote=True)
    name_part = f" ({esc_name})" if esc_name else ""
    if status == "denied":
        title = "Access denied"
        msg = (
            f"GitHub account <code>@{esc_login}</code>{name_part} is not approved to mint API keys "
            "or claim slots on draft.bertholomus.com."
        )
    elif status == "quarantined":
        title = "Account quarantined"
        msg = (
            f"GitHub account <code>@{esc_login}</code>{name_part} is currently quarantined. "
            "Contact the site owner to review your account before minting keys or claiming slots."
        )
    else:
        title = "Waiting for approval"
        msg = (
            f"Signed in with GitHub as <code>@{esc_login}</code>{name_part}. "
            "Your access request is pending owner approval. Until approved, your account has no API keys "
            "and cannot claim slots."
        )
    return (
        """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>"""
        + html.escape(title, quote=True)
        + """ · draft.bertholomus.com</title>
<style>"""
        + COMMON_STYLE
        + """</style>
</head>
<body>
<div class="wrap" style="max-width:680px;padding-top:48px">
  <section class="panel">
    <div class="head">
      <h3>"""
        + html.escape(title, quote=True)
        + """</h3>
      <div class="meta">draft.bertholomus.com</div>
    </div>
    <div class="body">
      <p>"""
        + msg
        + """</p>
      <div class="row-actions" style="margin-top:18px">
        <a class="btn" href="/">Back to projects</a>
        <form method="post" action="/auth/logout" style="margin:0">
          <input type="hidden" name="csrf_token" value=\""""
        + esc_csrf
        + """">
          <button class="btn btn-danger" type="submit">Log out</button>
        </form>
      </div>
    </div>
  </section>
</div>
</body>
</html>
"""
    )


def render_admin_html() -> str:
    return (
        """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Admin · Access Requests · draft.bertholomus.com</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>"""
        + COMMON_STYLE
        + """</style>
<script src="/static/app.js" defer></script>
</head>
<body data-page="admin">
<div class="wrap">
  <header class="top">
    <div class="brand">
      <div class="logo">DR</div>
      <div>
        <h1>draft.bertholomus.com · Admin</h1>
        <div class="sub">Owner Allow-List &amp; Pending GitHub Requests</div>
      </div>
    </div>
    <div class="top-actions">
      <a class="btn" href="/">Projects</a>
      <span id="authActions"></span>
      <button class="btn" id="refreshBtn" type="button">Refresh</button>
      <button class="btn" id="themeBtn" type="button">Theme</button>
    </div>
  </header>

  <section class="panel">
    <div class="head">
      <h3>GitHub Access Requests &amp; Allow-List</h3>
      <div class="meta" id="adminMeta">Loading…</div>
    </div>
    <div class="body">
      <div id="adminError" class="tag-bad" style="margin-bottom:10px;display:none"></div>
      <div id="adminBody">Loading…</div>
    </div>
  </section>
</div>
</body>
</html>
"""
    )


def render_auth_error_html(title: str, message: str) -> str:
    esc_title = html.escape(title, quote=True)
    esc_msg = html.escape(message, quote=True)
    return (
        """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>"""
        + esc_title
        + """ · draft.bertholomus.com</title>
<style>"""
        + COMMON_STYLE
        + """</style>
</head>
<body>
<div class="wrap" style="max-width:680px;padding-top:48px">
  <section class="panel">
    <div class="head">
      <h3>"""
        + esc_title
        + """</h3>
      <div class="meta">draft.bertholomus.com</div>
    </div>
    <div class="body">
      <p>"""
        + esc_msg
        + """</p>
      <div class="row-actions" style="margin-top:18px">
        <a class="btn btn-accent" href="/auth/login">Try Log in with GitHub again</a>
        <a class="btn" href="/">Back to projects</a>
      </div>
    </div>
  </section>
</div>
</body>
</html>
"""
    )
