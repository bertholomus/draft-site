"""Web UI for draft.bertholomus.com — modelled on quant.kindlingai.dev."""
from __future__ import annotations


def render_index_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>draft.bertholomus.com · Community Drafter Pilot</title>
<meta name="description" content="Closed-pilot community speculative drafter coordinator for GLM-5.3 DSpine across exlr8 width profiles.">
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
<script>
(function() {
  const saved = localStorage.getItem('drs-theme');
  const pref = saved || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  document.documentElement.dataset.theme = pref;
})();
</script>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div class="brand">
      <div class="logo">DS</div>
      <div>
        <h1>draft.bertholomus.com</h1>
        <div class="sub">Community Speculative Drafter Pilot · GLM-5.3 DSpine B=16</div>
      </div>
    </div>
    <div class="top-actions">
      <span class="pill"><span class="dot"></span><span id="statusText">Closed Pilot · Live</span></span>
      <button class="btn" id="refreshBtn" type="button">Refresh</button>
      <button class="btn" id="themeBtn" type="button">Theme</button>
    </div>
  </header>

  <section class="hero">
    <h2>Distributed Speculative Drafter Coordination for GLM-5.3</h2>
    <p>
      Modelled on <a href="https://quant.kindlingai.dev/" target="_blank" rel="noreferrer">Kindling AI's exlr8 quant site</a>
      (jobs, slot grids, claims, readiness gating, verified uploads, contributor credit), this pilot coordinates
      multi-profile Apache-2.0 <strong>DSpine-Conv B=16</strong> drafter training and evaluation for
      <code>kindlingai/glm-5.3-exlr8-k2-k3-k4@9e25c28a</code> across all 11 <code>mul1</code> width profiles
      (<code>bit200..bit400</code>). Instead of uploading 56 TB of raw hidden-state taps, workers run fused
      train-where-you-capture slots (0 B tap upload, compressed SparseLoCo/INT8 deltas) protected by an owner
      Held-Out Replay + FLTrust Cosine Byzantine Gate.
    </p>
    <div class="hero-grid">
      <div class="stage-card">
        <div class="k">Kind 1 · drafter-label</div>
        <div class="t">Same-Model Responses</div>
        <div class="d">Any GLM-5.3 rig (&ge;92% top-1 / &ge;99.5% top-5 vs exlr8). Permissive prompts only; ~1.2 MB gz upload; teacher-forced relabeling downstream.</div>
      </div>
      <div class="stage-card">
        <div class="k">Kind 2 · drafter-train</div>
        <div class="t">Fused Capture + DiLoCo / Soup</div>
        <div class="d">Local scratch capture (0 B taps uploaded), local AdamW + prefix-masked relabeling, safe <code>.safetensors</code> delta (59–548 MB) gated by Replay + FLTrust.</div>
      </div>
      <div class="stage-card">
        <div class="k">Kind 3 · drafter-eval</div>
        <div class="t">Cross-Profile Offline Matrix</div>
        <div class="d">Deterministic MXFP8 replay across exlr8 profiles (&lt;50 KB JSON upload) with duplicate-slot <code>pred_sha256</code> quorum verification.</div>
      </div>
      <div class="stage-card">
        <div class="k">Kind 4 · drafter-capture</div>
        <div class="t">Optional 4-Tap INT8 Archive</div>
        <div class="d">Standalone compressed 4-tap (<code>L33/47/61/75</code>) INT8 + zstd shards (~3.8&times; smaller at 0.00% &tau; loss) for anchor &amp; eval sets.</div>
      </div>
    </div>
  </section>

  <div class="tabs">
    <button class="tab active" data-tab="board">Job Board &amp; Slot Grid</button>
    <button class="tab" data-tab="matrix">Eval Matrix &amp; Rounds</button>
    <button class="tab" data-tab="contributors">Contributors &amp; Audit</button>
    <button class="tab" data-tab="api">API &amp; Worker CLI</button>
  </div>

  <!-- TAB 1: JOB BOARD & SLOT GRID -->
  <div id="tab-board">
    <section class="panel">
      <div class="head">
        <h3>Active Pilot Jobs</h3>
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
          <span><i class="sw" style="background:var(--uploaded)"></i>uploaded (pending gate)</span>
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
        <h3>Cross-Profile Offline Acceptance Matrix (MXFP8 B16 / B8 &tau;)</h3>
        <div class="meta"><code>drafter-eval</code> verified slots · target <code>kindlingai/glm-5.3-exlr8-k2-k3-k4@9e25c28a</code></div>
      </div>
      <div class="body" id="evalMatrixBody">Loading…</div>
    </section>

    <section class="panel">
      <div class="head">
        <h3>Distributed Training Rounds &amp; Offline Champion Comparison</h3>
        <div class="meta"><code>drafter-train</code> aggregated results vs live champion <code>v5-s2-w5@500</code></div>
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
          <div class="meta">Verified slots only (closed pilot)</div>
        </div>
        <div class="body" id="contribBody">Loading…</div>
      </section>
      <section class="panel">
        <div class="head">
          <h3>Verification &amp; Security Audit Log</h3>
          <div class="meta">Accepted honest uploads &amp; rejected bad uploads</div>
        </div>
        <div class="body" id="auditBody">Loading…</div>
      </section>
    </div>
  </div>

  <!-- TAB 4: API & WORKER -->
  <div id="tab-api" style="display:none">
    <section class="panel">
      <div class="head">
        <h3>API Reference (<code>quant.kindlingai.dev</code> Style)</h3>
        <div class="meta">All <code>/api/*</code> requests require <code>X-Quant: 1</code></div>
      </div>
      <div class="body">
        <table>
          <thead><tr><th>Endpoint</th><th>Method</th><th>Auth</th><th>Purpose</th></tr></thead>
          <tbody>
            <tr><td><code>/api/jobs</code></td><td>GET</td><td>Public</td><td>List active jobs, slot counts, verified totals, release info</td></tr>
            <tr><td><code>/api/grid?job=&lt;job&gt;</code></td><td>GET</td><td>Public (+optional Bearer)</td><td>Full slot grid, readiness gating, eval matrix, training rounds</td></tr>
            <tr><td><code>/api/runners</code> (alias <code>/api/encoders</code>)</td><td>GET</td><td>Public</td><td>Accepted worker runner commit SHAs &amp; container digests</td></tr>
            <tr><td><code>/api/contributors</code></td><td>GET</td><td>Public</td><td>Verified contributor credit &amp; per-kind slot breakdown</td></tr>
            <tr><td><code>/api/audit</code></td><td>GET</td><td>Public</td><td>Recent claims, verifications, rejections, and readiness transitions</td></tr>
            <tr><td><code>/api/me</code></td><td>GET</td><td>Bearer key</td><td>Authenticated contributor login, approved flag, tiers, strikes</td></tr>
            <tr><td><code>/api/claims</code></td><td>GET / POST</td><td>Bearer key</td><td>List held claims or atomically claim an open slot (returns inputs, plan, upload URLs)</td></tr>
            <tr><td><code>/api/claims/&lt;job&gt;/&lt;slot_id&gt;</code></td><td>DELETE</td><td>Bearer key</td><td>Release an active claim cleanly</td></tr>
            <tr><td><code>/api/artifacts/&lt;artifact_id&gt;</code></td><td>GET</td><td>Bearer key</td><td>Stream input shards/manifests with <code>X-SHA256</code> integrity header</td></tr>
            <tr><td><code>/api/uploads/direct/&lt;claim_id&gt;</code></td><td>POST</td><td>Bearer key</td><td>Single-request verified upload (&le;12 MiB: eval JSON, label gz JSONL)</td></tr>
            <tr><td><code>/api/uploads/init</code> &rarr; <code>PUT .../chunk/{i}</code> &rarr; <code>POST .../complete</code></td><td>POST/PUT</td><td>Bearer key</td><td>Resumable 8 MiB chunked upload with per-chunk &amp; full SHA-256 + safe zero-pickle verification</td></tr>
            <tr><td><code>/api/admin/*</code></td><td>GET / POST</td><td>Admin Bearer</td><td>Separate operator credential for job/runner management &amp; GPU verification verdicts</td></tr>
          </tbody>
        </table>
<pre># Example: inspect public jobs and run worker client via public URL
curl -sS -H "X-Quant: 1" https://draft.bertholomus.com/api/jobs | jq .
python3 worker/draft_worker.py status --url https://draft.bertholomus.com --key-file ~/.config/draft-site/key
python3 worker/draft_worker.py run --url https://draft.bertholomus.com --key-file ~/.config/draft-site/key --job glm53-dspine-eval</pre>
      </div>
    </section>
  </div>

  <footer>
    <div>
      <strong>Attribution &amp; Rights:</strong> Target model release
      <code>kindlingai/glm-5.3-exlr8-k2-k3-k4@9e25c28a</code> by <strong>Kindling AI</strong> (Matt Mastracci &amp; Chuck).
      Base model GLM-5.3 by Z.ai. DSpine architecture by Tencent (Apache-2.0); warm-start from
      <code>incoai/GLM-5.3-DFlash2@8f02251d</code> (Apache-2.0, <a href="/NOTICE">NOTICE</a> preserved).
      Prompts strictly from public permissive sources (Apache-2.0 / MIT / PSF); zero private traffic; zero CC BY-NC-ND weights.
      Uploaded artifacts are untrusted, parsed without pickle/execution, stored outside web root, and never loaded onto serving lanes.
    </div>
  </footer>
</div>

<script>
const H = { 'X-Quant': '1' };
let JOBS = [];
let ACTIVE_JOB = null;

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function fmtBytes(b) {
  if (!b) return '0 B';
  const u = ['B','KB','MB','GB'];
  let i = 0, v = Number(b);
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return (i === 0 ? v : v.toFixed(2)) + ' ' + u[i];
}
function fmtTime(ms) {
  if (!ms) return '—';
  return new Date(ms).toISOString().replace('T', ' ').slice(0, 19) + 'Z';
}

document.getElementById('themeBtn').onclick = () => {
  const cur = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
  document.documentElement.dataset.theme = cur;
  localStorage.setItem('drs-theme', cur);
};

document.querySelectorAll('.tab').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const t = btn.dataset.tab;
    ['board','matrix','contributors','api'].forEach(id => {
      document.getElementById('tab-' + id).style.display = (id === t) ? '' : 'none';
    });
  };
});

async function apiGet(path) {
  const r = await fetch(path, { headers: H });
  if (!r.ok) throw new Error(`${path} -> HTTP ${r.status}`);
  return await r.json();
}

function renderJobs() {
  const g = document.getElementById('jobsGrid');
  const totSlots = JOBS.reduce((a, j) => a + j.slots, 0);
  const totVer = JOBS.reduce((a, j) => a + j.verified, 0);
  document.getElementById('jobsMeta').textContent = `${JOBS.length} jobs · ${totVer}/${totSlots} slots verified`;
  g.innerHTML = JOBS.map(j => {
    const tot = Math.max(1, j.slots);
    const wVer = (100 * j.verified / tot).toFixed(1);
    const wPend = (100 * j.pending_verify / tot).toFixed(1);
    const wClm = (100 * j.claimed / tot).toFixed(1);
    const wOpen = (100 * j.open / tot).toFixed(1);
    const sel = j.job === ACTIVE_JOB ? ' selected' : '';
    return `<div class="job-card${sel}" onclick="selectJob('${esc(j.job)}')">
      <div class="kind">${esc(j.job_kind)} · round ${j.active_round}</div>
      <div class="title">${esc(j.title)}</div>
      <div class="bar">
        <span style="width:${wVer}%;background:var(--verified)"></span>
        <span style="width:${wPend}%;background:var(--uploaded)"></span>
        <span style="width:${wClm}%;background:var(--claimed)"></span>
        <span style="width:${wOpen}%;background:var(--open)"></span>
      </div>
      <div class="stats-row">
        <span><strong>${j.verified}</strong>/${j.slots} verified</span>
        <span>${j.open} open</span>
        <span>${j.claimed} claimed</span>
        <span>${j.not_ready} gated</span>
        <span>${j.contributors} contrib</span>
      </div>
    </div>`;
  }).join('');
}

async function selectJob(jobId) {
  ACTIVE_JOB = jobId;
  renderJobs();
  const data = await apiGet('/api/grid?job=' + encodeURIComponent(jobId));
  document.getElementById('gridTitle').textContent = `${data.title} (${data.job})`;
  document.getElementById('gridMeta').textContent = `${data.job_kind} · ${data.release}@${data.revision.slice(0,8)} · ${data.slots.length} slots`;
  const sg = document.getElementById('slotGrid');
  sg.innerHTML = data.slots.map(s => {
    const m = s.metrics || {};
    let mline = '';
    if (m.b16_ref_tau != null) mline = `B16=${m.b16_ref_tau} B8=${m.b8_ref_tau} ce8=${m.ce8}`;
    else if (m.owner_gate && m.owner_gate.delta_ce8 != null) mline = `Δce8=${m.owner_gate.delta_ce8} cos=${m.owner_gate.cos_vs_root}`;
    else if (m.server_delta_l2_norm != null) mline = `L2=${m.server_delta_l2_norm} (${fmtBytes(s.upload_bytes)})`;
    else if (m.n_rows != null) mline = `${m.n_rows} rows · ${m.total_gen_tokens || m.n_tokens || 0} tok`;
    else if (s.depends_on && s.state === 'not_ready') mline = `awaits ${s.depends_on}`;
    const by = s.contributor_login ? `by ${esc(s.contributor_login)}${s.host_tag ? ' @ '+esc(s.host_tag) : ''}` : '';
    return `<div class="slot-cell st-${esc(s.state)}" title="${esc(s.verify_reason || s.state)}">
      <div class="sid">${esc(s.slot_id)}</div>
      <div class="st">${esc(s.state)} · ${esc(s.profile || s.col)}</div>
      <div class="meta2">${by}${by && mline ? '<br>' : ''}${esc(mline)}${s.verify_status ? '<br>gate: ' + esc(s.verify_status) : ''}</div>
    </div>`;
  }).join('');
}

async function renderMatrixAndRounds() {
  const evalJob = JOBS.find(j => j.job_kind === 'drafter-eval');
  const trainJob = JOBS.find(j => j.job_kind === 'drafter-train');
  if (evalJob) {
    const g = await apiGet('/api/grid?job=' + encodeURIComponent(evalJob.job));
    const profs = ['bit200','bit240','bit280','bit322','bit400'];
    const ckpts = Object.keys(g.eval_matrix || {});
    if (!ckpts.length) {
      document.getElementById('evalMatrixBody').innerHTML = '<div class="meta">No verified eval slots yet.</div>';
    } else {
      let html = '<table><thead><tr><th>Checkpoint</th>' + profs.map(p => `<th class="num">${p} (B16 / B8)</th>`).join('') + '<th class="num">5-Prof Mean B16</th><th>Contributors</th></tr></thead><tbody>';
      for (const ck of ckpts) {
        const row = g.eval_matrix[ck];
        const b16s = [];
        const cs = new Set();
        const cells = profs.map(p => {
          const v = row[p];
          if (!v) return '<td class="num" style="color:var(--muted)">—</td>';
          if (v.b16_ref_tau != null) b16s.push(Number(v.b16_ref_tau));
          if (v.contributor_login) cs.add(v.contributor_login);
          return `<td class="num"><strong>${Number(v.b16_ref_tau).toFixed(3)}</strong> <span style="color:var(--muted)">/ ${Number(v.b8_ref_tau).toFixed(3)}</span></td>`;
        }).join('');
        const mean = b16s.length ? (b16s.reduce((a,b)=>a+b,0)/b16s.length).toFixed(4) : '—';
        html += `<tr><td><code>${esc(ck)}</code></td>${cells}<td class="num"><strong>${mean}</strong></td><td>${esc([...cs].join(', '))}</td></tr>`;
      }
      html += '</tbody></table>';
      document.getElementById('evalMatrixBody').innerHTML = html;
    }
  }
  if (trainJob) {
    const g = await apiGet('/api/grid?job=' + encodeURIComponent(trainJob.job));
    const rds = g.rounds || [];
    if (!rds.length) {
      document.getElementById('roundsBody').innerHTML = '<div class="meta">No training rounds registered.</div>';
    } else {
      let html = '<table><thead><tr><th>Round</th><th>Base Ckpt</th><th>Status</th><th>Agg Method</th><th class="num">MP 5-Prof B16</th><th class="num">14-Dump B16 / roll16</th><th>Champion Reference</th></tr></thead><tbody>';
      for (const r of rds) {
        const m = r.agg_metrics || {};
        const mp = m.mp_mean_b16 != null ? Number(m.mp_mean_b16).toFixed(4) : '—';
        const d14 = (m.sim14_b16 != null) ? `${Number(m.sim14_b16).toFixed(4)} / ${Number(m.sim14_roll16 || 0).toFixed(4)}` : '—';
        const champ = m.champion_ref ? `MP ${m.champion_ref.mp_mean_b16} · 14d ${m.champion_ref.sim14_b16}` : 'v5-s2-w5@500 (MP 5.899 · 14d 5.750)';
        html += `<tr><td>R${r.round}</td><td><code>${esc(r.base_ckpt_name)}</code></td><td>${esc(r.status)}</td><td>${esc(r.agg_method || '—')}</td><td class="num"><strong>${mp}</strong></td><td class="num">${d14}</td><td style="color:var(--muted)">${esc(champ)}</td></tr>`;
      }
      html += '</tbody></table>';
      document.getElementById('roundsBody').innerHTML = html;
    }
  }
}

async function renderContribsAndAudit() {
  const [contribs, audit] = await Promise.all([apiGet('/api/contributors'), apiGet('/api/audit?limit=40')]);
  document.getElementById('contribBody').innerHTML = '<table><thead><tr><th>Login</th><th class="num">Verified</th><th class="num">Rejected</th><th>By Kind</th><th class="num">Uploaded</th></tr></thead><tbody>' +
    contribs.map(c => {
      const bk = Object.entries(c.by_kind || {}).map(([k,v]) => `${k.replace('drafter-','')}:${v}`).join(' ');
      return `<tr><td><code>${esc(c.login)}</code></td><td class="num tag-ok">${c.verified_slots}</td><td class="num ${c.rejected_attempts ? 'tag-bad' : ''}">${c.rejected_attempts}</td><td>${esc(bk || '—')}</td><td class="num">${fmtBytes(c.total_bytes_uploaded)}</td></tr>`;
    }).join('') + '</tbody></table>';

  document.getElementById('auditBody').innerHTML = '<table><thead><tr><th>UTC</th><th>Actor</th><th>Action</th><th>Slot</th><th>Status</th></tr></thead><tbody>' +
    audit.map(a => {
      const bad = /reject|fail/i.test(a.action + ' ' + a.status);
      const ok = /pass|accept|aggregat|opened/i.test(a.action + ' ' + a.status);
      const cls = bad ? 'tag-bad' : (ok ? 'tag-ok' : '');
      const reason = a.detail && a.detail.reason ? ` — ${a.detail.reason}` : '';
      return `<tr><td>${esc(a.utc.slice(11,19))}</td><td><code>${esc(a.actor)}</code></td><td>${esc(a.action)}</td><td>${esc((a.slot_id || a.job || '—'))}</td><td class="${cls}" title="${esc(reason)}">${esc(a.status)}</td></tr>`;
    }).join('') + '</tbody></table>';
}

async function refreshAll() {
  try {
    JOBS = await apiGet('/api/jobs');
    if (!ACTIVE_JOB && JOBS.length) ACTIVE_JOB = JOBS[0].job;
    renderJobs();
    if (ACTIVE_JOB) await selectJob(ACTIVE_JOB);
    await Promise.all([renderMatrixAndRounds(), renderContribsAndAudit()]);
  } catch (e) {
    document.getElementById('jobsMeta').textContent = 'Error loading API: ' + e.message;
  }
}
document.getElementById('refreshBtn').onclick = refreshAll;
refreshAll();
</script>
</body>
</html>
"""
