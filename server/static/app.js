'use strict';

(function initTheme() {
  const saved = localStorage.getItem('drs-theme');
  const pref = saved || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  document.documentElement.dataset.theme = pref;
})();

const H = { 'X-Quant': '1' };
let JOBS = [];
let ACTIVE_JOB = null;
let ME = null;

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[c]));
}

function fmtBytes(b) {
  if (!b) return '0 B';
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let v = Number(b);
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i++;
  }
  return (i === 0 ? v : v.toFixed(2)) + ' ' + u[i];
}

function fmtUtc(iso) {
  if (!iso) return 'never';
  return String(iso).replace('T', ' ').replace('Z', ' UTC');
}

async function apiGet(path) {
  const r = await fetch(path, { headers: H, credentials: 'same-origin', cache: 'no-store' });
  if (!r.ok) {
    let msg = `${path} -> HTTP ${r.status}`;
    try {
      const d = await r.json();
      if (d && (d.detail || d.error)) msg = String(d.detail || d.error);
    } catch (_) {}
    throw new Error(msg);
  }
  return await r.json();
}

async function sessionMutate(method, path, body) {
  if (!ME || !ME.csrf_token) throw new Error('Not signed in');
  const headers = {
    'X-Quant': '1',
    'X-CSRF-Token': ME.csrf_token,
  };
  const opts = { method, headers, credentials: 'same-origin', cache: 'no-store' };
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  const text = await r.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch (_) {}
  if (!r.ok) {
    throw new Error(data.detail || data.error || `HTTP ${r.status}`);
  }
  return data;
}

function showKeyReveal(rawKey) {
  const box = document.getElementById('keyRevealBox');
  const valEl = document.getElementById('keyRevealValue');
  const hintEl = document.getElementById('keySaveHint');
  if (!box || !valEl) return;
  valEl.textContent = rawKey;
  if (hintEl) {
    hintEl.textContent =
      "mkdir -p ~/.config/draft-site\n" +
      "printf '%s\\n' '" + rawKey + "' > ~/.config/draft-site/key\n" +
      "chmod 600 ~/.config/draft-site/key";
  }
  box.style.display = '';
}

function clearKeyReveal() {
  const box = document.getElementById('keyRevealBox');
  const valEl = document.getElementById('keyRevealValue');
  const hintEl = document.getElementById('keySaveHint');
  if (valEl) valEl.textContent = '';
  if (hintEl) hintEl.textContent = '';
  if (box) box.style.display = 'none';
}

function setKeysError(msg) {
  const el = document.getElementById('keysError');
  if (!el) return;
  if (!msg) {
    el.textContent = '';
    el.style.display = 'none';
  } else {
    el.textContent = msg;
    el.style.display = '';
  }
}

function setAdminError(msg) {
  const el = document.getElementById('adminError');
  if (!el) return;
  if (!msg) {
    el.textContent = '';
    el.style.display = 'none';
  } else {
    el.textContent = msg;
    el.style.display = '';
  }
}

function activateTab(tabId) {
  const panelEl = document.getElementById('tab-' + tabId);
  if (!panelEl) return;
  document.querySelectorAll('.tab').forEach(b => {
    b.classList.toggle('active', b.dataset.tab === tabId);
  });
  ['board', 'keys', 'admin', 'contributors', 'api'].forEach(id => {
    const el = document.getElementById('tab-' + id);
    if (el) el.style.display = (id === tabId) ? '' : 'none';
  });
}

async function doLogout() {
  if (!ME || !ME.csrf_token) return;
  clearKeyReveal();
  await fetch('/auth/logout', {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'X-CSRF-Token': ME.csrf_token },
  });
  ME = null;
  window.location.href = '/';
}

function renderAuthHeader() {
  const box = document.getElementById('authActions');
  if (!box) return;
  if (!ME || !ME.authenticated) {
    box.innerHTML = '<a class="btn btn-accent" href="/auth/login">Log in with GitHub</a>';
    return;
  }
  const disp = esc(ME.display_login || ME.login);
  const dotCls = ME.status === 'approved' ? 'dot' : 'dot-warn';
  const suffix = ME.status === 'approved' ? '' : ` (${esc(ME.status)})`;
  const adminLink = ME.can_admin_requests && document.body.dataset.page !== 'admin'
    ? '<a class="btn" href="/admin">Admin</a>'
    : '';
  box.innerHTML =
    `<span class="pill"><span class="${dotCls}"></span><span>@${disp}${suffix}</span></span>` +
    adminLink +
    '<button class="btn" id="logoutBtn" type="button">Log out</button>';
  const logoutBtn = document.getElementById('logoutBtn');
  if (logoutBtn) logoutBtn.addEventListener('click', doLogout);
}

async function loadAndRenderKeys() {
  const accBar = document.getElementById('keysAccountBar');
  const createBox = document.getElementById('keysCreateBox');
  const body = document.getElementById('keysBody');
  const meta = document.getElementById('keysMeta');
  if (!accBar || !body) return;
  setKeysError('');

  if (!ME || !ME.authenticated) {
    meta.textContent = 'Sign in with GitHub to manage keys';
    createBox.style.display = 'none';
    accBar.innerHTML =
      '<p style="margin:0 0 12px">Sign in with your GitHub account to create, rotate, and revoke your own worker API keys (up to 2 live keys per account; full key shown once).</p>' +
      '<a class="btn btn-accent" href="/auth/login?next=/%23keys">Log in with GitHub</a>';
    body.innerHTML = '';
    return;
  }

  const disp = esc(ME.display_login || ME.login);
  if (ME.status !== 'approved') {
    meta.textContent = `Account @${ME.display_login || ME.login} · ${ME.status}`;
    createBox.style.display = 'none';
    const msg = ME.status === 'pending'
      ? `Signed in as <code>@${disp}</code>. Your access request is <strong>waiting for owner approval</strong>. No API keys can be minted until approved.`
      : `Signed in as <code>@${disp}</code>. Status: <strong>${esc(ME.status)}</strong>. Key management is disabled.`;
    accBar.innerHTML = `<p style="margin:0 0 10px">${msg}</p><a class="btn" href="/waiting">View status page</a>`;
    body.innerHTML = '';
    return;
  }

  const data = await apiGet('/api/keys');
  const maxKeys = data.max_live_keys || 2;
  const liveCount = (data.keys || []).length;
  meta.textContent = `@${data.display_login || data.login} · ${liveCount}/${maxKeys} live keys`;
  const kindsStr = (data.allowed_kinds || []).map(k => `<code>${esc(k)}</code>`).join(', ') || 'none';
  const siteNote = data.login !== data.display_login ? ` · site login <code>${esc(data.login)}</code>` : '';
  accBar.innerHTML =
    `<div>Signed in via GitHub as <strong><code>@${esc(data.display_login || data.login)}</code></strong>${siteNote} · allowed project kinds: ${kindsStr} · strikes: <code>${Number(ME.strikes || 0)}</code></div>`;

  createBox.style.display = '';
  const createBtn = document.getElementById('createKeyBtn');
  const capHint = document.getElementById('keysCapHint');
  if (createBtn) createBtn.disabled = liveCount >= maxKeys;
  if (capHint) {
    capHint.textContent = liveCount >= maxKeys
      ? `Limit reached (${liveCount}/${maxKeys}). Rotate or revoke a key to mint another.`
      : `${liveCount}/${maxKeys} live keys used.`;
  }

  if (!liveCount) {
    body.innerHTML = '<div class="meta">No active API keys yet. Create one above to run <code>worker/label_worker.py</code>.</div>';
    return;
  }

  body.innerHTML =
    '<table><thead><tr><th>Prefix</th><th>Label</th><th>Allowed Kinds</th><th>Created</th><th>Last Used</th><th>Actions</th></tr></thead><tbody>' +
    data.keys.map(k => {
      const kKinds = (k.allowed_kinds || data.allowed_kinds || []).join(', ');
      return `<tr>
        <td><code>${esc(k.prefix)}…</code></td>
        <td>${esc(k.label || '—')}</td>
        <td><code>${esc(kKinds)}</code></td>
        <td>${esc(fmtUtc(k.created_utc))}</td>
        <td>${esc(fmtUtc(k.last_used_utc))}</td>
        <td class="row-actions">
          <button class="btn btn-sm" data-key-action="rotate" data-key-id="${esc(k.key_id)}" type="button">Rotate</button>
          <button class="btn btn-danger btn-sm" data-key-action="revoke" data-key-id="${esc(k.key_id)}" type="button">Revoke</button>
        </td>
      </tr>`;
    }).join('') +
    '</tbody></table>';
}

async function loadAndRenderAdmin() {
  const adminBody = document.getElementById('adminBody');
  const adminMeta = document.getElementById('adminMeta');
  if (!adminBody) return;
  if (!ME || !ME.can_admin_requests) {
    if (adminMeta) adminMeta.textContent = 'Owner authentication required';
    adminBody.innerHTML = '<div class="meta">Owner sign-in required.</div>';
    return;
  }
  setAdminError('');
  try {
    const rows = await apiGet('/api/admin/access-requests');
    const pending = rows.filter(r => r.status === 'pending');
    const others = rows.filter(r => r.status !== 'pending');
    if (adminMeta) {
      adminMeta.textContent = `${pending.length} pending · ${others.length} total accounts`;
    }

    const renderTable = (list, emptyMsg) => {
      if (!list.length) return `<div class="meta" style="margin-bottom:16px">${esc(emptyMsg)}</div>`;
      return '<table style="margin-bottom:20px"><thead><tr><th>GitHub</th><th>Site Login</th><th>Status</th><th>Tiers</th><th class="num">Keys</th><th class="num">Strikes</th><th>Actions</th></tr></thead><tbody>' +
        list.map(r => {
          const stCls = r.status === 'approved' ? 'badge-ok' : (r.status === 'pending' ? 'badge-warn' : 'badge-bad');
          const ghDisplay = r.github_login ? `@${r.github_login}` : '—';
          const nameSub = r.github_name ? `<div style="color:var(--muted);font-size:11px">${esc(r.github_name)}</div>` : '';
          const gidSub = r.github_id ? `<div style="color:var(--muted);font-size:11px">id:${esc(r.github_id)}</div>` : '';
          const canApprove = r.status !== 'approved';
          const canDeny = !r.is_owner && r.status !== 'denied';
          return `<tr>
            <td><code>${esc(ghDisplay)}</code>${nameSub}${gidSub}</td>
            <td><code>${esc(r.login)}</code>${r.is_owner ? ' <span class="badge badge-ok">owner</span>' : ''}</td>
            <td><span class="badge ${stCls}">${esc(r.status)}</span></td>
            <td><code>${esc((r.tiers || []).join(', '))}</code></td>
            <td class="num">${Number(r.live_keys || 0)}</td>
            <td class="num">${Number(r.strikes || 0)}</td>
            <td class="row-actions">
              ${canApprove ? `<button class="btn btn-accent btn-sm" data-admin-action="approve" data-login="${esc(r.login)}" type="button">Approve</button>` : ''}
              ${canDeny ? `<button class="btn btn-danger btn-sm" data-admin-action="deny" data-login="${esc(r.login)}" type="button">Deny</button>` : ''}
            </td>
          </tr>`;
        }).join('') +
        '</tbody></table>';
    };

    adminBody.innerHTML =
      '<h4 style="margin:0 0 10px">Pending Requests</h4>' +
      renderTable(pending, 'No pending GitHub access requests.') +
      '<h4 style="margin:0 0 10px">Approved, Denied &amp; Quarantined GitHub Accounts</h4>' +
      renderTable(others, 'No other GitHub accounts yet.');
  } catch (e) {
    setAdminError('Failed to load access requests: ' + e.message);
  }
}

async function refreshSession() {
  try {
    const r = await fetch('/auth/me', { credentials: 'same-origin', cache: 'no-store' });
    ME = r.ok ? await r.json() : { authenticated: false };
  } catch (_) {
    ME = { authenticated: false };
  }
  renderAuthHeader();
  const adminTab = document.getElementById('adminTabBtn');
  if (adminTab) {
    adminTab.style.display = (ME && ME.can_admin_requests) ? '' : 'none';
  }
  if (document.body.dataset.page === 'admin') {
    await loadAndRenderAdmin();
  } else {
    await loadAndRenderKeys();
    if (ME && ME.can_admin_requests) {
      await loadAndRenderAdmin();
    }
  }
}

function renderJobs() {
  const g = document.getElementById('jobsGrid');
  if (!g) return;
  const totSlots = JOBS.reduce((a, j) => a + j.slots, 0);
  const totVer = JOBS.reduce((a, j) => a + (j.verified ?? j.uploaded ?? 0), 0);
  document.getElementById('jobsMeta').textContent = `${JOBS.length} projects · ${totVer}/${totSlots} slots verified`;
  g.innerHTML = JOBS.map(j => {
    const tot = Math.max(1, j.slots);
    const kind = j.kind || j.job_kind || 'drafter-label';
    const verified = Number(j.verified ?? 0);
    const pending = Number(j.pending_verify ?? 0);
    const claimed = Number(j.claimed ?? 0);
    const openN = Number(j.open ?? 0);
    const closedN = Number(j.closed ?? 0);
    const disabledN = Number(j.disabled ?? 0);
    const wVer = (100 * verified / tot).toFixed(1);
    const wPend = (100 * pending / tot).toFixed(1);
    const wClm = (100 * claimed / tot).toFixed(1);
    const wOpen = (100 * openN / tot).toFixed(1);
    const wClosed = (100 * closedN / tot).toFixed(1);
    const wDis = (100 * disabledN / tot).toFixed(1);
    const sel = j.job === ACTIVE_JOB ? ' selected' : '';
    const href = '/jobs/' + encodeURIComponent(j.job);
    return `<div class="job-card${sel}" data-job="${esc(j.job)}">
      <div class="kind"><span>${esc(kind)}</span><a href="${href}" style="font-size:11px">Slots &rarr;</a></div>
      <div class="title">${esc(j.title)}</div>
      <div class="bar">
        <span style="width:${wVer}%;background:var(--verified)"></span>
        <span style="width:${wPend}%;background:var(--uploaded)"></span>
        <span style="width:${wClm}%;background:var(--claimed)"></span>
        <span style="width:${wOpen}%;background:var(--open)"></span>
        <span style="width:${wClosed}%;background:var(--closed)"></span>
        <span style="width:${wDis}%;background:var(--disabled)"></span>
      </div>
      <div class="stats-row">
        <span><strong>${verified}</strong>/${j.slots} verified</span>
        <span>${pending} pending</span>
        <span>${openN} open</span>
        <span>${claimed} claimed</span>
        ${closedN ? `<span>${closedN} closed</span>` : ''}
        <span>${j.not_ready} not ready</span>
        <span>${j.contributors} contrib</span>
      </div>
    </div>`;
  }).join('');
}

async function selectJob(jobId) {
  ACTIVE_JOB = jobId;
  renderJobs();
  const data = await apiGet('/api/grid?job=' + encodeURIComponent(jobId) + '&offset=0&limit=500');
  const kind = data.kind || data.job_kind || '';
  const weightsHf = (data.weights && data.weights.hf) || data.release || '';
  const weightsRev = ((data.weights && data.weights.revision) || data.revision || '').slice(0, 8);
  document.getElementById('gridTitle').textContent = `${data.title} (${data.job})`;
  document.getElementById('gridMeta').textContent =
    `${kind} · ${weightsHf}@${weightsRev} · ${data.total_slots} slots`;
  const sg = document.getElementById('slotGrid');
  GRID_SLOTS = new Map(data.slots.map(s => [s.slot_id, s]));
  // Campfire-style compact board: one row per job row (batch, checkpoint, ...), one small tile per slot.
  const axisRows = (data.axes && Array.isArray(data.axes.rows)) ? data.axes.rows.slice() : [];
  const byRow = new Map(axisRows.map(r => [r, []]));
  for (const s of data.slots) {
    const r = s.row || '';
    if (!byRow.has(r)) byRow.set(r, []);
    byRow.get(r).push(s);
  }
  const hidden = Math.max(0, (data.total_slots || 0) - data.slots.length);
  sg.innerHTML = [...byRow.entries()].filter(([, list]) => list.length).map(([r, list]) => {
    const tiles = list.map(s => {
      const cls = `slot-cell st-${esc(s.state)}${s.mine ? ' mine' : ''}`;
      return `<i class="${cls}" data-slot="${esc(s.slot_id)}"></i>`;
    }).join('');
    return `<div class="slot-row"><span class="rlabel" title="${esc(r)}">${esc(r || '—')}</span><div class="tiles">${tiles}</div></div>`;
  }).join('') + (hidden > 0 ? `<div class="slot-more">+${hidden} more slots not shown</div>` : '');
  const selBox = document.getElementById('slotSel');
  if (selBox) selBox.innerHTML = '<span class="muted">Hover a tile for its slot; click it for details.</span>';
}

let GRID_SLOTS = new Map();

function slotLines(s) {
  const m = s.metrics || {};
  let mline = '';
  if (m.owner_gate && m.owner_gate.top1_agree != null) {
    mline = `gate top1=${(100 * m.owner_gate.top1_agree).toFixed(1)}% top5=${(100 * m.owner_gate.top5_agree).toFixed(1)}%`;
  } else if (m.n_rows != null) {
    mline = `${m.n_rows} rows · ${m.total_gen_tokens || m.n_tokens || 0} tok`;
  } else if (s.upload_bytes) {
    mline = fmtBytes(s.upload_bytes);
  } else if (s.depends_on && s.state === 'not_ready') {
    mline = `awaits ${s.depends_on}`;
  }
  const who = s.display_login || s.contributor_login;
  return {
    head: `${s.slot_id} · ${s.state} · ${s.profile || s.col}`,
    who: who ? `by ${who}` : '',
    mline,
    gate: s.verify_status ? `gate: ${s.verify_status}` : '',
  };
}

function slotTipText(s) {
  const L = slotLines(s);
  return [L.head, L.who, L.mline].filter(Boolean).join(' · ');
}

function showSlotDetail(s) {
  const box = document.getElementById('slotSel');
  if (!box) return;
  const L = slotLines(s);
  box.innerHTML = `<strong>${esc(L.head)}</strong>` +
    `<span class="muted">${esc(s.row || '')}${s.col ? ' / ' + esc(s.col) : ''}</span>` +
    [L.who, L.mline, L.gate].filter(Boolean).map(x => `<span>${esc(x)}</span>`).join('');
}

function wireSlotGrid() {
  const sg = document.getElementById('slotGrid');
  const tip = document.getElementById('slotTip');
  if (!sg || !tip) return;
  sg.addEventListener('mousemove', ev => {
    const t = ev.target.closest('.slot-cell');
    const s = t && GRID_SLOTS.get(t.dataset.slot);
    if (!s) { tip.style.display = 'none'; return; }
    tip.textContent = slotTipText(s);
    tip.style.display = 'block';
    tip.style.left = (ev.clientX + 12) + 'px';
    tip.style.top = (ev.clientY + 12) + 'px';
  });
  sg.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
  sg.addEventListener('click', ev => {
    const t = ev.target.closest('.slot-cell');
    const s = t && GRID_SLOTS.get(t.dataset.slot);
    if (!s) return;
    sg.querySelectorAll('.slot-cell.sel').forEach(x => x.classList.remove('sel'));
    t.classList.add('sel');
    showSlotDetail(s);
  });
}

async function renderContribsAndAudit() {
  const [contribs, audit] = await Promise.all([apiGet('/api/contributors'), apiGet('/api/audit?limit=40')]);
  document.getElementById('contribBody').innerHTML =
    '<table><thead><tr><th>Login</th><th class="num">Verified</th><th class="num">Rejected</th><th>By Kind</th><th class="num">Uploaded</th></tr></thead><tbody>' +
    contribs.map(c => {
      const bk = Object.entries(c.by_kind || {}).map(([k, v]) => `${k.replace('drafter-', '')}:${v}`).join(' ');
      const disp = c.display_login || c.login;
      return `<tr><td><code>${esc(disp)}</code></td><td class="num tag-ok">${c.verified_slots}</td><td class="num ${c.rejected_attempts ? 'tag-bad' : ''}">${c.rejected_attempts}</td><td>${esc(bk || '—')}</td><td class="num">${fmtBytes(c.total_bytes_uploaded)}</td></tr>`;
    }).join('') + '</tbody></table>';

  document.getElementById('auditBody').innerHTML =
    '<table><thead><tr><th>UTC</th><th>Actor</th><th>Action</th><th>Slot</th><th>Status</th></tr></thead><tbody>' +
    audit.map(a => {
      const cls = String(a.status).startsWith('failed') || String(a.status).startsWith('reject') ? 'tag-bad'
        : String(a.status).startsWith('pending') ? 'tag-warn' : 'tag-ok';
      return `<tr><td style="color:var(--muted)">${esc((a.utc || '').slice(11, 19))}</td><td><code>${esc(a.actor)}</code></td><td>${esc(a.action)}</td><td>${esc(a.slot_id || a.job || '—')}</td><td class="${cls}">${esc(a.status)}</td></tr>`;
    }).join('') + '</tbody></table>';
}

async function renderDrafterProgress() {
  const panel = document.getElementById('drafterProgressPanel');
  if (!panel) return;
  let d;
  try { d = await apiGet('/api/drafter-progress'); } catch (e) { return; }
  if (!d || !d.available) { panel.style.display = 'none'; return; }
  const pct = Math.max(0, Math.min(100, Number(d.overall_pct) || 0));
  const icon = { done: '✅', running: '🔄', todo: '⬜' };
  let html = `<div class="bar" style="height:14px"><span style="width:${pct.toFixed(1)}%;background:var(--verified)"></span></div>
    <div class="stats-row"><span><strong>${pct.toFixed(0)}%</strong> overall</span><span>${d.done}/${d.total} steps done</span></div>`;
  let sec = null;
  for (const s of d.steps) {
    if (s.section !== sec) { sec = s.section; html += `<div style="margin-top:10px;font:600 11px var(--font-mono);text-transform:uppercase;color:var(--accent-text)">${esc(sec)}</div>`; }
    html += `<div>${icon[s.state] || '⬜'} ${esc(s.label)}</div>`;
  }
  document.getElementById('drafterProgressBody').innerHTML = html;
  document.getElementById('drafterProgressMeta').textContent = d.updated ? 'Updated ' + d.updated : '';
  panel.style.display = '';
}

async function refreshAll() {
  await refreshSession();
  if (document.body.dataset.page === 'admin') return;
  renderDrafterProgress();
  try {
    JOBS = await apiGet('/api/jobs');
    const initJob = document.body.dataset.initialJob;
    if (!ACTIVE_JOB && initJob && JOBS.some(j => j.job === initJob)) {
      ACTIVE_JOB = initJob;
    }
    if (!ACTIVE_JOB && JOBS.length) {
      const pref = JOBS.find(j => (j.kind || j.job_kind) === 'drafter-label' && j.open > 0);
      ACTIVE_JOB = pref ? pref.job : JOBS[0].job;
    }
    renderJobs();
    if (ACTIVE_JOB) await selectJob(ACTIVE_JOB);
    await renderContribsAndAudit();
  } catch (e) {
    const jm = document.getElementById('jobsMeta');
    if (jm) jm.textContent = 'Error loading API: ' + e.message;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const themeBtn = document.getElementById('themeBtn');
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const cur = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
      document.documentElement.dataset.theme = cur;
      localStorage.setItem('drs-theme', cur);
    });
  }

  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
      const t = btn.dataset.tab;
      activateTab(t);
      if (t === 'keys') loadAndRenderKeys().catch(e => setKeysError(e.message));
      if (t === 'admin') loadAndRenderAdmin().catch(e => setAdminError(e.message));
    });
  });

  const jobsGrid = document.getElementById('jobsGrid');
  if (jobsGrid) {
    jobsGrid.addEventListener('click', ev => {
      if (ev.target.closest('a')) return;
      const card = ev.target.closest('.job-card');
      if (card && card.dataset.job) {
        selectJob(card.dataset.job);
      }
    });
  }

  wireSlotGrid();

  const createKeyBtn = document.getElementById('createKeyBtn');
  if (createKeyBtn) {
    createKeyBtn.addEventListener('click', async () => {
      setKeysError('');
      const labelEl = document.getElementById('newKeyLabel');
      const label = labelEl ? labelEl.value.trim() : '';
      try {
        const res = await sessionMutate('POST', '/api/keys', { label });
        if (labelEl) labelEl.value = '';
        if (res.api_key_shown_once) showKeyReveal(res.api_key_shown_once);
        await loadAndRenderKeys();
      } catch (e) {
        setKeysError(e.message);
      }
    });
  }

  const copyKeyBtn = document.getElementById('copyKeyBtn');
  if (copyKeyBtn) {
    copyKeyBtn.addEventListener('click', async () => {
      const valEl = document.getElementById('keyRevealValue');
      if (!valEl || !valEl.textContent) return;
      try {
        await navigator.clipboard.writeText(valEl.textContent);
        copyKeyBtn.textContent = 'Copied!';
        setTimeout(() => { copyKeyBtn.textContent = 'Copy key'; }, 1800);
      } catch (_) {}
    });
  }

  const dismissKeyBtn = document.getElementById('dismissKeyBtn');
  if (dismissKeyBtn) {
    dismissKeyBtn.addEventListener('click', clearKeyReveal);
  }

  const keysBody = document.getElementById('keysBody');
  if (keysBody) {
    keysBody.addEventListener('click', async ev => {
      const btn = ev.target.closest('button[data-key-action]');
      if (!btn) return;
      const action = btn.dataset.keyAction;
      const keyId = btn.dataset.keyId;
      if (!keyId) return;
      setKeysError('');
      try {
        if (action === 'rotate') {
          const res = await sessionMutate('POST', '/api/keys/' + encodeURIComponent(keyId) + '/rotate', {});
          if (res.api_key_shown_once) showKeyReveal(res.api_key_shown_once);
          await loadAndRenderKeys();
        } else if (action === 'revoke') {
          await sessionMutate('DELETE', '/api/keys/' + encodeURIComponent(keyId));
          await loadAndRenderKeys();
        }
      } catch (e) {
        setKeysError(e.message);
      }
    });
  }

  const adminBody = document.getElementById('adminBody');
  if (adminBody) {
    adminBody.addEventListener('click', async ev => {
      const btn = ev.target.closest('button[data-admin-action]');
      if (!btn) return;
      const action = btn.dataset.adminAction;
      const login = btn.dataset.login;
      if (!login) return;
      setAdminError('');
      try {
        if (action === 'approve') {
          await sessionMutate('POST', '/api/admin/access-requests/' + encodeURIComponent(login) + '/approve', {});
        } else if (action === 'deny') {
          await sessionMutate('POST', '/api/admin/access-requests/' + encodeURIComponent(login) + '/deny', {});
        }
        await loadAndRenderAdmin();
      } catch (e) {
        setAdminError(e.message);
      }
    });
  }

  const refreshBtn = document.getElementById('refreshBtn');
  if (refreshBtn) refreshBtn.addEventListener('click', refreshAll);

  const hash = (window.location.hash || '').replace(/^#/, '');
  if (hash && ['board', 'keys', 'admin', 'contributors', 'api'].includes(hash)) {
    activateTab(hash);
  }

  refreshAll();
});
