'use strict';

(function initTheme() {
  const saved = localStorage.getItem('drs-theme');
  const pref = saved || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  document.documentElement.dataset.theme = pref;
})();

const H = { 'X-Quant': '1' };
let JOBS = [];
let ACTIVE_JOB = null;

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
    return `<div class="job-card${sel}" data-job="${esc(j.job)}">
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
        <span>${j.pending_verify} pending</span>
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
  const data = await apiGet('/api/grid?job=' + encodeURIComponent(jobId) + '&offset=0&limit=500');
  document.getElementById('gridTitle').textContent = `${data.title} (${data.job})`;
  document.getElementById('gridMeta').textContent =
    `${data.job_kind} · ${data.release}@${data.revision.slice(0, 8)} · ${data.total_slots} slots`;
  const sg = document.getElementById('slotGrid');
  const activeSlots = data.slots.filter(s => s.state !== 'open');
  const openSlots = data.slots.filter(s => s.state === 'open');
  const maxOpenShow = 48;
  const shown = data.slots.length > 80 ? activeSlots.concat(openSlots.slice(0, maxOpenShow)) : data.slots;
  const hiddenOpen = Math.max(0, (data.slots.length > 80 ? openSlots.length - maxOpenShow : 0)) +
    Math.max(0, data.total_slots - data.slots.length);
  sg.innerHTML = shown.map(s => {
    const m = s.metrics || {};
    let mline = '';
    if (m.b16_ref_tau != null) mline = `B16=${m.b16_ref_tau} B8=${m.b8_ref_tau} ce8=${m.ce8}`;
    else if (m.rank_pack_sha256s && m.rank_pack_sha256s.length) {
      mline = `TP${m.nr} · ${m.elapsed_s || '?'}s · ${m.rank_pack_sha256s[0].slice(0, 10)}…`;
    } else if (m.owner_gate && m.owner_gate.top1_agree != null) {
      mline = `gate top1=${(100 * m.owner_gate.top1_agree).toFixed(1)}% top5=${(100 * m.owner_gate.top5_agree).toFixed(1)}%`;
    } else if (m.owner_gate && m.owner_gate.delta_ce8 != null) {
      mline = `Δce8=${m.owner_gate.delta_ce8} cos=${m.owner_gate.cos_vs_root}`;
    } else if (m.server_delta_l2_norm != null) {
      mline = `L2=${m.server_delta_l2_norm} (${fmtBytes(s.upload_bytes)})`;
    } else if (m.n_rows != null) {
      mline = `${m.n_rows} rows · ${m.total_gen_tokens || m.n_tokens || 0} tok`;
    } else if (s.depends_on && s.state === 'not_ready') {
      mline = `awaits ${s.depends_on}`;
    }
    const by = s.contributor_login ? `by ${esc(s.contributor_login)}` : '';
    return `<div class="slot-cell st-${esc(s.state)}" title="${esc(s.verify_status || s.state)}">
      <div class="sid">${esc(s.slot_id)}</div>
      <div class="st">${esc(s.state)} · ${esc(s.profile || s.col)}</div>
      <div class="meta2">${by}${by && mline ? '<br>' : ''}${esc(mline)}${s.verify_status ? '<br>gate: ' + esc(s.verify_status) : ''}</div>
    </div>`;
  }).join('') + (hiddenOpen > 0
    ? `<div class="slot-cell st-open"><div class="sid">+${hiddenOpen} more slots</div><div class="st">queued</div><div class="meta2">ready for label contributors</div></div>`
    : '');
}

async function renderMatrixAndRounds() {
  const evalJob = JOBS.find(j => j.job_kind === 'drafter-eval');
  const trainJobs = JOBS.filter(j => j.job_kind === 'drafter-train');
  if (evalJob) {
    const g = await apiGet('/api/grid?job=' + encodeURIComponent(evalJob.job));
    const profs = ['bit200', 'bit240', 'bit280', 'bit322', 'bit400'];
    const ckpts = Object.keys(g.eval_matrix || {});
    if (!ckpts.length) {
      document.getElementById('evalMatrixBody').innerHTML = '<div class="meta">No verified eval slots yet.</div>';
    } else {
      let html = '<table><thead><tr><th>Checkpoint</th>' +
        profs.map(p => `<th class="num">${p} (B16 / B8)</th>`).join('') +
        '<th class="num">5-Prof Mean B16</th><th>Contributors</th></tr></thead><tbody>';
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
        const mean = b16s.length ? (b16s.reduce((a, b) => a + b, 0) / b16s.length).toFixed(4) : '—';
        html += `<tr><td><code>${esc(ck)}</code></td>${cells}<td class="num"><strong>${mean}</strong></td><td>${esc([...cs].join(', '))}</td></tr>`;
      }
      html += '</tbody></table>';
      document.getElementById('evalMatrixBody').innerHTML = html;
    }
  }
  if (trainJobs.length) {
    const allRds = [];
    for (const tj of trainJobs) {
      const g = await apiGet('/api/grid?job=' + encodeURIComponent(tj.job));
      for (const r of (g.rounds || [])) allRds.push({ job: tj.job, ...r });
    }
    if (!allRds.length) {
      document.getElementById('roundsBody').innerHTML = '<div class="meta">No training rounds registered.</div>';
    } else {
      let html = '<table><thead><tr><th>Job / Round</th><th>Base Ckpt</th><th>Status</th><th>Agg Method</th><th class="num">MP 5-Prof B16</th><th class="num">14-Dump B16 / roll16</th><th>Champion Reference</th></tr></thead><tbody>';
      for (const r of allRds) {
        const m = r.agg_metrics || {};
        const mp = m.mp_mean_b16 != null ? Number(m.mp_mean_b16).toFixed(4) : '—';
        const d14 = (m.sim14_b16 != null) ? `${Number(m.sim14_b16).toFixed(4)} / ${Number(m.sim14_roll16 || 0).toFixed(4)}` : '—';
        const champ = m.champion_ref ? `MP ${m.champion_ref.mp_mean_b16} · 14d ${m.champion_ref.sim14_b16}` : 'v5-s2-w5@500';
        html += `<tr><td><code>${esc(r.job)}</code> R${r.round}</td><td><code>${esc(r.base_ckpt_name)}</code></td><td>${esc(r.status)}</td><td>${esc(r.agg_method || '—')}</td><td class="num"><strong>${mp}</strong></td><td class="num">${d14}</td><td style="color:var(--muted)">${esc(champ)}</td></tr>`;
      }
      html += '</tbody></table>';
      document.getElementById('roundsBody').innerHTML = html;
    }
  }
}

async function renderContribsAndAudit() {
  const [contribs, audit] = await Promise.all([apiGet('/api/contributors'), apiGet('/api/audit?limit=40')]);
  document.getElementById('contribBody').innerHTML =
    '<table><thead><tr><th>Login</th><th class="num">Verified</th><th class="num">Rejected</th><th>By Kind</th><th class="num">Uploaded</th></tr></thead><tbody>' +
    contribs.map(c => {
      const bk = Object.entries(c.by_kind || {}).map(([k, v]) => `${k.replace('drafter-', '')}:${v}`).join(' ');
      return `<tr><td><code>${esc(c.login)}</code></td><td class="num tag-ok">${c.verified_slots}</td><td class="num ${c.rejected_attempts ? 'tag-bad' : ''}">${c.rejected_attempts}</td><td>${esc(bk || '—')}</td><td class="num">${fmtBytes(c.total_bytes_uploaded)}</td></tr>`;
    }).join('') + '</tbody></table>';

  document.getElementById('auditBody').innerHTML =
    '<table><thead><tr><th>UTC</th><th>Actor</th><th>Action</th><th>Slot</th><th>Status</th></tr></thead><tbody>' +
    audit.map(a => {
      const bad = /reject|fail/i.test(a.action + ' ' + a.status);
      const ok = /pass|accept|aggregat|opened/i.test(a.action + ' ' + a.status);
      const cls = bad ? 'tag-bad' : (ok ? 'tag-ok' : '');
      return `<tr><td>${esc(a.utc.slice(11, 19))}</td><td><code>${esc(a.actor)}</code></td><td>${esc(a.action)}</td><td>${esc(a.slot_id || a.job || '—')}</td><td class="${cls}">${esc(a.status)}</td></tr>`;
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

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('themeBtn').addEventListener('click', () => {
    const cur = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = cur;
    localStorage.setItem('drs-theme', cur);
  });

  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const t = btn.dataset.tab;
      ['board', 'matrix', 'contributors', 'api'].forEach(id => {
        document.getElementById('tab-' + id).style.display = (id === t) ? '' : 'none';
      });
    });
  });

  document.getElementById('jobsGrid').addEventListener('click', ev => {
    const card = ev.target.closest('.job-card');
    if (card && card.dataset.job) {
      selectJob(card.dataset.job);
    }
  });

  document.getElementById('refreshBtn').addEventListener('click', refreshAll);
  refreshAll();
});
