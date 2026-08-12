/* ── PreViral App Logic ─────────────────────────────────────────────── */

const API_BASE = '';  // same origin
let selectedPlatform = 'instagram';
let mediaFile = null;
let lastResult = null;

// ── Init ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setupPlatformPills();
  setupUpload();
  setupForm();
  setupCharCounter();
  setupReanalyze();
  checkReportRoute();
});

// ── Route: shared report page ──────────────────────────────────────────
function checkReportRoute() {
  const path = window.location.pathname;
  const m = path.match(/^\/report\/([a-zA-Z0-9-]+)$/);
  if (m) {
    document.getElementById('hero').style.display = 'none';
    document.getElementById('inputSection').style.display = 'none';
    loadSharedReport(m[1]);
  }
}

// ── Platform Pills ──────────────────────────────────────────────────────
function setupPlatformPills() {
  document.querySelectorAll('.pill').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      selectedPlatform = btn.dataset.platform;
    });
  });
}

// ── Upload ──────────────────────────────────────────────────────────────
function setupUpload() {
  const zone = document.getElementById('uploadZone');
  const input = document.getElementById('mediaInput');
  const preview = document.getElementById('thumbPreview');
  const img = document.getElementById('thumbImg');
  const removeBtn = document.getElementById('thumbRemove');

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) setFile(f);
  });
  input.addEventListener('change', () => { if (input.files[0]) setFile(input.files[0]); });
  removeBtn.addEventListener('click', () => clearFile());

  function setFile(f) {
    mediaFile = f;
    const url = URL.createObjectURL(f);
    img.src = url;
    zone.style.display = 'none';
    preview.style.display = 'flex';
  }
  function clearFile() {
    mediaFile = null;
    input.value = '';
    zone.style.display = 'block';
    preview.style.display = 'none';
    img.src = '';
  }
}

// ── Char counter ────────────────────────────────────────────────────────
function setupCharCounter() {
  const ta = document.getElementById('caption');
  const cnt = document.getElementById('charCount');
  ta.addEventListener('input', () => {
    cnt.textContent = ta.value.length;
    cnt.style.color = ta.value.length > 2000 ? 'var(--orange)' : '';
  });
}

// ── Reanalyze button ────────────────────────────────────────────────────
function setupReanalyze() {
  document.getElementById('reanalyzeBtn').addEventListener('click', () => {
    showSection('input');
    document.getElementById('caption').value = '';
    document.getElementById('charCount').textContent = '0';
  });
}

// ── Form Submit ─────────────────────────────────────────────────────────
function setupForm() {
  document.getElementById('analyzeForm').addEventListener('submit', async e => {
    e.preventDefault();
    const caption = document.getElementById('caption').value.trim();
    if (!caption) { alert('Please enter a caption.'); return; }

    showSection('loading');
    startLoadingAnimation();

    const fd = new FormData();
    fd.append('caption', caption);
    fd.append('platform', selectedPlatform);
    fd.append('follower_count', document.getElementById('followerCount').value || 10000);
    fd.append('niche', document.getElementById('niche').value);
    fd.append('post_datetime', new Date().toISOString());
    if (mediaFile) fd.append('media', mediaFile);

    try {
      const res = await fetch(`${API_BASE}/api/v1/analyze`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      lastResult = { caption, platform: selectedPlatform, ...data };
      showSection('results');
      renderResults(data, caption);

      // Fire AI Director call async after main results are shown
      triggerAIDirector(caption, selectedPlatform, data.confidence, data.prediction);

    } catch (err) {
      alert(`Analysis failed: ${err.message}. Make sure the API is running.`);
      showSection('input');
    }
  });
}

// ── Loading animation ───────────────────────────────────────────────────
let loadingTimer = null;
function startLoadingAnimation() {
  const steps = [
    { text: 'Gemini reading visual signals...', sub: 0, pct: 20 },
    { text: 'Scoring hook strength & sentiment...', sub: 1, pct: 45 },
    { text: 'Running LightGBM prediction...', sub: 2, pct: 75 },
    { text: 'Generating counterfactuals...', sub: 3, pct: 95 },
  ];
  let i = 0;
  // Reset
  document.querySelectorAll('.substep').forEach(s => s.classList.remove('active','done'));
  document.getElementById('loadingBar').style.width = '5%';

  function advance() {
    if (i < steps.length) {
      const s = steps[i];
      document.getElementById('loadingStep').textContent = s.text;
      document.getElementById('loadingBar').style.width = s.pct + '%';
      if (i > 0) {
        document.getElementById('sub' + (i-1)).classList.remove('active');
        document.getElementById('sub' + (i-1)).classList.add('done');
      }
      document.getElementById('sub' + s.sub).classList.add('active');
      i++;
      loadingTimer = setTimeout(advance, 750);
    }
  }
  advance();
}

// ── Section manager ─────────────────────────────────────────────────────
function showSection(name) {
  const sections = {
    hero:    document.getElementById('hero'),
    input:   document.getElementById('inputSection'),
    loading: document.getElementById('loadingSection'),
    results: document.getElementById('resultsSection'),
    report:  document.getElementById('reportSection'),
  };
  Object.values(sections).forEach(s => { if(s) s.style.display = 'none'; });
  if (loadingTimer) { clearTimeout(loadingTimer); loadingTimer = null; }

  if (name === 'input') {
    sections.hero.style.display = '';
    sections.input.style.display = '';
  } else if (name === 'loading') {
    sections.loading.style.display = '';
  } else if (name === 'results') {
    sections.input.style.display = '';
    sections.results.style.display = '';
  } else if (name === 'report') {
    sections.report.style.display = '';
  }
}

// ── Render results ──────────────────────────────────────────────────────
function renderResults(data, caption) {
  const tier = data.prediction || 'MEDIUM';
  const conf = data.confidence || 0.5;
  const reach = data.reach_percentile || 50;

  // Verdict
  document.getElementById('verdictTier').textContent = tier;
  document.getElementById('verdictTier').className = `verdict-tier ${tier}`;
  document.getElementById('verdictHeadline').textContent = data.headline || '';

  // Gauge
  animateGauge(conf, tier);

  // Reach
  document.getElementById('reachPct').textContent = reach;

  // Score breakdown
  const nlp = data.nlp_features || {};
  const ht  = data.hashtag_features || {};
  const tm  = data.timing_features || {};

  const hookVal = nlp.gemini_hook_strength ?? nlp._gemini_hook_strength ?? (nlp.clickbait_score * 0.5 + (nlp.cta_present || 0) * 0.5);
  const sentVal = ((nlp.sentiment_score || 0) + 1) / 2;
  const timeVal = tm.peak_overlap_score || tm.day_of_week_score || 0.5;
  const hashVal = Math.min((ht.trending_hashtag_count || 0) * 0.15 + (ht.avg_competition_ratio !== undefined ? (1 - ht.avg_competition_ratio) * 0.5 : 0.4), 1);

  animateBar('bkHook', 'bkHookVal', hookVal);
  animateBar('bkSent', 'bkSentVal', sentVal);
  animateBar('bkTime', 'bkTimeVal', timeVal);
  animateBar('bkHash', 'bkHashVal', hashVal);

  // Trajectory
  if (data.trajectory && data.trajectory.length) {
    requestAnimationFrame(() => drawTrajectory(data.trajectory, tier));
  } else {
    document.getElementById('trajectoryCard').style.display = 'none';
  }

  // Counterfactuals
  renderCFs(data.suggestions || []);

  // Share button
  setupShareBtn(data, caption);

  // Scroll into view
  document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Gauge animation ──────────────────────────────────────────────────────
function animateGauge(conf, tier) {
  const fill = document.getElementById('gaugeFill');
  const pctEl = document.getElementById('gaugePct');
  const circumference = 314;
  fill.className = `gauge-fill ${tier}`;

  let current = 0;
  const target = Math.round(conf * 100);
  const interval = setInterval(() => {
    current = Math.min(current + 2, target);
    pctEl.textContent = current + '%';
    fill.style.strokeDashoffset = circumference - (current / 100) * circumference;
    if (current >= target) clearInterval(interval);
  }, 16);
}

// ── Bar animation ────────────────────────────────────────────────────────
function animateBar(barId, valId, value) {
  const pct = Math.round(Math.min(Math.max(value, 0), 1) * 100);
  document.getElementById(barId).style.width = pct + '%';
  document.getElementById(valId).textContent = pct + '%';
}

// ── Trajectory chart (canvas) ─────────────────────────────────────────────
function drawTrajectory(points, tier) {
  const canvas = document.getElementById('trajectoryChart');
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth * devicePixelRatio;
  const H = 220 * devicePixelRatio;
  canvas.width = W; canvas.height = H;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  const w = canvas.offsetWidth, h = 220;

  const colors = { HIGH: '#10b981', MEDIUM: '#f59e0b', LOW: '#ef4444' };
  const color = colors[tier] || '#6366f1';

  const PAD = { top: 20, right: 20, bottom: 40, left: 60 };
  const cw = w - PAD.left - PAD.right;
  const ch = h - PAD.top - PAD.bottom;

  const days  = points.map(p => p.day);
  const highs = points.map(p => p.high);
  const mids  = points.map(p => p.mid);
  const lows  = points.map(p => p.low);
  const maxV  = Math.max(...highs) * 1.1 || 1;

  const px = i => PAD.left + (i / (days.length - 1)) * cw;
  const py = v => PAD.top + ch - (v / maxV) * ch;

  const fmt = v => v >= 1e6 ? (v/1e6).toFixed(1)+'M' : v >= 1e3 ? (v/1e3).toFixed(0)+'K' : v;

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = PAD.top + (i / 4) * ch;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + cw, y); ctx.stroke();
    ctx.fillStyle = 'rgba(148,163,184,0.5)';
    ctx.font = '11px Inter';
    ctx.fillText(fmt(maxV * (1 - i/4)), 0, y + 4);
  }

  // Band (high-low)
  ctx.beginPath();
  highs.forEach((v, i) => i === 0 ? ctx.moveTo(px(i), py(v)) : ctx.lineTo(px(i), py(v)));
  lows.slice().reverse().forEach((v, i) => ctx.lineTo(px(lows.length - 1 - i), py(v)));
  ctx.closePath();
  ctx.fillStyle = color + '18';
  ctx.fill();

  // Mid line (animated)
  let progress = 0;
  const totalLen = mids.length - 1;
  function drawFrame() {
    ctx.clearRect(PAD.left, PAD.top - 5, cw + 5, ch + 10);

    // Re-draw band
    ctx.beginPath();
    highs.forEach((v, i) => i === 0 ? ctx.moveTo(px(i), py(v)) : ctx.lineTo(px(i), py(v)));
    lows.slice().reverse().forEach((v, i) => ctx.lineTo(px(lows.length - 1 - i), py(v)));
    ctx.closePath();
    ctx.fillStyle = color + '18';
    ctx.fill();

    // Animated mid line
    const draw_to = progress;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    for (let i = 0; i <= draw_to && i < mids.length; i++) {
      const frac = Math.min(progress - Math.floor(progress), 1);
      let x = px(i), y = py(mids[i]);
      if (i === Math.floor(draw_to) && i < mids.length - 1 && frac < 1) {
        x = px(i) + (px(i+1) - px(i)) * frac;
        y = py(mids[i]) + (py(mids[i+1]) - py(mids[i])) * frac;
      }
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Dot on last drawn point
    const ci = Math.min(Math.floor(draw_to), mids.length - 1);
    const frac = Math.min(draw_to - ci, 1);
    let dotX = px(ci), dotY = py(mids[ci]);
    if (ci < mids.length - 1) {
      dotX += (px(ci+1) - px(ci)) * frac;
      dotY += (py(mids[ci+1]) - py(mids[ci])) * frac;
    }
    ctx.beginPath();
    ctx.arc(dotX, dotY, 5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    if (progress < totalLen) {
      progress = Math.min(progress + 0.05, totalLen);
      requestAnimationFrame(drawFrame);
    } else {
      // Labels on final points
      ctx.fillStyle = 'rgba(148,163,184,0.8)';
      ctx.font = '11px Inter';
      days.forEach((d, i) => {
        ctx.fillText('Day ' + d, px(i) - 15, h - 8);
      });
      // Point values
      ctx.fillStyle = color;
      ctx.font = '11px Inter';
      mids.forEach((v, i) => ctx.fillText(fmt(v), px(i) - 12, py(v) - 10));
    }
  }
  drawFrame();
}

// ── Counterfactuals ──────────────────────────────────────────────────────
const CF_ICONS = ['🎯', '⏰', '#️⃣', '😊', '📸', '💡', '🔥', '📝'];
function renderCFs(suggestions) {
  const list = document.getElementById('cfList');
  list.innerHTML = '';
  const items = suggestions.slice(0, 3);
  if (!items.length) {
    list.innerHTML = '<div class="cf-item"><div class="cf-text"><strong>🎉 Your post is already optimized!</strong><span>The analysis found no major improvements needed.</span></div></div>';
    return;
  }
  items.forEach((s, i) => {
    const el = document.createElement('div');
    el.className = 'cf-item';
    const title = s.suggestion || s.title || 'Improvement';
    const detail = s.detail || s.explanation || '';
    const impact = s.impact || s.estimated_lift || '';
    el.innerHTML = `
      <div class="cf-icon">${CF_ICONS[i % CF_ICONS.length]}</div>
      <div class="cf-text">
        <strong>${title}</strong>
        <span>${detail}</span>
      </div>
      ${impact ? `<div class="cf-impact">+${impact}</div>` : ''}
    `;
    list.appendChild(el);
  });
}

// ── AI Content Director ──────────────────────────────────────────────────
async function triggerAIDirector(caption, platform, confidence, prediction) {
  const body = document.getElementById('directorBody');
  body.innerHTML = '<div class="director-loading"><div class="gemini-spinner"></div><span>Gemini is analyzing your content...</span></div>';

  const fd = new FormData();
  fd.append('caption', caption);
  fd.append('platform', platform);
  fd.append('follower_count', document.getElementById('followerCount').value || 10000);
  if (mediaFile) fd.append('media', mediaFile);

  try {
    const res = await fetch(`${API_BASE}/api/v1/ai-director`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const d = await res.json();
    renderDirector(d, caption);
    // Update lastResult with director data
    if (lastResult) lastResult.director = d;
  } catch (e) {
    body.innerHTML = `<div class="insight-row"><span class="insight-icon">⚠️</span><span class="insight-text">AI Content Director unavailable: ${e.message}</span></div>`;
  }
}

function renderDirector(d, originalCaption) {
  const body = document.getElementById('directorBody');
  const rewritten = d.rewritten_caption || originalCaption;
  const currentConf = d.current_confidence || 0;
  const afterConf   = d.predicted_score_after || Math.min(currentConf + 0.12, 0.95);
  const liftPct     = Math.round((afterConf - currentConf) * 100);

  body.innerHTML = `
    <div class="caption-compare">
      <div class="caption-box">
        <div class="caption-box-label before">❌ Original</div>
        <div class="caption-text before">${escHtml(originalCaption)}</div>
      </div>
      <div class="caption-box">
        <div class="caption-box-label after">✅ Gemini Rewrite</div>
        <div class="caption-text after" style="position:relative">
          ${escHtml(rewritten)}
          <button class="copy-btn" onclick="copyText(${JSON.stringify(rewritten)}, this)">Copy</button>
        </div>
      </div>
    </div>

    ${d.hook_rewrite ? `
    <div class="insight-row">
      <span class="insight-icon">🎣</span>
      <div class="insight-text">
        <strong>Hook rewrite:</strong>
        "${escHtml(d.hook_rewrite)}"
      </div>
    </div>` : ''}

    ${d.alignment_assessment ? `
    <div class="insight-row">
      <span class="insight-icon">👁️</span>
      <div class="insight-text">${escHtml(d.alignment_assessment)}</div>
    </div>` : ''}

    ${(d.specific_improvements || []).map(imp => `
    <div class="insight-row">
      <span class="insight-icon">💡</span>
      <div class="insight-text">${escHtml(imp)}</div>
    </div>`).join('')}

    ${d.thumbnail_suggestion ? `
    <div class="insight-row">
      <span class="insight-icon">🖼️</span>
      <div class="insight-text"><strong>Thumbnail:</strong> ${escHtml(d.thumbnail_suggestion)}</div>
    </div>` : ''}

    ${d.best_posting_time ? `
    <div class="insight-row">
      <span class="insight-icon">⏰</span>
      <div class="insight-text"><strong>Best time to post:</strong> ${escHtml(d.best_posting_time)}</div>
    </div>` : ''}

    ${d.vocabulary_suggestion ? `
    <div class="insight-row">
      <span class="insight-icon">🔤</span>
      <div class="insight-text"><strong>Trending vocabulary:</strong> ${escHtml(d.vocabulary_suggestion)}</div>
    </div>` : ''}

    ${liftPct > 0 ? `
    <div class="score-lift">
      <div>
        <div class="lift-label">Predicted score lift after applying suggestions</div>
        <div class="lift-val">+${liftPct}%</div>
      </div>
      <div class="lift-arrow">→</div>
      <div>
        <div class="lift-label">New predicted confidence</div>
        <div class="lift-val">${Math.round(afterConf * 100)}%</div>
      </div>
    </div>` : ''}
  `;
}

// ── Share report ───────────────────────────────────────────────────────────
function setupShareBtn(data, caption) {
  document.getElementById('shareBtn').onclick = async () => {
    const toast = document.getElementById('shareToast');
    const urlInput = document.getElementById('shareUrl');
    toast.style.display = 'none';
    try {
      const payload = { caption, platform: selectedPlatform, ...data };
      const res = await fetch(`${API_BASE}/api/v1/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const r = await res.json();
      const shareURL = `${window.location.origin}${r.share_url}`;
      urlInput.value = shareURL;
      toast.style.display = 'flex';
      navigator.clipboard?.writeText(shareURL);
      urlInput.select();
    } catch (e) {
      alert('Could not generate share link: ' + e.message);
    }
  };
}

// ── Load shared report ─────────────────────────────────────────────────────
async function loadSharedReport(id) {
  showSection('report');
  const body = document.getElementById('reportBody');
  const meta = document.getElementById('reportMeta');
  body.innerHTML = '<div style="text-align:center;padding:40px;color:var(--text-sub)">Loading report...</div>';
  try {
    const res = await fetch(`${API_BASE}/api/v1/report/${id}`);
    if (!res.ok) throw new Error('Report not found');
    const report = await res.json();
    const d = report.data || {};
    meta.textContent = `Created ${new Date(report.created_at).toLocaleString()} · Platform: ${(d.platform || '').toUpperCase()}`;

    const tier = d.prediction || 'MEDIUM';
    const conf = d.confidence || 0;
    body.innerHTML = `
      <div style="margin-bottom:24px">
        <div class="verdict-tier ${tier}" style="font-size:40px">${tier}</div>
        <div style="font-size:14px;color:var(--text-sub);margin-top:8px">${d.headline || ''}</div>
        <div style="margin-top:12px;font-size:28px;font-weight:800;color:var(--accent)">${Math.round(conf * 100)}% confidence</div>
      </div>
      ${d.caption ? `<div style="padding:16px;background:var(--surface2);border-radius:10px;font-size:14px;margin-bottom:20px;line-height:1.7">${escHtml(d.caption)}</div>` : ''}
      <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:16px">
        <a href="/" class="share-btn" style="text-decoration:none">⚡ Analyze your own post</a>
      </div>
    `;
  } catch (e) {
    body.innerHTML = `<div style="text-align:center;padding:40px;color:var(--red)">${e.message}</div>`;
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function copyText(text, btn) {
  navigator.clipboard?.writeText(text).then(() => {
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy', 2000);
  });
}
