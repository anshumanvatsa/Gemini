/* -- PreViral App Logic -------------------------------------------- */

const API_BASE = '';  // same origin
let selectedPlatform = 'instagram';
let selectedContentType = 'reel';
let mediaFile = null;
let lastResult = null;

// -- Content type definitions per platform --
const CONTENT_TYPES = {
  instagram: [
    { id: 'reel',     label: 'Reel',          visual: 'required' },
    { id: 'carousel', label: 'Carousel',       visual: 'recommended' },
    { id: 'post',     label: 'Single Post',    visual: 'recommended' },
    { id: 'story',    label: 'Story',          visual: 'required' },
  ],
  tiktok: [
    { id: 'video',    label: 'Video',          visual: 'required' },
    { id: 'carousel', label: 'Photo Carousel', visual: 'recommended' },
  ],
  youtube: [
    { id: 'video',    label: 'Video',          visual: 'required' },
    { id: 'shorts',   label: 'Shorts',         visual: 'required' },
    { id: 'community',label: 'Community Post', visual: 'none' },
  ],
  twitter: [
    { id: 'tweet',    label: 'Tweet',          visual: 'none' },
    { id: 'thread',   label: 'Thread',         visual: 'none' },
    { id: 'media',    label: 'Tweet + Media',  visual: 'recommended' },
  ],
  linkedin: [
    { id: 'text',     label: 'Text Post',      visual: 'none' },
    { id: 'image',    label: 'Image Post',     visual: 'recommended' },
    { id: 'article',  label: 'Article',        visual: 'recommended' },
    { id: 'document', label: 'Document',       visual: 'recommended' },
  ],
  facebook: [
    { id: 'post',     label: 'Post',           visual: 'recommended' },
    { id: 'reel',     label: 'Reel',           visual: 'required' },
    { id: 'story',    label: 'Story',          visual: 'required' },
  ],
};

const VISUAL_MODE_CONFIG = {
  required: {
    tag: 'strongly recommended - critical for algorithm',
    tagClass: 'tag-required',
    icon: '\ud83d\uddbc\ufe0f',
    uploadText: 'Upload your cover image or thumbnail',
    showDesc: true,
    disabled: false,
  },
  recommended: {
    tag: 'optional - improves accuracy',
    tagClass: 'tag-optional',
    icon: '\ud83d\uddbc\ufe0f',
    uploadText: 'Drop your image here or browse',
    showDesc: true,
    disabled: false,
  },
  none: {
    tag: 'not applicable for this content type',
    tagClass: 'tag-na',
    icon: '',
    uploadText: 'Drop your image here or browse',
    showDesc: false,
    disabled: true,
  },
};

// -- Init --
document.addEventListener('DOMContentLoaded', () => {
  setupPlatformPills();
  rebuildContentTypePills('instagram');
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
  document.querySelectorAll('#platformPills .pill').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#platformPills .pill').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      selectedPlatform = btn.dataset.platform;
      rebuildContentTypePills(selectedPlatform);
    });
  });
}

// Build content type pills dynamically for selected platform
function rebuildContentTypePills(platform) {
  const container = document.getElementById('contentTypePills');
  const types = CONTENT_TYPES[platform] || [];
  container.innerHTML = '';
  types.forEach((ct, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'ct-pill' + (idx === 0 ? ' active' : '');
    btn.dataset.ct = ct.id;
    btn.dataset.visual = ct.visual;
    btn.textContent = ct.label;
    btn.addEventListener('click', () => {
      container.querySelectorAll('.ct-pill').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      selectedContentType = ct.id;
      applyVisualMode(ct.visual);
    });
    container.appendChild(btn);
  });
  // Set first as default and apply its visual mode
  if (types.length > 0) {
    selectedContentType = types[0].id;
    applyVisualMode(types[0].visual);
  }
}

// Update visual input section based on visual mode
function applyVisualMode(mode) {
  const cfg = VISUAL_MODE_CONFIG[mode] || VISUAL_MODE_CONFIG.recommended;
  const group = document.getElementById('visualInputGroup');
  const label = document.getElementById('visualLabel');
  const tag   = document.getElementById('visualTag');
  const icon  = document.getElementById('uploadIcon');
  const text  = document.getElementById('uploadText');
  const descWrap = document.getElementById('visualDescWrap');

  if (cfg.hidden) {
    group.style.opacity = '0.4';
    group.style.pointerEvents = 'none';
  } else {
    group.style.opacity = '1';
    group.style.pointerEvents = '';
  }

  // Update tag text and class
  tag.textContent = cfg.tag;
  tag.className = 'optional-tag ' + cfg.tagClass;

  // Update upload zone text
  if (icon) icon.textContent = cfg.icon;
  const uploadTextEl = document.getElementById('uploadText');
  if (uploadTextEl) uploadTextEl.innerHTML = cfg.uploadText + ' or <span class="upload-link">browse</span>';

  // Show/hide describe fallback
  descWrap.style.display = cfg.showDesc ? 'block' : 'none';
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
  const analyzeBtn = document.getElementById('analyzeBtn');
  ta.addEventListener('input', () => {
    cnt.textContent = ta.value.length;
    cnt.style.color = ta.value.length > 2000 ? 'var(--orange)' : '';
    analyzeBtn.classList.toggle('pulse-ready', ta.value.trim().length > 10);
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
// ── Form Submit ───────────────────────────────────────────────────────
function setupForm() {
  document.getElementById('analyzeForm').addEventListener('submit', async e => {
    e.preventDefault();
    const caption = document.getElementById('caption').value.trim();
    if (!caption) { alert('Please enter a caption.'); return; }

    // Reset safety notice
    document.getElementById('safetyNotice').style.display = 'none';

    showSection('loading');
    startLoadingAnimation();

    const fd = new FormData();
    fd.append('caption', caption);
    fd.append('platform', selectedPlatform);
    fd.append('content_type', selectedContentType);
    fd.append('follower_count', document.getElementById('followerCount').value || 10000);
    fd.append('niche', document.getElementById('niche').value);
    fd.append('post_datetime', new Date().toISOString());
    if (mediaFile) fd.append('media', mediaFile);
    // Describe-your-visual fallback
    const visualDescEl = document.getElementById('visualDescription');
    if (visualDescEl && visualDescEl.value.trim() && !mediaFile) {
      fd.append('visual_description', visualDescEl.value.trim());
    }

    try {
      const res = await fetch(`${API_BASE}/api/v1/analyze`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      lastResult = { caption, platform: selectedPlatform, ...data };
      showSection('results');
      renderResults(data, caption);

      // Show safety notice if image was flagged
      if (data.safety_flag) {
        document.getElementById('safetyNotice').style.display = 'flex';
      }

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

  const hasHashtags = (ht.hashtag_count || 0) > 0;
  document.getElementById('bkHashBarWrap').style.display = hasHashtags ? '' : 'none';
  document.getElementById('bkHashVal').style.display = hasHashtags ? '' : 'none';
  document.getElementById('bkHashEmpty').style.display = hasHashtags ? 'none' : '';
  if (hasHashtags) animateBar('bkHash', 'bkHashVal', hashVal);

  // Trajectory
  if (data.trajectory && data.trajectory.length) {
    requestAnimationFrame(() => drawTrajectory(data.trajectory, tier));
  } else {
    document.getElementById('trajectoryCard').style.display = 'none';
  }

  // Counterfactuals
  renderCFs(data.suggestions || []);

  // Trending hashtags (Gemini)
  renderTrending(data.trending_hashtags || {});

  // 10-Day Report
  renderTenDaySummary(data.ten_day_summary || null);

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
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.offsetWidth;
  const h = 220;
  canvas.width  = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const colors = { HIGH: '#10b981', MEDIUM: '#f59e0b', LOW: '#ef4444' };
  const color = colors[tier] || '#6366f1';

  const PAD = { top: 24, right: 24, bottom: 40, left: 64 };
  const cw = w - PAD.left - PAD.right;
  const ch = h - PAD.top - PAD.bottom;

  const days  = points.map(p => p.day);
  const highs = points.map(p => p.high);
  const mids  = points.map(p => p.mid);
  const lows  = points.map(p => p.low);
  const maxV  = Math.max(...highs) * 1.15 || 1;

  const px = i => PAD.left + (i / (days.length - 1)) * cw;
  const py = v => PAD.top + ch - (v / maxV) * ch;

  const fmt = v => v >= 1e6 ? (v/1e6).toFixed(1)+'M' : v >= 1e3 ? Math.round(v/1e3)+'K' : String(Math.round(v));

  // ── Draw static background (grid + Y labels) ──────────────────────────
  function drawBackground() {
    // Grid lines
    for (let i = 0; i <= 4; i++) {
      const yPos = PAD.top + (i / 4) * ch;
      const val  = maxV * (1 - i / 4);
      const label = fmt(val);

      ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(PAD.left, yPos);
      ctx.lineTo(PAD.left + cw, yPos);
      ctx.stroke();

      // Right-aligned Y-axis label (avoid colliding with plot area)
      ctx.fillStyle = 'rgba(148,163,184,0.55)';
      ctx.font = '10px Inter';
      ctx.textAlign = 'right';
      ctx.fillText(label, PAD.left - 8, yPos + 4);
    }
    ctx.textAlign = 'left';
  }

  drawBackground();

  // ── Band (high-low confidence range) ──────────────────────────────────
  function drawBand() {
    ctx.beginPath();
    highs.forEach((v, i) => i === 0 ? ctx.moveTo(px(i), py(v)) : ctx.lineTo(px(i), py(v)));
    lows.slice().reverse().forEach((v, i) => ctx.lineTo(px(lows.length - 1 - i), py(v)));
    ctx.closePath();
    ctx.fillStyle = color + '1a';
    ctx.fill();
  }

  // ── Mid line (animated) ────────────────────────────────────────────────
  let progress = 0;
  const totalLen = mids.length - 1;

  function drawFrame() {
    // Full canvas clear each frame prevents ghost artifacts
    ctx.clearRect(0, 0, w, h);
    drawBackground();
    drawBand();

    const draw_to = progress;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.lineCap  = 'round';

    for (let i = 0; i <= draw_to && i < mids.length; i++) {
      const frac = progress - Math.floor(progress);
      let x = px(i), y = py(mids[i]);
      if (i === Math.floor(draw_to) && i < mids.length - 1 && frac < 1) {
        x = px(i) + (px(i + 1) - px(i)) * frac;
        y = py(mids[i]) + (py(mids[i + 1]) - py(mids[i])) * frac;
      }
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Leading dot
    const ci = Math.min(Math.floor(draw_to), mids.length - 1);
    const frac2 = Math.min(draw_to - ci, 1);
    let dotX = px(ci), dotY = py(mids[ci]);
    if (ci < mids.length - 1) {
      dotX += (px(ci + 1) - px(ci)) * frac2;
      dotY += (py(mids[ci + 1]) - py(mids[ci])) * frac2;
    }
    ctx.beginPath();
    ctx.arc(dotX, dotY, 5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    if (progress < totalLen) {
      progress = Math.min(progress + 0.05, totalLen);
      requestAnimationFrame(drawFrame);
    } else {
      // Final state — draw all point labels
      ctx.font = '10.5px Inter';
      ctx.textAlign = 'center';

      // Day labels along X axis
      ctx.fillStyle = 'rgba(148,163,184,0.7)';
      days.forEach((d, i) => ctx.fillText('Day ' + d, px(i), h - 6));

      // Value labels above each point (collision-aware: first point shifts right)
      ctx.fillStyle = color;
      mids.forEach((v, i) => {
        const lx = i === 0 ? px(i) + 14 : (i === mids.length - 1 ? px(i) - 14 : px(i));
        ctx.fillText(fmt(v), lx, py(v) - 12);
      });

      ctx.textAlign = 'left';

      // Permanent dots on each data point
      mids.forEach((v, i) => {
        ctx.beginPath();
        ctx.arc(px(i), py(v), 4, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = 'rgba(15,23,42,0.8)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
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

// ── Trending Hashtags (Gemini Search Grounding) ─────────────────────────────
function renderTrending(data) {
  const card = document.getElementById('trendCard');
  const body = document.getElementById('trendBody');
  const subtitle = document.getElementById('trendSubtitle');

  const trending = data.trending_now || [];
  const stable   = data.stable_performers || [];
  const avoid    = data.avoid || [];
  const topic    = data.topic_detected || '';
  const grounded = data.grounding_used || data._gemini_used;

  // Hide if no data at all
  if (!trending.length && !stable.length) {
    card.style.display = 'none';
    return;
  }

  card.style.display = '';
  subtitle.textContent = data._gemini_used
    ? `Powered by Gemini · Topic: ${topic || 'general'}`
    : `Based on hashtag database · Topic: ${topic || niche}`;

  let html = '';

  if (trending.length) {
    html += `<div class="trend-bucket">
      <div class="trend-bucket-label fire">🔥 Trending in your niche right now</div>
      <div class="trend-tags">
        ${trending.map(t => `
          <div class="trend-tag fire-tag" title="${escHtml(t.why || '')}">
            ${escHtml(t.tag)}
            ${t.velocity === 'high' ? '<span class="trend-tag-velocity">HOT</span>' : ''}
          </div>`).join('')}
      </div>
    </div>`;
  }

  if (stable.length) {
    html += `<div class="trend-bucket">
      <div class="trend-bucket-label check">✅ Stable discovery hashtags</div>
      <div class="trend-tags">
        ${stable.map(t => `
          <div class="trend-tag check-tag" title="${escHtml(t.why || '')}">
            ${escHtml(t.tag)}
          </div>`).join('')}
      </div>
    </div>`;
  }

  if (avoid.length) {
    html += `<div class="trend-bucket">
      <div class="trend-bucket-label skip">⚠️ Skip these (oversaturated / off-topic)</div>
      <div class="trend-tags">
        ${avoid.map(t => `
          <div class="trend-tag skip-tag" title="${escHtml(t.reason || '')}">
            ${escHtml(t.tag)}
          </div>`).join('')}
      </div>
    </div>`;
  }

  if (topic) {
    html += `<div class="trend-topic">💡 Topic detected: <strong>${escHtml(topic)}</strong> — all hashtags are filtered to this niche only</div>`;
  }

  body.innerHTML = html;
  card.classList.add('fade-in');
}

// ── 10-Day Impression Report ─────────────────────────────────────────────────
function renderTenDaySummary(s) {
  const card = document.getElementById('reportCard');
  if (!s || (!s.total_mid_fmt)) { card.style.display = 'none'; return; }

  card.style.display = '';

  // Tier badge
  const badge = document.getElementById('reportTierBadge');
  badge.textContent = s.tier_label || '—';
  badge.className = 'report-tier-badge ' + (s.tier_label === 'Strong Growth' ? 'high' : s.tier_label === 'Steady Reach' ? 'medium' : 'low');

  // Stats
  document.getElementById('reportMid').textContent   = s.total_mid_fmt   || '—';
  document.getElementById('reportBest').textContent  = s.total_best_fmt  || '—';
  document.getElementById('reportWorst').textContent = s.total_worst_fmt || '—';

  // Meta
  document.getElementById('reportPeakDay').textContent    = s.peak_day || '—';
  document.getElementById('reportReachRate').textContent  = s.daily_reach_rate !== undefined ? s.daily_reach_rate : '—';
  const fc = s.follower_count >= 1000 ? (s.follower_count/1000).toFixed(0)+'K' : s.follower_count;
  document.getElementById('reportFollowers').textContent  = fc || '—';

  // Narrative
  document.getElementById('reportNarrative').textContent = s.narrative || '';

  card.classList.add('fade-in');
}

// ── AI Content Director ──────────────────────────────────────────────────
async function triggerAIDirector(caption, platform, confidence, prediction) {
  const body = document.getElementById('directorBody');
  body.innerHTML = '<div class="director-loading"><div class="gemini-spinner"></div><span>Running model-validated loop… Gemini → Score → Refine → Score</span></div>';

  const fd = new FormData();
  fd.append('caption', caption);
  fd.append('platform', platform);
  fd.append('follower_count', document.getElementById('followerCount').value || 10000);
  if (mediaFile) fd.append('media', mediaFile);

  // ── Send DICE-ML directives so Gemini prescribes based on actual diagnosis ──
  // lastResult.suggestions = the DICE-ML counterfactual suggestions from the analysis
  if (lastResult && lastResult.suggestions && lastResult.suggestions.length > 0) {
    const topDirectives = lastResult.suggestions.slice(0, 3)
      .map((s, i) => `${i + 1}. ${s.suggestion}`)
      .join('\n');
    fd.append('directives', topDirectives);
  }

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
  const mode  = d.mode || (d.hook_variants ? 'optimizer' : 'fixer');
  const trail = d.iteration_trail || [];
  // beatOriginal / recommended / alternative come from the backend's own best-of-N
  // pick — this is the single source of truth for what "won," so the UI never
  // shows a rewrite as the winner when the ML model actually scored it lower.
  const beatOriginal = !!d.beat_original;
  const recommended  = d.recommended || { source: 'original', caption: originalCaption, score_pct: Math.round((d.current_confidence || 0) * 100), delta_pct: 0 };
  const alternative  = d.alternative || null;
  const liftPct      = beatOriginal ? Math.max(0, recommended.delta_pct || 0) : 0;
  const afterConf     = recommended.score_pct / 100;

  // -- Iteration Trail (collapsed by default — process detail, not the headline) --
  let trailHtml = '';
  if (trail.length > 1) {
    const origScore = trail[0]?.score || 0;
    const bestScore = Math.max(...trail.map(t => t.score));
    trailHtml = `
    <details class="iter-details">
      <summary>Show iteration details (${trail.length} steps)</summary>
      <div class="iter-trail">
        <div class="iter-trail-steps">
          ${trail.map((t, i) => {
            const isOrig = i === 0;
            const isBest = t.score === bestScore && (beatOriginal ? !isOrig : isOrig);
            const col    = t.score >= 75 ? '#10b981' : t.score >= 45 ? '#f59e0b' : '#ef4444';
            const delta  = !isOrig ? (t.score > origScore ? `<span class="iter-delta up">+${t.score - origScore}</span>` : t.score < origScore ? `<span class="iter-delta down">${t.score - origScore}</span>` : '') : '';
            return `
            <div class="iter-step ${isBest ? 'iter-best' : ''}">
              <div class="iter-label">${escHtml(t.label)}</div>
              <div class="iter-score" style="color:${col}">${t.score}${delta}</div>
              ${isBest ? '<div class="iter-best-badge">BEST</div>' : ''}
            </div>
            ${i < trail.length - 1 ? '<div class="iter-arrow">→</div>' : ''}`;
          }).join('')}
        </div>
      </div>
    </details>`;
  }

  if (mode === 'optimizer') {
    // ── OPTIMIZER MODE: best → better ────────────────────────────────────────
    const variants = d.hook_variants || [];
    const micro    = d.micro_improvements || [];
    const toAdd    = d.hashtags_to_add || [];
    const toRemove = d.hashtags_to_remove || [];
    const angleColors = { Curiosity: '#6366f1', Authority: '#10b981', Story: '#f59e0b' };

    body.innerHTML = `
      ${trailHtml}
      <div class="optimizer-badge">
        ⚡ Optimizer Mode — Marginal Gains Intelligence
        <span class="optimizer-sub">Your caption is already HIGH. These are the gains Claude can't give you.</span>
      </div>

      ${d.alignment_assessment ? `
      <div class="insight-row">
        <span class="insight-icon">✅</span>
        <div class="insight-text"><strong>What's already excellent:</strong> ${escHtml(d.alignment_assessment)}</div>
      </div>` : ''}

      ${variants.length ? `
      <div class="optimizer-section">
        <div class="optimizer-section-title">🎣 A/B Hook Variants — Split-Test These</div>
        <div class="optimizer-section-sub">3 alternative opening lines. Test which one gets more "See More" clicks.</div>
        ${variants.map((v, i) => {
          const col = angleColors[v.angle] || '#6366f1';
          return `
          <div class="hook-variant">
            <span class="hook-angle-badge" style="background:${col}22;color:${col};border:1px solid ${col}44">${v.angle}</span>
            <div class="hook-variant-text">"${escHtml(v.variant)}"</div>
            <button class="copy-btn-sm" onclick="copyText(${JSON.stringify(v.variant)}, this)">Copy</button>
          </div>`;
        }).join('')}
      </div>` : ''}

      ${micro.length ? `
      <div class="optimizer-section">
        <div class="optimizer-section-title">🔧 Micro-Improvements — Surgical Tweaks Only</div>
        ${micro.map(m => `
        <div class="insight-row">
          <span class="insight-icon">✂️</span>
          <div class="insight-text">${escHtml(m)}</div>
        </div>`).join('')}
      </div>` : ''}

      ${(toAdd.length || toRemove.length) ? `
      <div class="optimizer-section">
        <div class="optimizer-section-title">📈 Hashtag Velocity — Algorithm Signals</div>
        ${toAdd.length ? `<div class="hashtag-velocity-row">
          <span class="velocity-label add">▲ ADD</span>
          ${toAdd.map(t => `<span class="velocity-tag add">${escHtml(t)}</span>`).join('')}
        </div>` : ''}
        ${toRemove.length ? `<div class="hashtag-velocity-row">
          <span class="velocity-label remove">▼ REMOVE</span>
          ${toRemove.map(t => `<span class="velocity-tag remove">${escHtml(t)}</span>`).join('')}
        </div>` : ''}
      </div>` : ''}

      ${d.timing_edge ? `
      <div class="insight-row">
        <span class="insight-icon">⏱️</span>
        <div class="insight-text"><strong>Algorithm timing edge:</strong> ${escHtml(d.timing_edge)}</div>
      </div>` : ''}

      ${d.competitive_gap ? `
      <div class="insight-row" style="border-left:2px solid #f59e0b;padding-left:12px">
        <span class="insight-icon">🏆</span>
        <div class="insight-text"><strong>What top 1% posts do that this still lacks:</strong> ${escHtml(d.competitive_gap)}</div>
      </div>` : ''}

      ${d.vocabulary_suggestion ? `
      <div class="insight-row">
        <span class="insight-icon">🔤</span>
        <div class="insight-text"><strong>Trending vocabulary:</strong> ${escHtml(d.vocabulary_suggestion)}</div>
      </div>` : ''}

      ${liftPct > 0 ? `
      <div class="score-lift">
        <div>
          <div class="lift-label">Marginal gain from applying tweaks</div>
          <div class="lift-val">+${liftPct}%</div>
        </div>
        <div class="lift-arrow">→</div>
        <div>
          <div class="lift-label">New predicted confidence</div>
          <div class="lift-val">${Math.round(afterConf * 100)}%</div>
        </div>
      </div>` : ''}
    `;

  } else {
    // ── FIXER MODE: low/medium — full rewrite ─────────────────────────────────
    // Which caption wins is decided by the backend's actual LightGBM score
    // (d.beat_original), never assumed — so the ✅ never lands on a rewrite that
    // scored lower than what it's being compared against.
    const compareHtml = beatOriginal ? `
      <div class="caption-compare">
        <div class="caption-box">
          <div class="caption-box-label before">Original — ${trail[0]?.score ?? ''}%</div>
          <div class="caption-text before">${escHtml(originalCaption)}</div>
        </div>
        <div class="caption-box">
          <div class="caption-box-label rec">✅ Recommended — ${recommended.score_pct}%</div>
          <div class="caption-text after" style="position:relative">
            ${escHtml(recommended.caption)}
            <button class="copy-btn" onclick="copyText(${JSON.stringify(recommended.caption)}, this)">Copy</button>
          </div>
        </div>
      </div>` : `
      <div class="caption-compare">
        <div class="caption-box">
          <div class="caption-box-label rec">✅ Recommended — Keep Original (${recommended.score_pct}%)</div>
          <div class="caption-text after" style="position:relative">
            ${escHtml(originalCaption)}
            <button class="copy-btn" onclick="copyText(${JSON.stringify(originalCaption)}, this)">Copy</button>
          </div>
        </div>
        ${alternative ? `
        <div class="caption-box">
          <div class="caption-box-label alt">Gemini's Creative Alternative — ${alternative.score_pct}%</div>
          <div class="caption-text alt" style="position:relative">
            ${escHtml(alternative.caption)}
            <button class="copy-btn" onclick="copyText(${JSON.stringify(alternative.caption)}, this)">Copy</button>
          </div>
        </div>` : ''}
      </div>
      ${alternative ? `<div class="alt-note">${escHtml(alternative.note)}</div>` : ''}`;

    body.innerHTML = `
      ${trailHtml}
      ${compareHtml}

      ${d.hook_rewrite ? `
      <div class="insight-row">
        <span class="insight-icon">🎣</span>
        <div class="insight-text"><strong>Hook rewrite:</strong> "${escHtml(d.hook_rewrite)}"</div>
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
          <div class="lift-label">ML-validated score lift</div>
          <div class="lift-val">+${liftPct}%</div>
        </div>
        <div class="lift-arrow">→</div>
        <div>
          <div class="lift-label">New predicted confidence</div>
          <div class="lift-val">${recommended.score_pct}%</div>
        </div>
      </div>` : ''}
    `;
  }
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
