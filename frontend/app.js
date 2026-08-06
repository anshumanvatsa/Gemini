/* ============================================================
   PreViral — app.js
   Handles form submission, API calls, and all UI rendering
   ============================================================ */

const API_BASE = window.location.origin + '/api/v1';
let selectedPlatform = 'instagram';

// ── Platform Selector ─────────────────────────────────────────
document.querySelectorAll('.platform-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.platform-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    selectedPlatform = btn.dataset.platform;
    document.getElementById('selectedPlatform').value = selectedPlatform;
  });
});

// ── Caption live counter ──────────────────────────────────────
const captionEl = document.getElementById('caption');
captionEl.addEventListener('input', () => {
  const text = captionEl.value;
  const hashtags = (text.match(/#\w+/g) || []).length;
  document.getElementById('charCount').textContent = `${text.length} characters`;
  document.getElementById('hashtagCount').textContent = `${hashtags} hashtag${hashtags !== 1 ? 's' : ''}`;
});

// ── Image Upload Preview + Async Vision Preprocessing ────────
let visionCacheId = null;
let visionProcessing = false;

async function preprocessImage(file, platform) {
  if (!file || !file.type.startsWith('image/')) return;
  visionProcessing = true;

  // Show "Analyzing thumbnail..." indicator
  const uploadHint = document.querySelector('.upload-hint');
  if (uploadHint) uploadHint.textContent = 'Analyzing thumbnail (face detection, CLIP)...';

  try {
    const fd = new FormData();
    fd.append('media', file);
    fd.append('platform', selectedPlatform);
    const res = await fetch(`${API_BASE}/preprocess-media`, { method: 'POST', body: fd });
    if (res.ok) {
      const data = await res.json();
      visionCacheId = data.vision_cache_id;
      if (uploadHint) uploadHint.textContent = 'Thumbnail analyzed! Face detection + CLIP ready.';
    }
  } catch (e) {
    console.warn('Vision preprocessing failed (will run at analyze time):', e);
  } finally {
    visionProcessing = false;
  }
}

mediaInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file && file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = (ev) => {
      imagePreview.src = ev.target.result;
      imagePreview.classList.remove('hidden');
      uploadContent.classList.add('hidden');
    };
    reader.readAsDataURL(file);
    // Fire async CLIP preprocessing immediately — don't wait for Analyze
    preprocessImage(file, selectedPlatform);
  }
});

uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => { uploadZone.classList.remove('drag-over'); });
uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) { mediaInput.files = e.dataTransfer.files; mediaInput.dispatchEvent(new Event('change')); }
});


// ── Set default datetime to now + 2 hours ────────────────────
const dtInput = document.getElementById('postDatetime');
const now = new Date();
now.setHours(now.getHours() + 2);
dtInput.value = now.toISOString().slice(0, 16);

// ── Form Submit ───────────────────────────────────────────────
const form = document.getElementById('analyzeForm');
const analyzeBtn = document.getElementById('analyzeBtn');
const btnText = analyzeBtn.querySelector('.btn-text');
const btnIcon = analyzeBtn.querySelector('.btn-icon');
const btnLoading = analyzeBtn.querySelector('.btn-loading');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  setLoading(true);

  try {
    const formData = new FormData(form);
    // Ensure platform is set
    formData.set('platform', selectedPlatform);
    // Inject pre-computed vision cache id (CLIP ran during thumbnail upload)
    if (visionCacheId) {
      formData.set('vision_cache_id', visionCacheId);
    }
    // Convert engagement rate from % to decimal
    const er = parseFloat(formData.get('avg_engagement_rate')) / 100;
    formData.set('avg_engagement_rate', er.toString());

    const res = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error: ${res.status}`);
    }

    const data = await res.json();
    renderResults(data);

  } catch (err) {
    showToast(err.message || 'Analysis failed. Make sure the API is running.');
    console.error(err);
  } finally {
    setLoading(false);
  }
});

function setLoading(loading) {
  analyzeBtn.disabled = loading;
  btnText.classList.toggle('hidden', loading);
  btnIcon.classList.toggle('hidden', loading);
  btnLoading.classList.toggle('hidden', !loading);
}

// ── Render Results ────────────────────────────────────────────
function renderResults(data) {
  document.getElementById('inputPanel').classList.add('hidden');
  const resultsPanel = document.getElementById('resultsPanel');
  resultsPanel.classList.remove('hidden');
  resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Score Card
  const badge = document.getElementById('predictionBadge');
  badge.textContent = data.prediction;
  badge.className = 'prediction-badge ' + data.prediction.toLowerCase();

  const pct = document.getElementById('confidencePct');
  animateNumber(pct, 0, Math.round(data.confidence * 100), 1000, '%');

  // Confidence Ring
  const arc = document.getElementById('confidenceArc');
  const circumference = 326.73;
  const offset = circumference - (data.confidence * circumference);
  setTimeout(() => {
    arc.style.transition = 'stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1)';
    arc.style.strokeDashoffset = offset;
  }, 100);

  document.getElementById('scoreHeadline').textContent = data.headline;
  document.getElementById('reachPercentile').textContent = `Top ${100 - data.reach_percentile}%`;
  document.getElementById('processingTime').textContent = `${data.processing_time_ms.toFixed(0)}ms`;
  document.getElementById('platformBadge').textContent = data.platform.charAt(0).toUpperCase() + data.platform.slice(1);

  // Trajectory Chart
  renderTrajectory(data.trajectory, data.prediction);

  // Suggestions
  renderSuggestions(data.suggestions);

  // Feature Breakdown
  renderFeatures({...data.nlp_features, ...data.hashtag_features, ...data.vision_features, ...data.timing_features});

  // Hashtags
  renderHashtags(data.hashtag_suggestions);
}

// ── Trajectory Chart (Canvas) ─────────────────────────────────
function renderTrajectory(trajectory, prediction) {
  const canvas = document.getElementById('trajectoryCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const days = trajectory.map(t => t.day);
  const mids = trajectory.map(t => t.mid);
  const lows = trajectory.map(t => t.low);
  const highs = trajectory.map(t => t.high);

  const maxVal = Math.max(...highs) || 1;
  const pad = { top: 20, right: 20, bottom: 30, left: 60 };
  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top - pad.bottom;

  const xScale = (i) => pad.left + (i / (trajectory.length - 1)) * chartW;
  const yScale = (v) => pad.top + chartH - (v / maxVal) * chartH;

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  [0.25, 0.5, 0.75, 1.0].forEach(frac => {
    const y = yScale(maxVal * frac);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    ctx.fillStyle = 'rgba(148,163,184,0.5)'; ctx.font = '11px Inter';
    ctx.textAlign = 'right'; ctx.fillText(fmtNum(maxVal * frac), pad.left - 8, y + 4);
  });

  // Confidence band (fill between low and high)
  ctx.beginPath();
  highs.forEach((v, i) => i === 0 ? ctx.moveTo(xScale(i), yScale(v)) : ctx.lineTo(xScale(i), yScale(v)));
  [...lows].reverse().forEach((v, i) => ctx.lineTo(xScale(lows.length - 1 - i), yScale(v)));
  ctx.closePath();
  const bandColor = prediction === 'HIGH' ? 'rgba(168,85,247,0.12)' : 'rgba(249,115,22,0.08)';
  ctx.fillStyle = bandColor;
  ctx.fill();

  // Mid line
  const lineColor = prediction === 'HIGH' ? '#a855f7' : '#f97316';
  const grad = ctx.createLinearGradient(pad.left, 0, W - pad.right, 0);
  grad.addColorStop(0, prediction === 'HIGH' ? '#a855f7' : '#f97316');
  grad.addColorStop(1, prediction === 'HIGH' ? '#06b6d4' : '#ef4444');

  ctx.beginPath();
  mids.forEach((v, i) => i === 0 ? ctx.moveTo(xScale(i), yScale(v)) : ctx.lineTo(xScale(i), yScale(v)));
  ctx.strokeStyle = grad; ctx.lineWidth = 2.5; ctx.lineJoin = 'round';
  ctx.stroke();

  // Dots
  mids.forEach((v, i) => {
    ctx.beginPath();
    ctx.arc(xScale(i), yScale(v), 5, 0, Math.PI * 2);
    ctx.fillStyle = prediction === 'HIGH' ? '#a855f7' : '#f97316';
    ctx.fill();
    ctx.strokeStyle = 'rgba(5,5,8,0.8)'; ctx.lineWidth = 2; ctx.stroke();
  });

  // Day labels
  const labels = document.getElementById('trajectoryLabels');
  labels.innerHTML = '';
  trajectory.forEach(t => {
    const span = document.createElement('span');
    span.textContent = `Day ${t.day}`;
    labels.appendChild(span);
  });
}

function fmtNum(n) {
  if (n >= 1000000) return (n/1000000).toFixed(1) + 'M';
  if (n >= 1000) return (n/1000).toFixed(0) + 'K';
  return Math.round(n).toString();
}

// ── Suggestions ───────────────────────────────────────────────
function renderSuggestions(suggestions) {
  const list = document.getElementById('suggestionsList');
  list.innerHTML = '';
  if (!suggestions || suggestions.length === 0) {
    list.innerHTML = '<p style="color:var(--text-muted);font-size:13px">Your post looks good! No major changes needed.</p>';
    return;
  }
  suggestions.forEach((s, i) => {
    const div = document.createElement('div');
    div.className = 'suggestion-item';
    div.style.animationDelay = `${i * 0.1}s`;
    div.innerHTML = `
      <div class="suggestion-header">
        <span class="suggestion-feature">${s.feature.replace(/_/g, ' ')}</span>
        <span class="suggestion-impact">${s.estimated_impact}</span>
      </div>
      <p class="suggestion-text">${s.suggestion}</p>
    `;
    list.appendChild(div);
  });
}

// ── Feature Breakdown ─────────────────────────────────────────
function renderFeatures(features) {
  const grid = document.getElementById('featuresGrid');
  grid.innerHTML = '';

  const DISPLAY = {
    sentiment_score: 'Sentiment',
    emotional_valence: 'Emotional Valence',
    clickbait_score: 'Hook Strength',
    cta_present: 'Call-to-Action',
    avg_competition_ratio: 'Hashtag Quality',
    trending_hashtag_count: 'Trending Tags',
    peak_overlap_score: 'Timing Score',
    face_count: 'Face Detected',
    brightness_score: 'Brightness',
    color_vibrancy: 'Vibrancy',
  };

  Object.entries(DISPLAY).forEach(([key, label]) => {
    let raw = features[key];
    if (raw === undefined || raw === null) return;

    // Normalize to 0-1 for the bar
    let normalized = raw;
    if (key === 'avg_competition_ratio') {
      normalized = 1 - raw; // Lower competition = better
    } else if (key === 'trending_hashtag_count') {
      normalized = Math.min(raw / 5, 1);
    } else if (key === 'face_count') {
      normalized = Math.min(raw / 3, 1);
    } else if (key === 'cta_present') {
      normalized = raw;
    } else {
      normalized = Math.max(0, Math.min(raw, 1));
    }

    const display = typeof raw === 'number' ? (raw % 1 === 0 ? raw : raw.toFixed(2)) : raw;

    const row = document.createElement('div');
    row.className = 'feature-row';
    row.innerHTML = `
      <span class="feature-name">${label}</span>
      <div class="feature-bar-wrap">
        <div class="feature-bar" style="width: ${(normalized * 100).toFixed(0)}%"></div>
      </div>
      <span class="feature-val">${display}</span>
    `;
    grid.appendChild(row);
  });
}

// ── Hashtags ──────────────────────────────────────────────────
function renderHashtags(hashtags) {
  const list = document.getElementById('hashtagList');
  list.innerHTML = '';
  if (!hashtags || hashtags.length === 0) {
    list.innerHTML = '<p style="color:var(--text-muted);font-size:13px">No hashtag suggestions available for this niche yet.</p>';
    return;
  }
  hashtags.forEach(h => {
    const div = document.createElement('div');
    div.className = 'hashtag-item';
    div.title = `Relevance: ${(h.relevance_score * 100).toFixed(0)}% | Competition: ${(h.competition_score * 100).toFixed(0)}%`;
    div.innerHTML = `
      <span class="hashtag-tag">${h.hashtag}</span>
      <div class="hashtag-meta">
        <span class="hashtag-comp">comp: ${(h.competition_score * 100).toFixed(0)}%</span>
        <span class="hashtag-status ${h.trend_status}">${h.trend_status}</span>
      </div>
    `;
    div.addEventListener('click', () => {
      // Insert hashtag into caption
      const cap = document.getElementById('caption');
      cap.value = cap.value.trimEnd() + ' ' + h.hashtag;
      cap.dispatchEvent(new Event('input'));
      div.style.background = 'rgba(168,85,247,0.1)';
      setTimeout(() => div.style.background = '', 600);
    });
    list.appendChild(div);
  });
}

// ── Analyze Again ─────────────────────────────────────────────
document.getElementById('analyzeAgainBtn').addEventListener('click', () => {
  document.getElementById('resultsPanel').classList.add('hidden');
  document.getElementById('inputPanel').classList.remove('hidden');
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

// ── Number Animation ──────────────────────────────────────────
function animateNumber(el, start, end, duration, suffix = '') {
  const startTime = performance.now();
  const update = (time) => {
    const progress = Math.min((time - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(start + (end - start) * eased) + suffix;
    if (progress < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

// ── Toast ─────────────────────────────────────────────────────
function showToast(msg) {
  const toast = document.getElementById('errorToast');
  document.getElementById('toastMsg').textContent = msg;
  toast.classList.remove('hidden');
  setTimeout(() => toast.classList.add('hidden'), 5000);
}
