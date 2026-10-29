// ---- UI: drawing canvas, prediction, model loading ---------------------------
const $ = id => document.getElementById(id);
const canvas = $('canvas'), ctx = canvas.getContext('2d');
const resultArea = $('result-area'), sampleNote = $('sample-note');
const predictBtn = $('predict-btn'), sampleBtn = $('sample-btn'), autoBox = $('auto-predict');
const statusText = $('status-text'), statusBadge = $('status-badge');
let drawing = false, hasDrawn = false, weights = null, samples = null;
let currentTrueLabel = null, currentSampleGray = null, autoTimer = null;

const EMPTY = '<div class="placeholder"><svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 013 3L7 19l-4 1 1-4z"/></svg><p>Draw a digit on the board, then press <b>Predict</b> (or just lift your finger/mouse).</p></div>';

function resetCanvas() {
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 18; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.strokeStyle = '#FFFFFF';
  hasDrawn = false;
}
resetCanvas();

function pos(e) {
  const r = canvas.getBoundingClientRect();
  return { x: (e.clientX - r.left) * canvas.width / r.width, y: (e.clientY - r.top) * canvas.height / r.height };
}
function onDown(e) {
  if (e.button > 0) return;
  clearTimeout(autoTimer);
  if (currentSampleGray || !hasDrawn) { resetCanvas(); }
  drawing = true; hasDrawn = true; currentTrueLabel = null; currentSampleGray = null; sampleNote.innerHTML = '&nbsp;';
  try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
  const p = pos(e); ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x + 0.01, p.y); ctx.stroke();
  e.preventDefault();
}
function onMove(e) {
  if (!drawing) return;
  const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
  for (const ev of (evs.length ? evs : [e])) { const p = pos(ev); ctx.lineTo(p.x, p.y); ctx.stroke(); }
  e.preventDefault();
}
function onUp() {
  if (!drawing) return;
  drawing = false;
  if (autoBox.checked && weights) { clearTimeout(autoTimer); autoTimer = setTimeout(runPredict, 700); }
}
canvas.addEventListener('pointerdown', onDown);
canvas.addEventListener('pointermove', onMove);
canvas.addEventListener('pointerup', onUp);
canvas.addEventListener('pointercancel', onUp);
canvas.addEventListener('contextmenu', e => e.preventDefault());

$('clear-btn').addEventListener('click', () => {
  clearTimeout(autoTimer); resetCanvas();
  currentTrueLabel = null; currentSampleGray = null; sampleNote.innerHTML = '&nbsp;';
  resultArea.innerHTML = EMPTY;
});
predictBtn.addEventListener('click', () => { clearTimeout(autoTimer); runPredict(); });

function runPredict() {
  if (!hasDrawn || !weights) return;
  let gray, stage = null;
  if (currentSampleGray) {
    gray = currentSampleGray;
  } else {
    const W = canvas.width, H = canvas.height;
    const px = ctx.getImageData(0, 0, W, H).data;
    const full = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++) full[i] = px[i * 4] / 255;
    const slim = penNormalize(full, H, W);
    gray = slim ? normalizeDigit(slim.data, slim.h, slim.w) : null;
    if (gray === null) return;
    stage = slim;
  }
  const t0 = performance.now();
  const result = predictDigit(gray, weights);
  renderResult(result, currentTrueLabel, (performance.now() - t0).toFixed(1), gray, stage);
}

function paint(cv, data, w, h) {
  cv.width = w; cv.height = h;
  const c = cv.getContext('2d'), id = c.createImageData(w, h);
  for (let p = 0; p < w * h; p++) { const v = Math.round(Math.min(1, Math.max(0, data[p])) * 255); id.data[p*4] = v; id.data[p*4+1] = v; id.data[p*4+2] = v; id.data[p*4+3] = 255; }
  c.putImageData(id, 0, 0);
}

function renderResult(result, trueLabel, ms, seen, stage) {
  const { digit: pred, probabilities } = result;
  const maxP = Math.max(...probabilities);
  const order = probabilities.map((p, d) => [p, d]).sort((a, b) => b[0] - a[0]);
  const second = order[1];
  let bars = '';
  for (let d = 0; d < 10; d++) {
    const pct = (probabilities[d] * 100).toFixed(1);
    bars += `<div class="bar-row ${d === pred ? 'top' : ''}"><div class="digit-label">${d}</div><div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div><div class="pct">${pct}%</div></div>`;
  }
  let verdict = '';
  if (trueLabel !== null) {
    const ok = trueLabel === pred;
    verdict = `<div class="verdict ${ok ? 'ok' : 'miss'}">${ok ? '✓ matches the true label' : '✗ true label was ' + trueLabel}</div>`;
  } else if (maxP < 0.6) {
    verdict = `<div class="verdict miss">Not sure — could be ${pred} or ${second[1]}. Try drawing it larger and clearer.</div>`;
  }
  resultArea.innerHTML = `
    <div class="result-hero">
      <div class="result-digit" aria-live="polite">${pred}</div>
      <div class="result-meta">
        <div class="conf">${(maxP * 100).toFixed(1)}% confident</div>
        <div class="label">Predicted digit · ${ms} ms in your browser</div>
        ${verdict}
      </div>
    </div>
    <div class="stages">
      <figure><canvas id="st-pen" ${stage ? '' : 'hidden'}></canvas><figcaption>${stage ? 'Pen line' : ''}</figcaption></figure>
      <span class="arrow" ${stage ? '' : 'hidden'}>→</span>
      <figure><canvas id="st-seen"></canvas><figcaption>28×28 input</figcaption></figure>
      <p>${stage ? 'Your strokes were thinned to a single pen line, then cropped, scaled and centred exactly like MNIST digits.' : 'A real MNIST test image goes in as-is.'}</p>
    </div>
    <div class="bars">${bars}</div>`;
  if (stage) paint($('st-pen'), stage.data, stage.w, stage.h);
  paint($('st-seen'), seen, 28, 28);
}

// ---- model + samples (plain <script> data, so it also works from file://) -----
function b64(s) { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u; }
function loadModel() {
  const M = window.DIGIT_MODEL;
  if (!M) throw new Error('model-data.js did not load');
  const flat = new Float32Array(b64(M.weights).buffer);
  const g = (e, k) => { const w = M.manifest[e].weights[k]; return flat.slice(w.offset, w.offset + w.count); };
  weights = { conv1_k: g(0,0), conv1_b: g(0,1), conv2_k: g(2,0), conv2_b: g(2,1), dense1_k: g(5,0), dense1_b: g(5,1), dense2_k: g(7,0), dense2_b: g(7,1) };
  samples = { labels: M.samples.labels, pixels: b64(M.samples.data) };
}
try {
  loadModel();
  statusText.textContent = 'Model ready'; predictBtn.disabled = false; sampleBtn.disabled = false;
  statusBadge.dataset.state = 'ok';
} catch (err) {
  statusText.textContent = 'Model failed to load — refresh the page'; statusBadge.dataset.state = 'err'; console.error(err);
}

sampleBtn.addEventListener('click', () => {
  if (!samples) return;
  clearTimeout(autoTimer);
  const i = Math.floor(Math.random() * samples.labels.length), off = i * 784;
  currentSampleGray = new Float32Array(784);
  for (let p = 0; p < 784; p++) currentSampleGray[p] = samples.pixels[off + p] / 255;
  const small = document.createElement('canvas'); paint(small, currentSampleGray, 28, 28);
  ctx.imageSmoothingEnabled = false; ctx.fillStyle = '#000'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(small, 0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 18; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.strokeStyle = '#FFFFFF';
  hasDrawn = true; currentTrueLabel = samples.labels[i];
  sampleNote.innerHTML = `Loaded a real MNIST test image — true label: <b>${currentTrueLabel}</b>`;
  runPredict();
});

resultArea.innerHTML = EMPTY;
if ('serviceWorker' in navigator && /^https?:$/.test(location.protocol)) {
  window.addEventListener('load', () => navigator.serviceWorker.register('sw.js').catch(() => {}));
}
