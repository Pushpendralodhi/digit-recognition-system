// ---- Pure-JS forward pass for the trained CNN ------------------------------
// Mirrors src/models.py's build_cnn() exactly:
// Conv2D(32,3x3,relu) -> MaxPool(2x2) -> Conv2D(64,3x3,relu) -> MaxPool(2x2)
// -> Flatten -> Dense(128,relu) -> Dense(10,softmax)
function conv2dValid(x, H, W, Cin, kernel, kh, kw, Cout, bias) {
  const oh = H - kh + 1, ow = W - kw + 1;
  const out = new Float32Array(oh * ow * Cout);
  for (let i = 0; i < oh; i++) {
    for (let j = 0; j < ow; j++) {
      for (let co = 0; co < Cout; co++) {
        let sum = bias[co];
        for (let ki = 0; ki < kh; ki++) {
          for (let kj = 0; kj < kw; kj++) {
            const xBase = ((i + ki) * W + (j + kj)) * Cin;
            const kBase = ((ki * kw + kj) * Cin) * Cout + co;
            for (let ci = 0; ci < Cin; ci++) {
              sum += x[xBase + ci] * kernel[kBase + ci * Cout];
            }
          }
        }
        out[(i * ow + j) * Cout + co] = sum;
      }
    }
  }
  return { data: out, H: oh, W: ow, C: Cout };
}
function reluInPlace(x) { for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0; return x; }
function maxPool2x2(x, H, W, C) {
  const oh = Math.floor(H / 2), ow = Math.floor(W / 2);
  const out = new Float32Array(oh * ow * C);
  for (let i = 0; i < oh; i++) for (let j = 0; j < ow; j++) for (let c = 0; c < C; c++) {
    const a = x[((2*i) * W + (2*j)) * C + c], b = x[((2*i) * W + (2*j+1)) * C + c];
    const cc = x[((2*i+1) * W + (2*j)) * C + c], d = x[((2*i+1) * W + (2*j+1)) * C + c];
    out[(i * ow + j) * C + c] = Math.max(a, b, cc, d);
  }
  return { data: out, H: oh, W: ow, C };
}
function dense(x, inDim, weight, outDim, bias) {
  const out = new Float32Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let sum = bias[o];
    for (let i = 0; i < inDim; i++) sum += x[i] * weight[i * outDim + o];
    out[o] = sum;
  }
  return out;
}
function softmax(x) {
  let max = -Infinity;
  for (let i = 0; i < x.length; i++) if (x[i] > max) max = x[i];
  let sum = 0;
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) { out[i] = Math.exp(x[i] - max); sum += out[i]; }
  for (let i = 0; i < x.length; i++) out[i] /= sum;
  return out;
}
function predictDigit(img, w) {
  let x = { data: img, H: 28, W: 28, C: 1 };
  let c1 = conv2dValid(x.data, x.H, x.W, x.C, w.conv1_k, 3, 3, 32, w.conv1_b);
  reluInPlace(c1.data);
  let p1 = maxPool2x2(c1.data, c1.H, c1.W, c1.C);
  let c2 = conv2dValid(p1.data, p1.H, p1.W, p1.C, w.conv2_k, 3, 3, 64, w.conv2_b);
  reluInPlace(c2.data);
  let p2 = maxPool2x2(c2.data, c2.H, c2.W, c2.C);
  let d1 = dense(p2.data, p2.H * p2.W * p2.C, w.dense1_k, 128, w.dense1_b);
  reluInPlace(d1);
  let d2 = dense(d1, 128, w.dense2_k, 10, w.dense2_b);
  const probs = softmax(d2);
  let best = 0;
  for (let i = 1; i < 10; i++) if (probs[i] > probs[best]) best = i;
  return { digit: best, probabilities: Array.from(probs) };
}

// ---- MNIST-style normalisation (JS twin of src/preprocess.py) --------------
// MNIST digits were cropped to their bounding box, scaled so the longer side is
// 20 px, then centred in 28x28 by centre of mass. Feeding the raw canvas squashed
// to 28x28 makes small / off-centre / oddly-sized drawings fail, so we redo that
// recipe here before the network sees anything.
const COM_TARGET = 14.0, BOX = 20, OUT = 28, BBOX_THRESHOLD = 0.1;

function resizeMatrix(srcLen, dstLen) {
  const s = srcLen / dstLen;
  const W = [];
  for (let i = 0; i < dstLen; i++) {
    const row = new Float64Array(srcLen);
    if (s >= 1) {                       // shrinking: area average
      const lo = i * s, hi = (i + 1) * s;
      let sum = 0;
      for (let j = Math.floor(lo); j < Math.min(Math.ceil(hi), srcLen); j++) {
        const w = Math.max(0, Math.min(hi, j + 1) - Math.max(lo, j));
        row[j] = w; sum += w;
      }
      for (let j = 0; j < srcLen; j++) row[j] /= sum;
    } else {                            // growing: linear interpolation
      const c = (i + 0.5) * s - 0.5, j0 = Math.floor(c), f = c - j0;
      const a = Math.min(Math.max(j0, 0), srcLen - 1), b = Math.min(Math.max(j0 + 1, 0), srcLen - 1);
      row[a] += 1 - f; row[b] += f;
    }
    W.push(row);
  }
  return W;
}
function resizeArea(img, h, w, dh, dw) {
  const Wy = resizeMatrix(h, dh), Wx = resizeMatrix(w, dw);
  const tmp = new Float64Array(dh * w);             // rows resized
  for (let i = 0; i < dh; i++) for (let k = 0; k < h; k++) {
    const wy = Wy[i][k]; if (wy === 0) continue;
    for (let j = 0; j < w; j++) tmp[i * w + j] += wy * img[k * w + j];
  }
  const out = new Float64Array(dh * dw);            // then columns
  for (let i = 0; i < dh; i++) for (let j = 0; j < dw; j++) {
    let acc = 0; const wx = Wx[j];
    for (let k = 0; k < w; k++) if (wx[k] !== 0) acc += wx[k] * tmp[i * w + k];
    out[i * dw + j] = acc;
  }
  return out;
}
// gray: Float32Array(H*W), values 0-1, white strokes on black. Returns Float32Array(784) or null if blank.
// round-half-to-even, identical to Python/NumPy round() so ties land on the same pixel
function roundEven(v) { const f = Math.floor(v), d = v - f; return d < 0.5 ? f : d > 0.5 ? f + 1 : (f % 2 === 0 ? f : f + 1); }
function normalizeDigit(gray, H, W) {
  let m = 0;
  for (let i = 0; i < gray.length; i++) if (gray[i] > m) m = gray[i];
  if (m <= 0.05) return null;
  const thr = BBOX_THRESHOLD * m;
  let r0 = H, r1 = -1, c0 = W, c1 = -1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (gray[y * W + x] > thr) {
    if (y < r0) r0 = y; if (y > r1) r1 = y; if (x < c0) c0 = x; if (x > c1) c1 = x;
  }
  const h = r1 - r0 + 1, w = c1 - c0 + 1;
  const crop = new Float64Array(h * w);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) crop[y * w + x] = gray[(r0 + y) * W + (c0 + x)];
  const scale = BOX / Math.max(h, w);
  const nh = Math.max(1, roundEven(h * scale)), nw = Math.max(1, roundEven(w * scale));
  const small = resizeArea(crop, h, w, nh, nw);

  const placed = new Float64Array(OUT * OUT);
  const top = Math.floor((OUT - nh) / 2), left = Math.floor((OUT - nw) / 2);
  for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) placed[(top + y) * OUT + left + x] = small[y * nw + x];

  let total = 0, cy = 0, cx = 0;
  for (let y = 0; y < OUT; y++) for (let x = 0; x < OUT; x++) { const v = placed[y * OUT + x]; total += v; cy += v * y; cx += v * x; }
  const out = new Float32Array(OUT * OUT);
  let dy = 0, dx = 0;
  if (total > 0) { dy = roundEven(COM_TARGET - cy / total); dx = roundEven(COM_TARGET - cx / total); }
  for (let y = 0; y < OUT; y++) for (let x = 0; x < OUT; x++) {
    const sy = y - dy, sx = x - dx;
    if (sy >= 0 && sy < OUT && sx >= 0 && sx < OUT) out[y * OUT + x] = Math.min(1, Math.max(0, placed[sy * OUT + sx]));
  }
  return out;
}


// ---- Pen normalisation (JS twin of preprocess.pen_normalize) ---------------
// A canvas drawing is a constant-width pen line, but its width *relative to the digit*
// changes with how big you draw: small digits look fat, loops fill in, a 6 reads as 0.
// MNIST digits always have the same slim stroke, so we (1) scale the drawing to a fixed
// working size, (2) thin every stroke to a 1-pixel centre line (Zhang-Suen thinning),
// (3) redraw that line with one fixed MNIST-like pen. Pen size and digit size stop mattering.
const WORK = 96, PEN = 7;

function zhangSuenThin(bin, h, w) {                 // bin: Uint8Array(h*w) of 0/1 -> thinned Uint8Array(h*w)
  const H = h + 2, W = w + 2, img = new Uint8Array(H * W);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) img[(y + 1) * W + x + 1] = bin[y * w + x];
  let changed = true;
  while (changed) {
    changed = false;
    for (let step = 0; step < 2; step++) {
      const kill = [];
      for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
        const i = y * W + x;
        if (!img[i]) continue;
        const P2 = img[i - W], P3 = img[i - W + 1], P4 = img[i + 1], P5 = img[i + W + 1];
        const P6 = img[i + W], P7 = img[i + W - 1], P8 = img[i - 1], P9 = img[i - W - 1];
        const B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9;
        if (B < 2 || B > 6) continue;
        const A = (!P2 && P3) + (!P3 && P4) + (!P4 && P5) + (!P5 && P6) + (!P6 && P7) + (!P7 && P8) + (!P8 && P9) + (!P9 && P2);
        if (A !== 1) continue;
        const ok = step === 0 ? (P2 * P4 * P6 === 0 && P4 * P6 * P8 === 0) : (P2 * P4 * P8 === 0 && P2 * P6 * P8 === 0);
        if (ok) kill.push(i);
      }
      if (kill.length) { for (const i of kill) img[i] = 0; changed = true; }
    }
  }
  const out = new Uint8Array(h * w);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = img[(y + 1) * W + x + 1];
  return out;
}
function discDilate(bin, h, w, diameter) {          // redraw every 1-pixel as a filled disc
  const r = diameter / 2, ri = Math.ceil(r), offs = [];
  for (let dy = -ri; dy <= ri; dy++) for (let dx = -ri; dx <= ri; dx++)
    if (dy * dy + dx * dx <= (r - 0.5) * (r - 0.5) + 0.25 || (dy === 0 && dx === 0)) offs.push([dy, dx]);
  const out = new Float32Array(h * w);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (bin[y * w + x])
    for (const [dy, dx] of offs) { const yy = y + dy, xx = x + dx; if (yy >= 0 && yy < h && xx >= 0 && xx < w) out[yy * w + xx] = 1; }
  return out;
}
// gray: Float32Array(H*W) 0-1, white strokes on black. Returns {data, h, w} (slim 0/1 drawing) or null if blank.
function penNormalize(gray, H, W) {
  let m = 0;
  for (let i = 0; i < gray.length; i++) if (gray[i] > m) m = gray[i];
  if (m <= 0.05) return null;
  const thr = BBOX_THRESHOLD * m;
  let r0 = H, r1 = -1, c0 = W, c1 = -1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (gray[y * W + x] > thr) {
    if (y < r0) r0 = y; if (y > r1) r1 = y; if (x < c0) c0 = x; if (x > c1) c1 = x;
  }
  const h = r1 - r0 + 1, w = c1 - c0 + 1;
  const crop = new Float64Array(h * w);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) crop[y * w + x] = gray[(r0 + y) * W + (c0 + x)];
  const s = WORK / Math.max(h, w);
  const nh = Math.max(3, roundEven(h * s)), nw = Math.max(3, roundEven(w * s));
  const big = resizeArea(crop, h, w, nh, nw);
  const bin = new Uint8Array(nh * nw);
  for (let i = 0; i < bin.length; i++) bin[i] = big[i] > 0.5 * m ? 1 : 0;
  const thin = zhangSuenThin(bin, nh, nw);
  return { data: discDilate(thin, nh, nw, PEN), h: nh, w: nw };
}

