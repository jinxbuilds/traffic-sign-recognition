// Faithful JavaScript port of the training/inference preprocessing:
//   cv2.resize(img, (32, 32))  (INTER_LINEAR, uint8)
//   grayscale = 0.299 R + 0.587 G + 0.114 B
//   / 255  (float32)
//   skimage.exposure.equalize_adapthist (CLAHE, kernel 4x4, clip 0.01, 256 bins)
// Verified bit-exact against OpenCV 4.10 + scikit-image 0.24 (web/verify/).

const f32 = Math.fround;
const SIZE = 32;
const NR_OF_GRAY = 16384;

/** numpy-style round half to even. */
function rint(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** OpenCV saturate_cast<short>(float) uses cvRound (half to even). */
const cvRound = rint;

/** cv2.resize(src, (32, 32)) with INTER_LINEAR on an RGB uint8 image. */
export function resizeLinear(rgb, w, h, dw = SIZE, dh = SIZE) {
  const cn = 3;
  const out = new Uint8Array(dw * dh * cn);
  // OpenCV switches to INTER_AREA for an exact 2x downscale.
  if (w === dw * 2 && h === dh * 2) {
    for (let y = 0; y < dh; y++)
      for (let x = 0; x < dw; x++)
        for (let c = 0; c < cn; c++) {
          const i = (2 * y * w + 2 * x) * cn + c;
          const s = rgb[i] + rgb[i + cn] + rgb[i + w * cn] + rgb[i + w * cn + cn];
          out[(y * dw + x) * cn + c] = (s + 2) >> 2;
        }
    return out;
  }
  const SCALE = 2048; // INTER_RESIZE_COEF_SCALE
  const sxTab = [], axTab = [];
  const scaleX = w / dw, scaleY = h / dh;
  for (let dx = 0; dx < dw; dx++) {
    let fx = (dx + 0.5) * scaleX - 0.5;
    let sx = Math.floor(fx);
    fx -= sx;
    if (sx < 0) { fx = 0; sx = 0; }
    if (sx >= w - 1) { fx = 0; sx = w - 1; }
    sxTab.push(sx);
    axTab.push([cvRound((1 - fx) * SCALE), cvRound(fx * SCALE)]);
  }
  const rowAt = (sy) => {
    const row = new Int32Array(dw * cn);
    for (let dx = 0; dx < dw; dx++) {
      const sx = sxTab[dx], [a0, a1] = axTab[dx];
      const sx1 = Math.min(sx + 1, w - 1);
      for (let c = 0; c < cn; c++) {
        const p0 = rgb[(sy * w + sx) * cn + c];
        const p1 = rgb[(sy * w + sx1) * cn + c];
        row[dx * cn + c] = p0 * a0 + (a1 ? p1 * a1 : 0);
      }
    }
    return row;
  };
  for (let dy = 0; dy < dh; dy++) {
    let fy = (dy + 0.5) * scaleY - 0.5;
    let sy = Math.floor(fy);
    fy -= sy;
    if (sy < 0) { fy = 0; sy = 0; }
    if (sy >= h - 1) { fy = 0; sy = h - 1; }
    const b0 = cvRound((1 - fy) * SCALE), b1 = cvRound(fy * SCALE);
    const r0 = rowAt(sy), r1 = rowAt(Math.min(sy + 1, h - 1));
    for (let i = 0; i < dw * cn; i++) {
      // Matches OpenCV's vectorised VResizeLinear for 8-bit output.
      const v = (((b0 * (r0[i] >> 4)) >> 16) + ((b1 * (r1[i] >> 4)) >> 16) + 2) >> 2;
      out[dy * dw * cn + i] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
  return out;
}

function clipHistogram(hist, clip) {
  let nExcess = 0;
  for (let i = 0; i < hist.length; i++) if (hist[i] > clip) { nExcess += hist[i] - clip; hist[i] = clip; }
  const binIncr = Math.floor(nExcess / hist.length);
  const upper = clip - binIncr;
  for (let i = 0; i < hist.length; i++) if (hist[i] < upper) { nExcess -= binIncr; hist[i] += binIncr; }
  for (let i = 0; i < hist.length; i++)
    if (hist[i] >= upper && hist[i] < clip) { nExcess += hist[i] - clip; hist[i] = clip; }
  while (nExcess > 0) {
    const prev = nExcess;
    for (let index = 0; index < hist.length; index++) {
      let under = 0;
      for (let i = 0; i < hist.length; i++) if (hist[i] < clip) under++;
      const step = Math.max(1, Math.floor(under / nExcess));
      const snapshot = hist.map((v) => v < clip);
      for (let i = index; i < hist.length; i += step) if (snapshot[i]) { hist[i] += 1; nExcess -= 1; }
      if (nExcess <= 0) break;
    }
    if (prev === nExcess) break;
  }
  return hist;
}

/** skimage.exposure.equalize_adapthist on a 2-D float32 image in [0, 1]. */
export function equalizeAdapthist(img, h, w) {
  // img_as_uint (float32 math, rint), then rescale to 0..16383 (float64), round.
  const u = new Float64Array(h * w);
  let imin = Infinity, imax = -Infinity;
  for (let i = 0; i < h * w; i++) {
    const v = Math.min(65535, Math.max(0, rint(f32(img[i] * 65535))));
    u[i] = v;
    if (v < imin) imin = v;
    if (v > imax) imax = v;
  }
  const g = new Int32Array(h * w);
  for (let i = 0; i < h * w; i++)
    g[i] = imin !== imax ? rint(((u[i] - imin) / (imax - imin)) * (NR_OF_GRAY - 1)) : Math.min(u[i], NR_OF_GRAY - 1);

  const kh = Math.max(Math.floor(h / 8), 1), kw = Math.max(Math.floor(w / 8), 1);
  const ps = [Math.floor(kh / 2), Math.floor(kw / 2)];
  const pe = [((kh - (h % kh)) % kh) + Math.ceil(kh / 2), ((kw - (w % kw)) % kw) + Math.ceil(kw / 2)];
  const H = h + ps[0] + pe[0], W = w + ps[1] + pe[1];
  const reflect = (i, n) => {
    if (n === 1) return 0;
    const p = 2 * (n - 1);
    i = ((i % p) + p) % p;
    return i < n ? i : p - i;
  };
  const binSize = 1 + Math.floor(NR_OF_GRAY / 256);
  const pad = new Int32Array(H * W);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++)
      pad[y * W + x] = Math.floor(g[reflect(y - ps[0], h) * w + reflect(x - ps[1], w)] / binSize);

  const nh = [Math.trunc(H / kh) - 1, Math.trunc(W / kw) - 1];
  const kernelElements = kh * kw;
  const clim = Math.max(1, Math.trunc(0.01 * kernelElements));
  // map_array, padded by one block with edge values: (nh0 + 2) x (nh1 + 2) x 256
  const maps = [];
  for (let by = 0; by < nh[0]; by++) {
    maps.push([]);
    for (let bx = 0; bx < nh[1]; bx++) {
      const hist = new Array(256).fill(0);
      for (let y = 0; y < kh; y++)
        for (let x = 0; x < kw; x++)
          hist[pad[(Math.floor(kh / 2) + by * kh + y) * W + Math.floor(kw / 2) + bx * kw + x]]++;
      clipHistogram(hist, clim);
      const m = new Int32Array(256);
      let cum = 0;
      for (let i = 0; i < 256; i++) {
        cum += hist[i];
        m[i] = Math.trunc(Math.min(cum * ((NR_OF_GRAY - 1) / kernelElements), NR_OF_GRAY - 1));
      }
      maps[by].push(m);
    }
  }
  const mapAt = (y, x) => maps[Math.min(Math.max(y - 1, 0), nh[0] - 1)][Math.min(Math.max(x - 1, 0), nh[1] - 1)];

  const res = new Float32Array(H * W);
  const np = [Math.trunc(H / kh), Math.trunc(W / kw)];
  for (let by = 0; by < np[0]; by++)
    for (let bx = 0; bx < np[1]; bx++)
      for (let py = 0; py < kh; py++)
        for (let px = 0; px < kw; px++) {
          const y = by * kh + py, x = bx * kw + px;
          const bin = pad[y * W + x];
          const cy = py / kh, cx = px / kw;
          let acc = 0;
          for (const [e0, e1] of [[0, 0], [0, 1], [1, 0], [1, 1]]) {
            const coeff = (e1 ? cx : 1 - cx) * (e0 ? cy : 1 - cy);
            acc = f32(acc + f32(mapAt(by + e0, bx + e1)[bin] * coeff));
          }
          res[y * W + x] = Math.trunc(acc); // astype(uint16)
        }

  // unpad, astype(float32), rescale_intensity to [0, 1]
  const out = new Float32Array(h * w);
  let lo = Infinity, hi = -Infinity;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const v = res[(y + ps[0]) * W + x + ps[1]];
      out[y * w + x] = v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  if (lo === hi) return out.map((v) => Math.min(Math.max(v, 0), 1));
  const d = f32(hi - lo);
  for (let i = 0; i < out.length; i++) out[i] = f32(f32(out[i] - lo) / d);
  return out;
}

/** Full pipeline: RGB uint8 (w x h) -> Float32Array(32*32) ready for the model. */
export function preprocess(rgb, w, h) {
  const small = resizeLinear(rgb, w, h);
  const gray = new Float32Array(SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const g = 0.299 * small[i * 3] + 0.587 * small[i * 3 + 1] + 0.114 * small[i * 3 + 2];
    gray[i] = f32(g / 255);
  }
  return equalizeAdapthist(gray, SIZE, SIZE);
}
