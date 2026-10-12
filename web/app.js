import { CLASSES } from './classes.js';
import { preprocess, resizeLinear } from './preprocess.js';

const $ = (id) => document.getElementById(id);
const MAX_SIDE = 2048; // guard against huge photos; resizing happens in JS
let model;
let pending;

async function loadModel() {
  try {
    model = await tf.loadLayersModel('model/model.json');
    tf.tidy(() => model.predict(tf.zeros([1, 32, 32, 1]))); // warm up
    $('status').textContent = 'Model ready · runs on your device';
    $('statusDot').classList.add('ready');
    if (pending) run(pending);
  } catch (err) {
    console.error(err);
    $('status').textContent = 'Could not load the model';
  }
}

function drawGray(canvas, values, scale = 1) {
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(32, 32);
  for (let i = 0; i < 1024; i++) {
    const v = Math.round(values[i] * scale);
    img.data.set([v, v, v, 255], i * 4);
  }
  ctx.putImageData(img, 0, 0);
}

function readPixels(image) {
  let { naturalWidth: w, naturalHeight: h } = image;
  const k = Math.min(1, MAX_SIDE / Math.max(w, h));
  w = Math.round(w * k); h = Math.round(h * k);
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(image, 0, 0, w, h);
  const rgba = ctx.getImageData(0, 0, w, h).data;
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    rgb[j] = rgba[i]; rgb[j + 1] = rgba[i + 1]; rgb[j + 2] = rgba[i + 2];
  }
  return { rgb, w, h };
}

function run(image) {
  if (!model) { pending = image; $('result').innerHTML = '<p class="muted">Loading model…</p>'; return; }
  const { rgb, w, h } = readPixels(image);
  const input = preprocess(rgb, w, h);

  // pipeline previews
  const small = resizeLinear(rgb, w, h);
  const ctx = $('cResized').getContext('2d');
  const img = ctx.createImageData(32, 32);
  const gray = new Float32Array(1024);
  for (let i = 0; i < 1024; i++) {
    img.data.set([small[i * 3], small[i * 3 + 1], small[i * 3 + 2], 255], i * 4);
    gray[i] = 0.299 * small[i * 3] + 0.587 * small[i * 3 + 1] + 0.114 * small[i * 3 + 2];
  }
  ctx.putImageData(img, 0, 0);
  drawGray($('cGray'), gray);
  drawGray($('cClahe'), input, 255);
  $('pipeline').hidden = false;

  const probs = tf.tidy(() => model.predict(tf.tensor4d(input, [1, 32, 32, 1])).dataSync());
  const ranked = Array.from(probs, (p, i) => [i, p]).sort((a, b) => b[1] - a[1]).slice(0, 5);
  const [best, bestP] = ranked[0];
  $('result').innerHTML = `
    <p class="conf">CLASS ${best} · ${(bestP * 100).toFixed(2)}% CONFIDENCE</p>
    <p class="top">${CLASSES[best]}</p>
    ${bestP < 0.6 ? '<p class="muted">Low confidence: try a tighter crop of just the sign.</p>' : ''}
    <ul class="bars">${ranked.map(([i, p]) => `
      <li><span>${CLASSES[i]}</span><span class="pct">${(p * 100).toFixed(1)}%</span>
      <span class="track"><span class="fill" style="width:${Math.max(p * 100, 0.5)}%"></span></span></li>`).join('')}
    </ul>`;
}

function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  const preview = $('preview');
  preview.onload = () => run(preview);
  preview.src = url;
  preview.alt = 'Your uploaded image';
  preview.hidden = false;
  $('dropText').hidden = true;
}

const drop = $('drop');
$('file').addEventListener('change', (e) => handleFile(e.target.files[0]));
drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('file').click(); } });
['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => handleFile(e.dataTransfer.files[0]));
window.addEventListener('paste', (e) => {
  const item = [...e.clipboardData.items].find((i) => i.type.startsWith('image/'));
  if (item) handleFile(item.getAsFile());
});

loadModel();
