/** UI for the lesion check app: pick or capture a photo, analyse, show result. */

import { analyse } from './lib/pipeline.js';
import { loadModel, isLoaded } from './lib/model.js';

// Model files carry a content hash: they are served immutable, so without one
// a browser keeps the previous model forever and silently pairs a stale head
// with a new graph.
const MODEL_URL = './model/panderm_vit_int8.onnx?v=b8a38550';
const HEAD_URL = './model/head.json?v=63f065b4';

const $ = (id) => document.getElementById(id);
const state = { img: null, stream: null };

// --- photo input -----------------------------------------------------------

function showImage(src) {
  stopCamera();
  const img = new Image();
  img.onload = () => {
    state.img = img;
    $('preview').src = src;
    $('preview').hidden = false;
    $('dropText').hidden = true;
    $('run').disabled = false;
  };
  img.src = src;
}

function readFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => showImage(reader.result);
  reader.readAsDataURL(file);
}

$('file').addEventListener('change', (e) => readFile(e.target.files[0]));
$('btnFile').addEventListener('click', () => $('file').click());
$('drop').addEventListener('click', () => {
  if (!state.stream) $('file').click();
});

$('drop').addEventListener('dragover', (e) => { e.preventDefault(); });
$('drop').addEventListener('drop', (e) => {
  e.preventDefault();
  readFile(e.dataTransfer.files[0]);
});

// --- camera ----------------------------------------------------------------

async function startCamera() {
  try {
    // Rear camera where there is one; falls back to whatever is available.
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } },
      audio: false
    });
  } catch (err) {
    $('dropText').hidden = false;
    $('dropText').textContent =
      'Camera unavailable — choose a photo instead.';
    return;
  }
  const v = $('cam');
  v.srcObject = state.stream;
  await v.play();
  v.hidden = false;
  $('preview').hidden = true;
  $('dropText').hidden = true;
  $('pickRow').hidden = true;
  $('camRow').hidden = false;
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
  $('cam').hidden = true;
  $('pickRow').hidden = false;
  $('camRow').hidden = true;
}

$('btnCam').addEventListener('click', startCamera);
$('btnCancel').addEventListener('click', () => {
  stopCamera();
  if (!state.img) $('dropText').hidden = false;
});
$('btnShot').addEventListener('click', () => {
  const v = $('cam');
  const c = document.createElement('canvas');
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0);
  showImage(c.toDataURL('image/jpeg', 0.92));
});

// --- model -----------------------------------------------------------------

async function ensureModel() {
  if (isLoaded()) return;
  $('load').hidden = false;
  await loadModel(MODEL_URL, HEAD_URL, (stage, pct) => {
    $('loadText').textContent = pct >= 1 ? stage : `${stage} — ${Math.round(pct * 100)}%`;
    $('bar').style.width = `${Math.round(pct * 100)}%`;
  });
  $('load').hidden = true;
}

// --- heatmap ---------------------------------------------------------------

function jet(v) {
  v = Math.min(1, Math.max(0, v));
  return [
    Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 3)))),
    Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 2)))),
    Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 1))))
  ];
}

function drawPhoto(img, w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  c.getContext('2d').drawImage(img, 0, 0, w, h);
  return c;
}

function drawHeatmap(img, relevance, grid, w, h) {
  const c = drawPhoto(img, w, h);
  const ctx = c.getContext('2d');
  const overlay = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    const gy = (y / h) * grid - 0.5;
    const y0 = Math.max(0, Math.min(grid - 1, Math.floor(gy)));
    const y1 = Math.max(0, Math.min(grid - 1, y0 + 1));
    const fy = Math.max(0, Math.min(1, gy - y0));
    for (let x = 0; x < w; x++) {
      const gx = (x / w) * grid - 0.5;
      const x0 = Math.max(0, Math.min(grid - 1, Math.floor(gx)));
      const x1 = Math.max(0, Math.min(grid - 1, x0 + 1));
      const fx = Math.max(0, Math.min(1, gx - x0));
      const v =
        relevance[y0 * grid + x0] * (1 - fx) * (1 - fy) +
        relevance[y0 * grid + x1] * fx * (1 - fy) +
        relevance[y1 * grid + x0] * (1 - fx) * fy +
        relevance[y1 * grid + x1] * fx * fy;
      const [r, g, b] = jet(v);
      const i = (y * w + x) * 4;
      overlay.data[i] = r; overlay.data[i + 1] = g; overlay.data[i + 2] = b;
      overlay.data[i + 3] = Math.round(140 * Math.min(1, Math.max(0, v)) + 25);
    }
  }
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  tmp.getContext('2d').putImageData(overlay, 0, 0);
  ctx.drawImage(tmp, 0, 0);
  return c;
}

// --- result ----------------------------------------------------------------

function render(result) {
  const { report, relevance, grid, width, height } = result;
  const card = $('resultCard');
  const tier = report.level.toLowerCase();

  // The model picks from six lesion types and cannot say "something else", so
  // anything outside them is forced into one of the six. Treat a weak top
  // probability, or an escalation, as "do not name a diagnosis confidently" --
  // and never pair a reassuring description with raised advice.
  const escalated = report.level !== report.modelLevel;
  const uncertain = report.diagnosisConfidence < 0.5 || escalated;

  const poor = report.quality && !report.quality.passed
    ? `<p class="warn">Photo quality is poor — ${report.quality.reason}
       Retake it for a more reliable result.</p>` : '';

  let head;
  if (uncertain) {
    const alts = report.ranked.slice(0, 3)
      .map((r) => `<li><span>${r.name}</span><b>${Math.round(r.p * 100)}%</b></li>`)
      .join('');
    head = `
      <div class="verdict ${tier}">
        <div class="vk">Not confident</div>
        <div class="vn">${report.level} risk</div>
        <p class="vd">The model could not identify this with confidence. Its
          closest matches are below — treat them as guesses, not a diagnosis.</p>
        <ul class="alts">${alts}</ul>
        <p class="vr">${report.recommendation}</p>
      </div>`;
  } else {
    head = `
      <div class="verdict ${tier}">
        <div class="vk">Most likely</div>
        <div class="vn">${report.diagnosisName}</div>
        <p class="vd">${report.diagnosisDescription}</p>
        <p class="vr">${report.level} risk — ${report.recommendation}</p>
      </div>`;
  }

  card.innerHTML = `
    ${poor}
    ${head}
    <div class="pair">
      <figure><figcaption>Photo</figcaption><div id="pA"></div></figure>
      <figure><figcaption>What the model looked at</figcaption><div id="pB"></div></figure>
    </div>
    <div class="expl">
      <h3>Why</h3>
      <ul id="why"></ul>
    </div>
    <p class="scope">Recognises six common lesion types only: mole, seborrhoeic
      keratosis, actinic keratosis, basal cell carcinoma, squamous cell carcinoma
      and melanoma. Anything else — including rarer cancers — will be given one
      of these labels in error.</p>
    <div class="row" style="margin-top:1rem">
      <button id="again">Check another photo</button>
    </div>`;

  $('pA').appendChild(drawPhoto(state.img, width, height));
  $('pB').appendChild(drawHeatmap(state.img, relevance, grid, width, height));

  const ul = $('why');
  for (const f of report.abcde.features) {
    const cls = !f.measurable ? 'na' : (f.flagged ? 'flag' : 'ok');
    const li = document.createElement('li');
    li.innerHTML = `<span class="dot ${cls}"></span><span>${f.text}</span>`;
    ul.appendChild(li);
  }

  $('again').addEventListener('click', () => {
    card.hidden = true;
    $('inputCard').hidden = false;
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  $('inputCard').hidden = true;
  card.hidden = false;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// --- run -------------------------------------------------------------------

$('run').addEventListener('click', async () => {
  if (!state.img) return;
  const btn = $('run');
  btn.disabled = true;
  btn.textContent = 'Analysing…';
  try {
    await ensureModel();
    const result = await analyse(state.img, {
      age: Number($('age').value),
      sex: $('sex').value,
      site: $('site').value,
      frameWidthMm: 0,
      prevImg: null
    });
    render(result);
  } catch (err) {
    $('load').hidden = false;
    $('loadText').textContent = `Could not analyse: ${err.message}`;
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Analyse';
  }
});
