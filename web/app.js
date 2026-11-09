/** UI wiring for the in-browser screening app. */

import { analyse, CFG } from './lib/pipeline.js';
import { loadModel, isLoaded } from './lib/model.js';

// Model files are served with an immutable cache header, so the URLs carry
// a content hash. Without it a browser keeps the previous model forever and
// silently mixes a new graph with a stale head.
const MODEL_URL = './model/panderm_vit_int8.onnx?v=a500fca1';
const HEAD_URL = './model/head.json?v=b548555a';

const $ = (id) => document.getElementById(id);
const state = { img: null, prevImg: null, ready: false };

// --- model loading ---------------------------------------------------------

async function ensureModel() {
  if (isLoaded()) return;
  const bar = $('loadBar');
  const wrap = $('loading');
  const label = $('loadLabel');
  wrap.hidden = false;
  try {
    await loadModel(MODEL_URL, HEAD_URL, (stage, pct) => {
      label.textContent = pct >= 1 ? stage : `${stage} — ${Math.round(pct * 100)}%`;
      bar.style.width = `${Math.round(pct * 100)}%`;
    });
    state.ready = true;
    wrap.hidden = true;
    $('status').innerHTML =
      '<span class="ok">PanDerm loaded — running entirely in your browser</span>';
  } catch (err) {
    wrap.hidden = true;
    $('status').innerHTML = `<span class="bad">Could not load the model: ${err.message}</span>`;
    throw err;
  }
}

// --- rendering -------------------------------------------------------------

function heatmapCanvas(relevance, grid, srcImg, w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.drawImage(srcImg, 0, 0, w, h);

  // Bilinear upsample of the grid, drawn as a translucent jet overlay.
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
      overlay.data[i + 3] = Math.round(150 * Math.min(1, Math.max(0, v)) + 40);
    }
  }
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  tmp.getContext('2d').putImageData(overlay, 0, 0);
  ctx.drawImage(tmp, 0, 0);
  return c;
}

function jet(v) {
  v = Math.min(1, Math.max(0, v));
  const r = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 3))));
  const g = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 2))));
  const b = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 1))));
  return [r, g, b];
}

function render(result) {
  const { report, relevance, grid, width, height, focusRatio } = result;
  const out = $('result');
  out.hidden = false;

  const cls = report.level.toLowerCase();
  const pct = (v) => `${Math.round(v * 100)}%`;

  // Lead with what the lesion appears to be, then what to do about it.
  let html = `
    <div class="verdict ${cls}">
      <div class="vlabel">Most likely</div>
      <div class="vlevel">${report.diagnosisName}</div>
      <p class="dxdesc">${report.diagnosisDescription}</p>
      <p class="vrec"><strong>${report.level} risk — ${report.recommendation}</strong></p>
    </div>`;

  // A six-way call at ~71% accuracy is not a single confident answer, so the
  // runners-up are shown rather than hidden behind one headline.
  html += '<h3>Other possibilities</h3><ul class="ranked">';
  for (const r of report.ranked.slice(0, 4)) {
    const lead = r.code === report.diagnosis ? ' lead' : '';
    html += `<li class="${lead}"><span class="bar" style="width:${Math.max(2, r.p * 100)}%"></span>
      <span class="nm">${r.name}</span><span class="pc">${pct(r.p)}</span></li>`;
  }
  html += '</ul>';

  if (report.escalationReasons.length) {
    html += `<p class="escalate">Advice raised to <strong>${report.level}</strong> because
      ${report.escalationReasons.join('; ')}.</p>`;
  }

  html += `<div class="panes">
      <figure><figcaption>Your photo</figcaption><div id="paneA"></div></figure>
      <figure><figcaption>Where the model looked</figcaption><div id="paneB"></div></figure>
    </div>`;

  html += `<h3>ABCDE check</h3><p class="muted">${report.abcde.summary}</p><ul class="abcde">`;
  for (const f of report.abcde.features) {
    const tag = !f.measurable ? '<em class="na">not measured</em>'
      : f.flagged ? '<em class="flag">worth attention</em>'
      : '<em class="fine">looks fine</em>';
    html += `<li><b>${f.letter}</b> <span class="t">${f.title}</span> ${tag}<p>${f.text}</p></li>`;
  }
  html += '</ul>';

  if (report.change) {
    html += `<p class="${report.change.changed ? 'warnbox' : 'okbox'}">
      <strong>${report.change.changed ? 'Changed since your earlier photo' : 'No significant change'}.</strong>
      ${report.change.reason}</p>`;
  }

  // Caveats are collapsed: present and findable, but not shouting over the result.
  if (report.notes.length) {
    html += '<details class="notes"><summary>Caveats for this result</summary><ul>' +
      report.notes.map((n) => `<li>${n}</li>`).join('') + '</ul></details>';
  }

  out.innerHTML = html;

  const a = document.createElement('canvas');
  a.width = width; a.height = height;
  a.getContext('2d').drawImage(state.img, 0, 0, width, height);
  $('paneA').appendChild(a);
  $('paneB').appendChild(heatmapCanvas(relevance, grid, state.img, width, height));
  if (focusRatio !== null) {
    const cap = document.createElement('p');
    cap.className = 'muted small';
    cap.textContent = `${focusRatio.toFixed(1)}× more focused on the lesion than chance.`;
    $('paneB').appendChild(cap);
  }
  out.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// --- events ----------------------------------------------------------------

function readFile(file, slot) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => { state[slot] = img; resolve(img); };
    img.src = URL.createObjectURL(file);
  });
}

$('photo').addEventListener('change', async (e) => {
  if (!e.target.files[0]) return;
  await readFile(e.target.files[0], 'img');
  $('photoName').textContent = e.target.files[0].name;
  $('run').disabled = false;
});

$('prev').addEventListener('change', async (e) => {
  if (!e.target.files[0]) return;
  await readFile(e.target.files[0], 'prevImg');
  $('prevName').textContent = e.target.files[0].name;
});

$('useScale').addEventListener('change', (e) => {
  $('scaleWrap').hidden = !e.target.checked;
});

$('run').addEventListener('click', async () => {
  if (!state.img) return;
  $('run').disabled = true;
  $('run').textContent = 'Analysing…';
  try {
    await ensureModel();
    const result = await analyse(state.img, {
      age: Number($('age').value),
      sex: $('sex').value,
      site: $('site').value,
      frameWidthMm: $('useScale').checked ? Number($('frameMm').value) : 0,
      prevImg: state.prevImg
    });
    render(result);
  } catch (err) {
    $('result').hidden = false;
    $('result').innerHTML = `<p class="bad">Analysis failed: ${err.message}</p>`;
    console.error(err);
  } finally {
    $('run').disabled = false;
    $('run').textContent = 'Analyse photo';
  }
});

// Samples bundled with the site so the app is usable with no photo to hand.
document.querySelectorAll('[data-sample]').forEach((btn) => {
  btn.addEventListener('click', async () => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    await new Promise((res, rej) => {
      img.onload = res; img.onerror = rej;
      img.src = btn.dataset.sample;
    });
    // Apply the sample's real patient metadata. Age and body site genuinely
    // move the prediction, so running a real ISIC image against arbitrary
    // form defaults would misrepresent the model.
    if (btn.dataset.age) $('age').value = btn.dataset.age;
    if (btn.dataset.sex) $('sex').value = btn.dataset.sex;
    if (btn.dataset.site) $('site').value = btn.dataset.site;
    state.img = img;
    state.prevImg = null;
    $('photoName').textContent = `${btn.textContent.trim()} (ISIC test image)`;
    $('prevName').textContent = '';
    $('run').disabled = false;
    $('run').click();
  });
});
