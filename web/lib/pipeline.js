/**
 * Orchestration: metadata encoding, ABCDE wording, escalation, change
 * detection. Mirrors src/pipeline.py and src/report/report_generator.py.
 */

import * as cv from './vision.js';
import { predict, runHead, softmax } from './model.js';

export const CFG = {
  imageSize: 224,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  quality: {
    minWidth: 224, minHeight: 224, blurMin: 60,
    brightnessMin: 40, brightnessMax: 215, contrastMin: 5
  },
  abcde: {
    asymmetry: 0.18, border: 1.55, colourVar: 0.16,
    colourClusters: 3, diameterMm: 6
  },
  sexVocab: ['female', 'male', 'unknown'],
  siteVocab: ['head_neck', 'trunk', 'upper_extremity', 'lower_extremity',
    'palms_soles', 'genital', 'unknown'],
  lowConfidence: 0.5,
  recommendations: {
    Low: 'Keep monitoring. Re-photograph this lesion in about 3 months and compare.',
    Moderate: 'Book a non-urgent appointment with a doctor or dermatologist to have this lesion checked.',
    High: 'See a dermatologist promptly. Do not wait for it to change further.'
  },
  disclaimer: 'This tool is a student research prototype. It is NOT a medical device, it has NOT been clinically validated, and it does NOT provide a diagnosis. Its output must never be used to rule out skin cancer. If you are worried about a skin lesion — or if it is changing, bleeding, itching, or does not heal — see a doctor or dermatologist regardless of what this tool says.'
};

const ORDER = ['Low', 'Moderate', 'High'];

/** Encode age/sex/site exactly as src/dataset.py does. */
export function encodeMetadata(age, sex, site) {
  const v = new Float32Array(2 + CFG.sexVocab.length + CFG.siteVocab.length);
  let missing = 0;
  let a = Number(age);
  if (!Number.isFinite(a) || a < 0) { a = 50; missing = 1; }
  v[0] = Math.min(1, Math.max(0, a / 100));
  v[1] = missing;
  const si = CFG.sexVocab.indexOf(sex);
  v[2 + (si >= 0 ? si : CFG.sexVocab.indexOf('unknown'))] = 1;
  const ti = CFG.siteVocab.indexOf(site);
  v[2 + CFG.sexVocab.length + (ti >= 0 ? ti : CFG.siteVocab.indexOf('unknown'))] = 1;
  return v;
}

/** Draw an image onto a canvas at a given size and return its pixels. */
export function rasterise(img, w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

/** Resize + normalise into the (1,3,224,224) tensor the trunk expects. */
export function preprocess(img) {
  const s = CFG.imageSize;
  const { data } = rasterise(img, s, s);
  const out = new Float32Array(3 * s * s);
  const plane = s * s;
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    out[i] = (data[p] / 255 - CFG.mean[0]) / CFG.std[0];
    out[plane + i] = (data[p + 1] / 255 - CFG.mean[1]) / CFG.std[1];
    out[2 * plane + i] = (data[p + 2] / 255 - CFG.mean[2]) / CFG.std[2];
  }
  return out;
}

/** Build the five ABCDE criteria with patient-facing wording. */
export function buildAbcde(rgba, mask, w, h, bbox, found, frameWidthMm, change) {
  const a = cv.asymmetry(mask, w, h, bbox);
  const b = cv.borderIrregularity(mask, w, h);
  const { variation, shades } = cv.colourVariation(rgba, mask, w, h);
  const dPx = cv.diameterPx(mask, w, h, bbox);
  const share = dPx / w;

  const f = [];
  const aFlag = a > CFG.abcde.asymmetry;
  f.push({
    letter: 'A', title: 'Asymmetry', score: a, flagged: aFlag, measurable: true,
    text: aFlag
      ? 'The two halves of this mark do not match each other well. Harmless moles are usually close to symmetrical.'
      : 'The mark looks fairly symmetrical — one half is a good match for the other, which is reassuring.'
  });

  const bFlag = b > CFG.abcde.border;
  f.push({
    letter: 'B', title: 'Border', score: b, flagged: bFlag, measurable: true,
    text: bFlag
      ? 'The edge is ragged and uneven rather than a smooth curve. Notched or scalloped edges are one of the things doctors look for.'
      : 'The edge is smooth and well defined, the way an ordinary mole’s edge usually looks.'
  });

  const cFlag = variation > CFG.abcde.colourVar || shades >= CFG.abcde.colourClusters;
  f.push({
    letter: 'C', title: 'Colour', score: variation, flagged: cFlag, measurable: true,
    text: cFlag
      ? `We count about ${shades} clearly different shades in this mark. Several colours mixed together in one lesion is a feature worth having checked.`
      : (shades > 1
        ? `There are around ${shades} shades here, but they are close to one another. Even colouring is typical of a harmless mole.`
        : 'The mark is a single even colour throughout, which is typical of a harmless mole.')
  });

  // A photo carries no scale, so diameter only counts when the user supplies
  // the frame width. Otherwise it would flag essentially every lesion.
  if (frameWidthMm > 0) {
    const mm = share * frameWidthMm;
    const dFlag = mm > CFG.abcde.diameterMm;
    f.push({
      letter: 'D', title: 'Diameter', score: mm, flagged: dFlag, measurable: true,
      text: `Using the ${frameWidthMm.toFixed(0)} mm frame width you gave us, this mark is about ${mm.toFixed(1)} mm across` +
        (dFlag ? `, wider than the ${CFG.abcde.diameterMm} mm guideline (roughly a pencil eraser).`
               : `, under the ${CFG.abcde.diameterMm} mm guideline.`)
    });
  } else {
    f.push({
      letter: 'D', title: 'Diameter', score: share, flagged: false, measurable: false,
      text: `The mark spans about ${Math.round(share * 100)}% of the photo’s width, but we cannot convert that to millimetres without knowing how close the camera was. Measure it with a ruler: anything wider than ${CFG.abcde.diameterMm} mm (about a pencil eraser) is worth showing to a doctor.`
    });
  }

  if (change) {
    f.push({
      letter: 'E', title: 'Evolving', score: change.score, flagged: change.changed, measurable: true,
      text: change.changed
        ? `Compared with the earlier photo you gave us, this mark has changed noticeably. ${change.reason}`
        : 'Compared with the earlier photo you gave us, this mark looks much the same. No significant change detected.'
    });
  } else {
    f.push({
      letter: 'E', title: 'Evolving', score: null, flagged: false, measurable: false,
      text: 'We cannot tell whether this mark is changing from a single photo. Add an earlier photo of the same lesion to have this checked — change over time is one of the most important warning signs.'
    });
  }

  const measurable = f.filter((x) => x.measurable).length;
  const flagged = f.filter((x) => x.flagged);
  const summary = !found
    ? 'The lesion outline could not be found reliably, so these ABCDE measurements are unreliable. Retake the photo with the lesion centred against plain skin.'
    : (flagged.length === 0
      ? `None of the ${measurable} ABCDE features we could measure looked concerning.`
      : `${flagged.length} of the ${measurable} ABCDE features we could measure looked concerning (${flagged.map((x) => x.letter).join(', ')}).`);

  return { features: f, flagCount: flagged.length, summary, segmentationConfident: found };
}

/** Cosine distance between two vectors. */
function cosineDistance(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  const d = Math.sqrt(na) * Math.sqrt(nb);
  return d < 1e-12 ? 0 : Math.min(2, Math.max(0, 1 - dot / d));
}

/** Compare two lesion photos using embedding distance plus shape descriptors. */
export function detectChange(prev, curr) {
  const signals = [];
  const appearance = cosineDistance(prev.embedding, curr.embedding);
  signals.push({ name: 'appearance', value: appearance, threshold: 0.15, weight: 0.40,
    changed: appearance > 0.15,
    text: appearance > 0.15 ? 'Its overall appearance has shifted noticeably.' : 'Its overall appearance is essentially unchanged.' });

  const dAsym = Math.abs(curr.asym - prev.asym);
  signals.push({ name: 'asymmetry', value: dAsym, threshold: 0.10, weight: 0.30,
    changed: dAsym > 0.10,
    text: dAsym > 0.10 ? 'It has become more lopsided than it was.' : 'Its symmetry is about the same.' });

  const dBorder = Math.abs(curr.border - prev.border);
  signals.push({ name: 'border', value: dBorder, threshold: 0.35, weight: 0.30,
    changed: dBorder > 0.35,
    text: dBorder > 0.35 ? 'Its edge has become more ragged.' : 'Its edge looks much as it did.' });

  const totalW = signals.reduce((s, x) => s + x.weight, 0);
  const score = signals.filter((x) => x.changed).reduce((s, x) => s + x.weight, 0) / totalW;
  const changed = score >= 0.35;
  return {
    changed, score, signals,
    reason: changed
      ? signals.filter((x) => x.changed).map((x) => x.text).join(' ')
      : 'Nothing we measure has shifted meaningfully between the two photos.'
  };
}

function escalate(level, steps = 1) {
  const i = ORDER.indexOf(level);
  return i < 0 ? level : ORDER[Math.min(i + steps, ORDER.length - 1)];
}

/**
 * Assemble the final report. Evidence may raise the risk level, never lower
 * it — the same rule as the Python implementation.
 */
export function buildReport({ probs, classes, abcde, quality, change, focusRatio, segFound }) {
  const top = probs.indexOf(Math.max(...probs));
  const modelLevel = classes[top];
  const confidence = probs[top];
  let level = modelLevel;
  const reasons = [];
  const notes = [];

  if (confidence < CFG.lowConfidence) {
    level = escalate(level);
    reasons.push(`the model was only ${Math.round(confidence * 100)}% confident, below the ${Math.round(CFG.lowConfidence * 100)}% mark we treat as reliable`);
  }
  if (change && change.changed) {
    level = escalate(level);
    reasons.push('the lesion has visibly changed since your earlier photo');
  }
  if (abcde.flagCount >= 3) {
    level = escalate(level);
    reasons.push(`${abcde.flagCount} ABCDE features were flagged`);
  }
  if (focusRatio !== null && focusRatio < 1.2) {
    notes.push(`Caution: the model’s attention was no more concentrated on the lesion than on the surrounding skin. It may be reacting to the background rather than the mark.`);
    if (focusRatio < 1.0) {
      level = escalate(level);
      reasons.push('the model’s attention fell on the lesion less than chance alone would predict');
    }
  }
  if (!segFound) {
    notes.push('We could not trace the lesion’s outline confidently, so the ABCDE measurements above are rough. A clearer, better-lit photo would help.');
  }
  if (quality && !quality.passed) {
    notes.push(`This photo did not pass our quality checks, so the result is less reliable than usual. ${quality.reason}`);
  }
  if (!change) {
    notes.push('You did not add an earlier photo, so we could not check whether this lesion is changing — one of the strongest warning signs there is.');
  }
  notes.push('This result came from a model trained on synthetic demonstration images. It shows the pipeline working; it is not a validated classifier.');

  return {
    level, modelLevel, confidence,
    probabilities: Object.fromEntries(classes.map((c, i) => [c, probs[i]])),
    recommendation: CFG.recommendations[level],
    escalationReasons: reasons,
    notes, abcde, quality, change
  };
}

/** Run the whole pipeline on one image element. */
export async function analyse(img, { age, sex, site, frameWidthMm, prevImg }) {
  const W = 320;
  const H = Math.max(1, Math.round((img.naturalHeight / img.naturalWidth) * W));
  const full = rasterise(img, W, H);

  const seg = cv.segment(full.data, W, H);
  const quality = cv.assessQuality(full.data, W, H, CFG.quality, seg.found);

  const meta = encodeMetadata(age, sex, site);
  const out = await predict(preprocess(img), meta);

  // How concentrated is the relevance map on the lesion, versus chance?
  const grid = out.grid;
  let inside = 0, total = 0, lesionPx = 0;
  for (let gy = 0; gy < grid; gy++) {
    for (let gx = 0; gx < grid; gx++) {
      const v = out.relevance[gy * grid + gx];
      total += v;
      const x = Math.min(W - 1, Math.floor((gx + 0.5) / grid * W));
      const y = Math.min(H - 1, Math.floor((gy + 0.5) / grid * H));
      if (seg.mask[y * W + x]) { inside += v; lesionPx++; }
    }
  }
  const areaFrac = lesionPx / (grid * grid);
  const focusRatio = total > 1e-6 && areaFrac > 0 ? (inside / total) / areaFrac : null;

  let change = null;
  if (prevImg) {
    const pW = 320;
    const pH = Math.max(1, Math.round((prevImg.naturalHeight / prevImg.naturalWidth) * pW));
    const pFull = rasterise(prevImg, pW, pH);
    const pSeg = cv.segment(pFull.data, pW, pH);
    const pOut = await predict(preprocess(prevImg), meta);
    change = detectChange(
      {
        embedding: pOut.relevance,
        asym: cv.asymmetry(pSeg.mask, pW, pH, pSeg.bbox),
        border: cv.borderIrregularity(pSeg.mask, pW, pH)
      },
      {
        embedding: out.relevance,
        asym: cv.asymmetry(seg.mask, W, H, seg.bbox),
        border: cv.borderIrregularity(seg.mask, W, H)
      }
    );
  }

  const abcde = buildAbcde(full.data, seg.mask, W, H, seg.bbox, seg.found,
    frameWidthMm, change);
  const report = buildReport({
    probs: Array.from(out.probs), classes: out.classes, abcde, quality,
    change, focusRatio, segFound: seg.found
  });

  return { report, relevance: out.relevance, grid, seg, width: W, height: H, focusRatio };
}
