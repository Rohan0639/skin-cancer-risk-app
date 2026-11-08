/**
 * Classical computer vision, ported from the Python pipeline to run in-browser.
 *
 * Covers the image-quality gate, lesion segmentation and the ABCDE
 * measurements. Everything here is plain typed-array work on canvas pixels --
 * no OpenCV, which would not fit in a browser payload.
 *
 * Difference from the Python original: segmentation has no GrabCut refinement
 * step (it is OpenCV-only). Flat-fielding, Otsu, component selection with a
 * border penalty, and hole filling are all faithful ports.
 */

// --- small helpers ---------------------------------------------------------

/** Luma (0-255) for an RGBA buffer. */
export function luma(rgba, w, h) {
  const out = new Float32Array(w * h);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
  }
  return out;
}

export function mean(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}

export function std(a) {
  const m = mean(a);
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - m;
    s += d * d;
  }
  return Math.sqrt(s / a.length);
}

/** Variance of the Laplacian -- the sharpness proxy used by the quality gate. */
export function laplacianVar(gray, w, h) {
  const out = [];
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      out.push(
        -4 * gray[i] + gray[i - 1] + gray[i + 1] + gray[i - w] + gray[i + w]
      );
    }
  }
  return std(Float32Array.from(out)) ** 2;
}

/**
 * Separable box blur, repeated to approximate a Gaussian.
 * Used for flat-fielding, where only the low-frequency envelope matters.
 */
export function boxBlur(src, w, h, radius, passes = 3) {
  let cur = Float32Array.from(src);
  let tmp = new Float32Array(w * h);
  for (let p = 0; p < passes; p++) {
    // horizontal
    for (let y = 0; y < h; y++) {
      let acc = 0;
      const row = y * w;
      for (let x = -radius; x <= radius; x++) {
        acc += cur[row + Math.min(w - 1, Math.max(0, x))];
      }
      for (let x = 0; x < w; x++) {
        tmp[row + x] = acc / (2 * radius + 1);
        const add = Math.min(w - 1, x + radius + 1);
        const sub = Math.max(0, x - radius);
        acc += cur[row + add] - cur[row + sub];
      }
    }
    // vertical
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let y = -radius; y <= radius; y++) {
        acc += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
      }
      for (let y = 0; y < h; y++) {
        cur[y * w + x] = acc / (2 * radius + 1);
        const add = Math.min(h - 1, y + radius + 1);
        const sub = Math.max(0, y - radius);
        acc += tmp[add * w + x] - tmp[sub * w + x];
      }
    }
  }
  return cur;
}

/** Otsu's threshold over a 0-255 float array. */
export function otsu(values) {
  const hist = new Float64Array(256);
  for (let i = 0; i < values.length; i++) {
    hist[Math.min(255, Math.max(0, Math.round(values[i])))]++;
  }
  const total = values.length;
  let sum = 0;
  for (let t = 0; t < 256; t++) sum += t * hist[t];

  let sumB = 0, wB = 0, best = 0, bestVar = -1;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > bestVar) {
      bestVar = between;
      best = t;
    }
  }
  return best;
}

/** sRGB -> CIELAB, operating on an RGBA buffer. Returns {L,a,b} planes. */
export function toLab(rgba, w, h) {
  const n = w * h;
  const L = new Float32Array(n), A = new Float32Array(n), B = new Float32Array(n);
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    let r = rgba[p] / 255, g = rgba[p + 1] / 255, b = rgba[p + 2] / 255;
    r = r > 0.04045 ? ((r + 0.055) / 1.055) ** 2.4 : r / 12.92;
    g = g > 0.04045 ? ((g + 0.055) / 1.055) ** 2.4 : g / 12.92;
    b = b > 0.04045 ? ((b + 0.055) / 1.055) ** 2.4 : b / 12.92;
    const X = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047;
    const Y = r * 0.2126 + g * 0.7152 + b * 0.0722;
    const Z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;
    const fx = f(X), fy = f(Y), fz = f(Z);
    L[i] = 116 * fy - 16;
    A[i] = 500 * (fx - fy);
    B[i] = 200 * (fy - fz);
  }
  return { L, A, B };
}

// --- quality gate ----------------------------------------------------------

/**
 * Run the configured quality checks. Mirrors src/quality/image_quality.py.
 * @returns {{passed:boolean, checks:Array, reason:string}}
 */
export function assessQuality(rgba, w, h, cfg, segFound) {
  const g = luma(rgba, w, h);
  const m = mean(g), s = std(g);
  const lap = laplacianVar(g, w, h);
  const checks = [];

  const resOk = w >= cfg.minWidth && h >= cfg.minHeight;
  checks.push({
    name: 'resolution', passed: resOk, value: Math.min(w, h),
    message: resOk ? 'Photo quality looks good.'
      : `The photo is too small (${w}x${h} pixels). Please use at least ${cfg.minWidth}x${cfg.minHeight} and do not crop it down.`
  });

  const blurOk = lap >= cfg.blurMin;
  checks.push({
    name: 'blur', passed: blurOk, value: lap,
    message: blurOk ? 'Photo quality looks good.'
      : 'The photo looks out of focus. Hold the camera steady, tap to focus on the lesion, and take it again.'
  });

  const brightOk = m >= cfg.brightnessMin && m <= cfg.brightnessMax;
  checks.push({
    name: 'brightness', passed: brightOk, value: m,
    message: brightOk ? 'Photo quality looks good.'
      : (m < cfg.brightnessMin
        ? 'The photo is too dark. Move somewhere brighter and take it again.'
        : 'The photo is over-exposed — flash or sunlight has washed out the detail. Try again without direct flash.')
  });

  const contrastOk = s >= cfg.contrastMin;
  checks.push({
    name: 'contrast', passed: contrastOk, value: s,
    message: contrastOk ? 'Photo quality looks good.'
      : 'The photo is almost completely blank — we cannot see any detail. Check the camera was pointed at the lesion.'
  });

  checks.push({
    name: 'framing', passed: !!segFound, value: segFound ? 1 : 0,
    message: segFound ? 'Photo quality looks good.'
      : 'We could not clearly pick out a lesion. Centre it in the frame against plain skin and try again.'
  });

  const failed = checks.filter((c) => !c.passed);
  return {
    passed: failed.length === 0,
    checks,
    reason: failed.length === 0
      ? 'Photo quality looks good.'
      : failed.map((c) => c.message).join(' ')
  };
}

// --- segmentation ----------------------------------------------------------

/**
 * Segment the dominant lesion.
 *
 * Flat-fields a darkness+blueness channel so a flash gradient cannot dominate
 * the threshold, applies Otsu, then picks the largest, most central component
 * that does not hug the frame edge, and fills its interior holes.
 *
 * @returns {{mask:Uint8Array, found:boolean, areaFraction:number, bbox:number[]}}
 */
export function segment(rgba, w, h) {
  const n = w * h;
  const { L, B } = toLab(rgba, w, h);

  // Lesions are darker and bluer than skin.
  const chan = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    chan[i] = 0.65 * (100 - L[i]) * 2.55 + 0.35 * (128 - B[i]);
  }

  // Flat-field: divide out the low-frequency illumination envelope. The radius
  // must be much larger than any lesion or it suppresses the lesion centre.
  const radius = Math.max(8, Math.round(Math.max(w, h) * 0.25));
  const bg = boxBlur(chan, w, h, radius, 3);
  let lo = Infinity, hi = -Infinity;
  const flat = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    flat[i] = chan[i] / Math.max(1e-3, Math.abs(bg[i]) + 1);
    if (flat[i] < lo) lo = flat[i];
    if (flat[i] > hi) hi = flat[i];
  }
  const span = Math.max(1e-6, hi - lo);
  for (let i = 0; i < n; i++) flat[i] = ((flat[i] - lo) / span) * 255;

  const blurred = boxBlur(flat, w, h, 2, 1);
  const t = otsu(blurred);
  const bin = new Uint8Array(n);
  for (let i = 0; i < n; i++) bin[i] = blurred[i] > t ? 1 : 0;

  // Connected components (4-way BFS), scored on size, centrality and how much
  // of the frame border they occupy.
  const labels = new Int32Array(n).fill(-1);
  const cx0 = w / 2, cy0 = h / 2;
  const maxDist = Math.hypot(cx0, cy0);
  let best = null, lab = 0;
  const queue = new Int32Array(n);

  for (let start = 0; start < n; start++) {
    if (bin[start] !== 1 || labels[start] !== -1) continue;
    let qh = 0, qt = 0;
    queue[qt++] = start;
    labels[start] = lab;
    let area = 0, sx = 0, sy = 0, border = 0;
    let minX = w, maxX = 0, minY = h, maxY = 0;

    while (qh < qt) {
      const i = queue[qh++];
      const x = i % w, y = (i / w) | 0;
      area++; sx += x; sy += y;
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) border++;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;

      if (x > 0 && bin[i - 1] === 1 && labels[i - 1] === -1) { labels[i - 1] = lab; queue[qt++] = i - 1; }
      if (x < w - 1 && bin[i + 1] === 1 && labels[i + 1] === -1) { labels[i + 1] = lab; queue[qt++] = i + 1; }
      if (y > 0 && bin[i - w] === 1 && labels[i - w] === -1) { labels[i - w] = lab; queue[qt++] = i - w; }
      if (y < h - 1 && bin[i + w] === 1 && labels[i + w] === -1) { labels[i + w] = lab; queue[qt++] = i + w; }
    }

    const frac = area / n;
    const borderFrac = border / (2 * (w + h));
    if (frac >= 0.004 && frac <= 0.92 && borderFrac <= 0.35) {
      const centred = 1 - Math.hypot(sx / area - cx0, sy / area - cy0) / maxDist;
      const score = frac + 1.2 * centred - 1.5 * borderFrac;
      if (!best || score > best.score) {
        best = { lab, score, area, bbox: [minY, minX, maxY + 1, maxX + 1] };
      }
    }
    lab++;
  }

  const mask = new Uint8Array(n);
  if (!best) {
    // Fallback: a centred ellipse, flagged as not found.
    const rx = w * 0.3, ry = h * 0.3;
    let area = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const dx = (x - cx0) / rx, dy = (y - cy0) / ry;
        if (dx * dx + dy * dy <= 1) { mask[y * w + x] = 1; area++; }
      }
    }
    return { mask, found: false, areaFraction: area / n, bbox: [0, 0, h, w] };
  }

  for (let i = 0; i < n; i++) if (labels[i] === best.lab) mask[i] = 1;
  fillHoles(mask, w, h);

  let area = 0;
  for (let i = 0; i < n; i++) area += mask[i];
  return { mask, found: true, areaFraction: area / n, bbox: best.bbox };
}

/**
 * Fill interior holes by flood-filling background from the border and
 * inverting. Holes would corrupt area, asymmetry and especially perimeter.
 */
export function fillHoles(mask, w, h) {
  const n = w * h;
  const outside = new Uint8Array(n);
  const queue = new Int32Array(n);
  let qh = 0, qt = 0;
  const push = (i) => {
    if (!outside[i] && mask[i] === 0) { outside[i] = 1; queue[qt++] = i; }
  };
  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  while (qh < qt) {
    const i = queue[qh++];
    const x = i % w, y = (i / w) | 0;
    if (x > 0) push(i - 1);
    if (x < w - 1) push(i + 1);
    if (y > 0) push(i - w);
    if (y < h - 1) push(i + w);
  }
  for (let i = 0; i < n; i++) if (mask[i] === 0 && !outside[i]) mask[i] = 1;
}

// --- ABCDE -----------------------------------------------------------------

/** Fraction of lesion area that fails to overlap its mirror, per axis. */
export function asymmetry(mask, w, h, bbox) {
  const [y0, x0, y1, x1] = bbox;
  const bw = x1 - x0, bh = y1 - y0;
  if (bw < 3 || bh < 3) return 0;
  let area = 0, mmH = 0, mmV = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const v = mask[y * w + x];
      const hv = mask[y * w + (x1 - 1 - (x - x0))];
      const vv = mask[(y1 - 1 - (y - y0)) * w + x];
      area += v;
      if (v !== hv) mmH++;
      if (v !== vv) mmV++;
    }
  }
  if (area === 0) return 0;
  return Math.min(1, (mmH / 2 + mmV / 2) / (2 * area));
}

/** perimeter^2 / (4*pi*area) -- 1.0 for a circle, higher when ragged. */
export function borderIrregularity(mask, w, h) {
  let area = 0, perim = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      area++;
      if (x === 0 || x === w - 1 || y === 0 || y === h - 1 ||
          !mask[i - 1] || !mask[i + 1] || !mask[i - w] || !mask[i + w]) {
        perim++;
      }
    }
  }
  if (area < 10 || perim === 0) return 1;
  // Pixel-staircase perimeters over-count by roughly pi/4 versus a smooth curve.
  const p = perim * (Math.PI / 4);
  return Math.max(1, (p * p) / (4 * Math.PI * area));
}

/** Colour spread and distinct-shade count inside the lesion, in CIELAB. */
export function colourVariation(rgba, mask, w, h) {
  const { L, A, B } = toLab(rgba, w, h);
  const px = [];
  for (let i = 0; i < w * h; i++) {
    if (mask[i]) px.push([L[i], A[i], B[i]]);
  }
  if (px.length < 20) return { variation: 0, shades: 1 };

  const sample = px.length > 3000
    ? px.filter((_, i) => i % Math.ceil(px.length / 3000) === 0)
    : px;

  const dims = [0, 1, 2].map((d) => std(Float32Array.from(sample.map((p) => p[d]))));
  const variation = Math.min(1, (dims[0] + dims[1] + dims[2]) / 3 / 40);

  // k-means for k = 2..5, keeping clusters that are both well-populated and
  // perceptually distinct (>18 units apart in LAB).
  let shades = 1;
  for (let k = 2; k <= 5; k++) {
    const cent = [];
    for (let j = 0; j < k; j++) cent.push(sample[Math.floor((j + 0.5) * sample.length / k)].slice());
    let assign = new Int32Array(sample.length);
    for (let it = 0; it < 8; it++) {
      for (let i = 0; i < sample.length; i++) {
        let bi = 0, bd = Infinity;
        for (let j = 0; j < k; j++) {
          const d = (sample[i][0] - cent[j][0]) ** 2 + (sample[i][1] - cent[j][1]) ** 2 + (sample[i][2] - cent[j][2]) ** 2;
          if (d < bd) { bd = d; bi = j; }
        }
        assign[i] = bi;
      }
      const sums = Array.from({ length: k }, () => [0, 0, 0, 0]);
      for (let i = 0; i < sample.length; i++) {
        const a = assign[i];
        sums[a][0] += sample[i][0]; sums[a][1] += sample[i][1];
        sums[a][2] += sample[i][2]; sums[a][3]++;
      }
      for (let j = 0; j < k; j++) {
        if (sums[j][3] > 0) cent[j] = [sums[j][0] / sums[j][3], sums[j][1] / sums[j][3], sums[j][2] / sums[j][3]];
      }
    }
    const counts = new Array(k).fill(0);
    for (let i = 0; i < sample.length; i++) counts[assign[i]]++;
    const kept = cent.filter((_, j) => counts[j] / sample.length >= 0.08);
    if (kept.length >= 2) {
      let far = 0;
      for (let a = 0; a < kept.length; a++) {
        for (let b = a + 1; b < kept.length; b++) {
          far = Math.max(far, Math.hypot(kept[a][0] - kept[b][0], kept[a][1] - kept[b][1], kept[a][2] - kept[b][2]));
        }
      }
      if (far >= 18) shades = Math.max(shades, kept.length);
    }
  }
  return { variation, shades };
}

/** Widest lesion span in pixels, from the bounding box diagonal extent. */
export function diameterPx(mask, w, h, bbox) {
  const [y0, x0, y1, x1] = bbox;
  return Math.max(x1 - x0, y1 - y0);
}
