/**
 * In-browser inference: ONNX ViT trunk + the fusion head reimplemented in JS.
 *
 * The head lives here rather than in the ONNX graph because it has to run
 * twice -- once on the CLS token for the prediction, once per patch token for
 * the relevance map -- and two copies inside one graph broke the int8
 * quantiser. It is small, so replaying it here is cheap.
 *
 * This file's maths is verified against PyTorch by
 * `scripts/verify_web_model.py`, which runs the identical operations in NumPy
 * and agreed on 30/30 real images with 0.00000 probability drift.
 */

export const CLASSES = ['Low', 'Moderate', 'High'];

/** Risk-class order must match the exported head. */
let session = null;
let head = null;

/**
 * Load the ONNX trunk and the head weights.
 * @param {string} modelUrl  URL of the .onnx file
 * @param {string} headUrl   URL of head.json
 * @param {(stage:string, pct:number)=>void} onProgress
 */
export async function loadModel(modelUrl, headUrl, onProgress = () => {}) {
  // Run the WASM backend inside a Web Worker. A ViT-B forward pass takes
  // seconds on WASM; on the main thread that freezes the whole tab, including
  // the progress bar meant to reassure the user it is still working.
  ort.env.wasm.wasmPaths = new URL('vendor/ort/', location.href).href;
  ort.env.wasm.proxy = true;
  ort.env.wasm.simd = true;
  // Multithreading needs cross-origin isolation (COOP/COEP). Those headers are
  // set in vercel.json; fall back to single-threaded if they are absent, which
  // is what happens on a plain local file server.
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 2)
    : 1;

  onProgress('Downloading model', 0);

  // Stream the model so the UI can show real progress -- it is ~83 MB.
  const res = await fetch(modelUrl);
  if (!res.ok) throw new Error(`Model download failed: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) onProgress('Downloading model', received / total);
  }
  const bytes = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { bytes.set(c, off); off += c.length; }

  onProgress('Starting inference engine', 1);
  session = await ort.InferenceSession.create(bytes.buffer, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all'
  });

  head = await (await fetch(headUrl)).json();
  onProgress('Ready', 1);
  return { classes: head.classes || CLASSES };
}

export function isLoaded() {
  return session !== null && head !== null;
}

export function metadataDim() {
  return head ? head.metadata_dim : 12;
}

// --- head ops (mirror of scripts/verify_web_model.py) ----------------------

function linear(x, rows, spec) {
  const { w, b } = spec;
  const inD = spec.in, outD = spec.out;
  const y = new Float32Array(rows * outD);
  for (let r = 0; r < rows; r++) {
    for (let o = 0; o < outD; o++) {
      let acc = b[o];
      const wo = o * inD, xr = r * inD;
      for (let i = 0; i < inD; i++) acc += x[xr + i] * w[wo + i];
      y[r * outD + o] = acc;
    }
  }
  return y;
}

function relu(x) {
  for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0;
  return x;
}

function layerNorm(x, rows, dim, spec) {
  const { g, b, eps } = spec;
  for (let r = 0; r < rows; r++) {
    const o = r * dim;
    let m = 0;
    for (let i = 0; i < dim; i++) m += x[o + i];
    m /= dim;
    let v = 0;
    for (let i = 0; i < dim; i++) { const d = x[o + i] - m; v += d * d; }
    v /= dim;
    const inv = 1 / Math.sqrt(v + eps);
    for (let i = 0; i < dim; i++) x[o + i] = (x[o + i] - m) * inv * g[i] + b[i];
  }
  return x;
}

/**
 * Apply the fusion head.
 * @param {Float32Array} feats  (rows x 768) features
 * @param {number} rows
 * @param {Float32Array} meta   (metadata_dim) encoded metadata, shared by all rows
 * @returns {Float32Array} (rows x nClasses) logits
 */
export function runHead(feats, rows, meta) {
  const ip = head.image_proj;
  let x = linear(feats, rows, ip.linear);
  relu(x);
  layerNorm(x, rows, ip.linear.out, ip.norm);
  let dim = ip.linear.out;

  if (head.use_metadata && head.metadata_encoder) {
    const me = head.metadata_encoder;
    let m = linear(meta, 1, me.linear);
    relu(m);
    layerNorm(m, 1, me.linear.out, me.norm);
    const mDim = me.linear.out;

    const merged = new Float32Array(rows * (dim + mDim));
    for (let r = 0; r < rows; r++) {
      merged.set(x.subarray(r * dim, r * dim + dim), r * (dim + mDim));
      merged.set(m, r * (dim + mDim) + dim);
    }
    x = merged;
    dim += mDim;
  }

  for (const block of head.mlp) {
    x = linear(x, rows, block.linear);
    dim = block.linear.out;
    if (block.relu) relu(x);
    if (block.bn) {
      const { scale, shift } = block.bn;
      for (let r = 0; r < rows; r++) {
        const o = r * dim;
        for (let i = 0; i < dim; i++) x[o + i] = x[o + i] * scale[i] + shift[i];
      }
    }
  }
  return x;
}

export function softmax(v) {
  let max = -Infinity;
  for (let i = 0; i < v.length; i++) if (v[i] > max) max = v[i];
  let sum = 0;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) { out[i] = Math.exp(v[i] - max); sum += out[i]; }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

/**
 * Run the full model on a preprocessed image tensor.
 *
 * @param {Float32Array} input  (1x3x224x224) normalised CHW tensor
 * @param {Float32Array} meta   encoded metadata
 * @returns {{probs:Float32Array, classes:string[], relevance:Float32Array, grid:number}}
 *   `relevance` is a 14x14 map of how strongly each patch pushes the predicted
 *   class, obtained by applying the head to every patch token. This is the
 *   gradient-free stand-in for Grad-CAM.
 */
export async function predict(input, meta) {
  const tensor = new ort.Tensor('float32', input, [1, 3, 224, 224]);
  const out = await session.run({ image: tensor });
  const tokens = out.tokens.data;             // (1, 197, 768)
  const nTok = out.tokens.dims[1];
  const dim = out.tokens.dims[2];

  const cls = tokens.slice(0, dim);
  const probs = softmax(runHead(Float32Array.from(cls), 1, meta));
  const top = probs.indexOf(Math.max(...probs));

  const nPatch = nTok - 1;
  const patches = Float32Array.from(tokens.subarray(dim, dim + nPatch * dim));
  const patchLogits = runHead(patches, nPatch, meta);

  const nCls = probs.length;
  const relevance = new Float32Array(nPatch);
  for (let p = 0; p < nPatch; p++) {
    // Margin of the predicted class over the mean of the others: how much
    // this patch argues *for* the reported class specifically.
    let other = 0;
    for (let c = 0; c < nCls; c++) if (c !== top) other += patchLogits[p * nCls + c];
    relevance[p] = patchLogits[p * nCls + top] - other / (nCls - 1);
  }

  // Normalise on percentiles rather than min/max. Straight min-max stretches
  // the background's small fluctuations across the whole colour ramp, which
  // renders as rainbow noise and buries the actual signal. Clipping at the
  // 60th/99th percentile pins typical background to zero and keeps the ramp
  // for patches that genuinely argue for the class. The gamma then further
  // suppresses the low end.
  const sorted = Float32Array.from(relevance).sort();
  const pick = (q) => sorted[Math.min(sorted.length - 1,
    Math.max(0, Math.round(q * (sorted.length - 1))))];
  const lo = pick(0.60);
  const hi = pick(0.99);
  const span = Math.max(1e-6, hi - lo);
  for (let i = 0; i < relevance.length; i++) {
    const v = Math.min(1, Math.max(0, (relevance[i] - lo) / span));
    relevance[i] = Math.pow(v, 1.4);
  }

  return {
    probs,
    classes: head.classes || CLASSES,
    relevance,
    grid: Math.round(Math.sqrt(nPatch))
  };
}
