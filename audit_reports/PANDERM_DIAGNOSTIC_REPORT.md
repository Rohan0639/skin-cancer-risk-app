# PanDerm Skin Lesion Classifier — Technical Audit

**Date:** 2026-11-21
**Auditor:** Claude Opus 5 (Claude Code)
**Subject:** `skin-cancer-risk-app` — PanDerm-based skin lesion classifier, deployed in-browser
**Model audited:** frozen `PanDerm_Base_LP` + `FusionHead` (`checkpoints/risk_head.pt`) — the configuration actually in production

---

## Scope note, read first

Two code changes were made **earlier the same day, before this audit was requested**, both in response to
live failures the user reported. They are in the audited build and are marked ⚠️ where relevant:

1. **Risk tier-mass floor** (`web/lib/pipeline.js`) — a graver tier holding ≥25% of probability mass now
   raises the advice. Before this, a melanoma at 38% was reported as "Low risk, re-photograph in 3 months"
   because a mole at 49% narrowly led.
2. **Metadata defaults** (`web/index.html`, `web/app.js`) — the form pre-filled age 45 and site "trunk" as
   if they were facts. Fields now start empty and a blank age sets the missing-data flag.

No files were modified during the audit itself. All findings below describe the current build.

---

## 1. Project overview

A browser-only skin lesion screening aid. A user supplies a photo and optional patient metadata; the app
returns a lesion-type prediction, a three-tier risk rating, an attention heatmap and an ABCDE explanation.
Inference runs entirely client-side via ONNX Runtime Web — the photo is never uploaded.

| Property | Value |
|---|---|
| Classes | 6 — NEV, SEK, ACK, BCC, SCC, MEL |
| Risk tiers | Low (NEV, SEK), Moderate (ACK), High (BCC, SCC, MEL) |
| Training data | 3,104 ISIC clinical photographs |
| Deployment | Static site on Vercel, WASM inference in a Web Worker |
| Backbone licence | CC-BY-NC (non-commercial) |

---

## 2. Architecture and dependency map

### Prediction pipeline, traced to source

| Stage | File | Function / symbol |
|---|---|---|
| Image upload / camera | `web/app.js` | `readFile`, `startCamera`, `btnShot` handler |
| Quality gate | `web/lib/pipeline.js` | `analyse` → quality checks |
| Preprocessing | `web/lib/pipeline.js` | `drawTo`, tensor build (224×224, ImageNet norm) |
| PanDerm forward | `web/lib/model.js` | `predict` → ONNX session run |
| ONNX graph | `scripts/export_onnx.py` | `TrunkWrapper.forward` |
| Feature extraction | `src/backbone/panderm_wrapper.py` | `PanDermBackbone.forward` (line 241) |
| Classification head | `src/model/fusion.py` | `FusionHead.forward` (line 83) |
| Head weights in browser | `web/model/head.json` | emitted by `dump_head` |
| Risk tier + escalation | `web/lib/pipeline.js` | `buildReport` (line ~207) |
| Abstention + render | `web/app.js` | `render` |
| Attention heatmap | `scripts/export_onnx.py` | rollout inside `TrunkWrapper` |
| Heatmap normalisation | `web/lib/model.js` | percentile p50→p99, gamma 1.2 |
| Training | `src/model/train.py` | `load_or_build_feature_cache`, `macro_f1` |
| Dataset / splits | `src/dataset.py` | `group_aware_split`, `encode_metadata` |
| Evaluation | `src/evaluate.py`, `scripts/audit_evaluate.py` | — |

### Flow

```
Camera / upload  →  quality gate  →  resize 224² + ImageNet norm
                                          ↓
                      PanDerm ViT-B/16 (frozen, ONNX int8, WASM worker)
                                          ↓
                    tokens [1,197,768]          rollout [1,196]
                            ↓                        ↓
              768-d embedding + 12-d metadata    heatmap
                            ↓
                 FusionHead (JS, head.json) → 6 logits → softmax
                            ↓
              diagnosis → risk tier (lookup) → escalation rules → abstention
                            ↓
                    rendered result + heatmap + ABCDE
```

There is **no backend**. The "API communication" layer in the original brief does not exist in this build:
the model, head weights and all logic ship to the browser as static assets.

---

## 3. PanDerm integration audit — **VERIFIED CORRECT**

This was checked by weight comparison, not by reading configuration.

| Check | Result |
|---|---|
| Configured backend | `backbone.backend: panderm`, set to fail loudly (no silent fallback) |
| Checkpoint on disk | `panderm_bb_data6_checkpoint-499.pth`, 343 MB — the authors' release |
| Architecture | `PanDerm_Base_LP`, ViT-B/16, 86M params, 768-d embedding |
| State-dict load | 2 missing keys, both the discarded pretraining head — expected |
| **Deployed ONNX vs checkpoint** | **`patch_embed.proj.weight` max abs diff = `0.000e+00` — bit-identical** |
| Deployed file vs local export | SHA-256 identical (`b8a38550…`) |
| Eval mode | `.eval()` set in `load_risk_classifier` |
| Device | `cuda:0` (RTX 4050) for Python; WASM int8 in browser |
| Output dims | tokens `[1,197,768]`, pooled `[1,768]`, logits `[1,6]` |
| Softmax | applied once, at the end, in JS and PyTorch alike |
| Export parity | trunk max diff `4.63e-05`, rollout `2.61e-08` |

**Configuration in use:** frozen PanDerm embeddings + a separately trained classification head
(linear-probe setting, which is what `_LP` denotes). The backbone has never seen this dataset.

**Train/inference consistency:** the same `build_transform(train=False)` and the same `FusionHead`
definition are used in training, evaluation and export. The JS head is verified against PyTorch by
`scripts/verify_web_model.py`. **No discrepancy found.**

**PanDerm does not natively support these 6 classes** — correct, and the implementation does not assume it
does. The class set is defined in `configs/default.yaml` and learned entirely by the head.

### Is PanDerm earning its place?

Measured, same head, same split, matched parameter budget:

| Setting | PanDerm | ConvNeXt-Base (ImageNet) | Gap |
|---|---|---|---|
| Frozen (5 seeds) | 69.3% ±1.2 | 69.2% ±1.3 | +0.1 — noise |
| Fine-tuned (28.4M vs 29.6M trainable) | **71.2%** | 66.7% | **+4.5** |

Frozen, PanDerm is indistinguishable from a generic ImageNet backbone on this data. Fine-tuned, it is
clearly better — fine-tuning *helped* PanDerm and *hurt* ConvNeXt. The justified claim is therefore not
"PanDerm's features are better" but "PanDerm can be fine-tuned on a small dermatology set without
collapsing, and a generic backbone cannot."

---

## 4. Dataset and data integrity

| Property | Value |
|---|---|
| Source | ISIC Archive, clinical close-up images |
| Total | 3,104 |
| Split | 2,170 train / 469 val / **465 test**, grouped by `lesion_id` |
| Test distribution | NEV 95, SEK 83, ACK 93, BCC 87, SCC 32, MEL 75 |

### Leakage checks — **CLEAN**

| Check | Result |
|---|---|
| Lesion IDs shared train↔test | **0** |
| Identical image paths shared | **0** |
| Test images with train cosine > 0.99 | **2** (0.4%) |
| Mean test→train nearest-neighbour cosine | 0.73 |

Splitting is by lesion, not image, so multiple photographs of one lesion cannot straddle the boundary.
Two near-duplicates exist; at 0.4% they cannot materially inflate the headline figure.

**Class imbalance:** SCC has 207 images in total (140 in train) against ~600 for most classes. This is the
single largest driver of the model's worst failure (§6).

---

## 5. Preprocessing audit — **NO DEFECTS FOUND**

| Check | Finding |
|---|---|
| Colour order | RGB throughout. `load_image_rgb` converts once; canvas `getImageData` is RGBA→RGB. No BGR path exists. |
| Resize | 224×224, bilinear |
| Aspect ratio | **Not preserved** — direct resize to square. Consistent between train and inference, so not a train/serve skew; see note below. |
| Normalisation | ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]` — matches PanDerm upstream |
| Tensor shape | `(1, 3, 224, 224)`, verified |
| Pixel range after norm | `[-1.93, 1.75]`, as expected |
| Orientation | No EXIF rotation handling — see BUG-07 |
| Train vs inference | Identical transform object; augmentation only on the training split |

**Is preprocessing destroying lesion information?** Partly, and measurably. A lesion occupying ~10% of a
frame survives as roughly 70×70 px at 224². An experiment that cropped to the segmented lesion before
embedding was run to test this:

| Features | Test macro-F1 |
|---|---|
| Full frame (current) | **0.673 ±0.011** |
| Lesion crop | 0.625 ±0.013 (−0.048) |
| Both concatenated | 0.675 ±0.011 (+0.002, noise) |

Cropping made it **worse**. Surrounding skin carries real signal (sun damage, contrast with normal skin),
and the segmentation adds its own errors. Preprocessing is not the bottleneck.

---

## 6. Classification performance

**Held-out test set, n = 465, no lesion shared with training.**

### Overall

| Metric | Value |
|---|---|
| Accuracy | **0.6731** |
| Balanced accuracy | 0.6635 |
| Top-3 accuracy | **0.9742** |
| Macro F1 | 0.6522 |
| Weighted F1 | 0.6865 |
| Macro precision | 0.7080 |
| Macro recall | 0.6635 |
| **Macro ROC-AUC (OvR)** | **0.9373** |
| Macro PR-AUC | 0.7381 |
| **Cohen's κ** | **0.6065** |
| **Matthews correlation** | **0.6101** |
| Benign vs malignant ROC-AUC | **0.9420** |
| Benign vs malignant accuracy | 0.8690 |

### Per class

| Class | Precision | Recall (sens.) | Specificity | F1 | FNR | ROC-AUC | PR-AUC | n |
|---|---|---|---|---|---|---|---|---|
| NEV | 0.778 | 0.663 | 0.951 | 0.716 | 0.337 | 0.948 | 0.835 | 95 |
| SEK | 0.713 | 0.687 | 0.940 | 0.699 | 0.313 | 0.925 | 0.781 | 83 |
| ACK | 0.899 | 0.667 | 0.981 | 0.765 | 0.333 | 0.956 | 0.894 | 93 |
| BCC | 0.713 | 0.655 | 0.939 | 0.683 | 0.345 | 0.949 | 0.769 | 87 |
| **SCC** | **0.281** | 0.562 | 0.894 | **0.375** | 0.438 | 0.896 | **0.373** | **32** |
| MEL | 0.615 | **0.747** | 0.910 | 0.675 | **0.253** | 0.950 | 0.777 | 75 |

### Confusion matrix (rows = true)

| | NEV | SEK | ACK | BCC | SCC | MEL |
|---|---|---|---|---|---|---|
| **NEV** | **63** | 10 | 0 | 1 | 0 | 21 |
| **SEK** | 6 | **57** | 4 | 2 | 2 | 12 |
| **ACK** | 0 | 6 | **62** | 7 | **18** | 0 |
| **BCC** | 0 | 0 | 3 | **57** | **25** | 2 |
| **SCC** | 0 | 1 | 0 | 13 | **18** | 0 |
| **MEL** | **12** | 6 | 0 | 0 | 1 | **56** |

**SCC is a sink.** 45 images from other classes are predicted SCC against only 18 correct — hence 0.281
precision. The inflow is BCC (25) and ACK (18), the two other keratinocyte lesions. With 140 training
images, the head has not learned a boundary; the class-weight of 2.58× pushes predictions *into* SCC
without the data to place them correctly.

**The decisive gap:** macro ROC-AUC **0.937** against top-1 accuracy **0.673**. The features separate these
classes well; the loss happens in forcing a single six-way argmax. This is why top-3 reaches 97.4%.

---

## 7. Calibration — **GOOD, and the one pleasant surprise**

| Metric | Value |
|---|---|
| Expected Calibration Error (10-bin) | **0.0574** |
| Mean Brier score (OvR) | 0.0697 |
| Mean confidence | 0.6732 |
| Accuracy | 0.6731 |
| **Overconfidence gap** | **+0.0001** |

Mean confidence matches accuracy to four decimal places. Softmax output here is *not* the usual
overconfident garbage — it can be read as an approximate probability, which is what makes the abstention
threshold and the tier-mass rule defensible.

**Abstention behaviour** (threshold 0.5):

| | Value |
|---|---|
| Abstention rate | 19.4% |
| Accuracy when confident | **74.9%** |
| Accuracy when abstaining | **35.6%** |

The gate is selecting the right cases: images it declines are ones it would get wrong twice as often.

See `calibration.png`.

---

## 8. Grad-CAM / attribution — ⛔ **CRITICAL FAILURE**

This is the most serious finding in the audit, and it confirms the user's complaint.

### History

The original implementation used CNN Grad-CAM hooked to a conv layer. On a ViT this produced all-zero maps,
and it was replaced with **attention rollout** (max-fused over heads, from block 6). That was tuned to
maximise *focus ratio* — the share of heat inside the lesion, normalised by lesion area.

### What the audit measured

HAM10000 ships **expert segmentation masks** (10,015 DatasetNinja annotations; 2,570 match our dermoscopic
set). This permits real localization metrics rather than visual inspection. **n = 300.**

| Metric | Deployed rollout | Random baseline |
|---|---|---|
| **Pointing Game** | **0.127** | **0.287** |
| IoU | 0.246 | 0.203 |
| Dice | 0.374 | 0.308 |
| Focus ratio | 1.245 | 1.000 |
| Images with above-chance focus | 46.3% | — |
| Mean lesion area | 30.8% of frame | — |

**The single hottest pixel lands on the lesion less often than random chance.** Pointing Game 0.127 against
a 0.287 baseline means the peak *actively avoids* the lesion. Fewer than half of images have above-chance
focus.

### Root cause

Both facts are true at once — focus ratio 1.245 (bulk of heat on the lesion) and Pointing Game 0.127 (peak
off it) — because a handful of isolated high-attention patches sit on background skin and outrank the
lesion's broad, moderate response. This is the documented **ViT artifact-token effect** (Darcet et al.,
2023, *Vision Transformers Need Registers*): ViTs repurpose low-information background patches as global
scratch space, and those tokens acquire very high attention while carrying no local signal.

**`max`-fusion over heads is the amplifier.** Taking the maximum across heads propagates a single rogue
head's artifact token into the final map; `mean` averages it away.

**And the tuning metric was blind to it.** Focus ratio measures diffuse mass overlap and says nothing about
where the peak sits. The deployed configuration was selected because it scored highest on exactly the
metric that cannot see this failure.

### Variant comparison (n = 200, same expert masks)

| Variant | Pointing | IoU | Dice | Focus | >chance |
|---|---|---|---|---|---|
| **mean-fuse, block 3** | **0.305** | **0.306** | **0.440** | 1.211 | **59.5%** |
| max-fuse b6 + 3×3 median | 0.230 | 0.286 | 0.408 | 1.209 | 51.0% |
| mean-fuse b6 + 3×3 median | 0.195 | 0.272 | 0.388 | 1.147 | 47.0% |
| max-fuse b6 + clip p98 | 0.150 | 0.256 | 0.385 | 1.269 | 50.5% |
| **deployed: max-fuse, block 6** | **0.130** | 0.250 | 0.378 | **1.276** | 47.5% |
| mean-fuse, block 9 | 0.100 | 0.166 | 0.253 | 0.889 | 19.5% |
| mean-fuse, block 6 | 0.090 | 0.190 | 0.295 | 1.113 | 36.0% |
| mean-fuse, block 0 | 0.055 | 0.146 | 0.222 | 0.999 | 29.0% |

**`mean`-fuse from block 3 more than doubles Pointing Game (0.130 → 0.305)**, clears the random baseline,
and improves IoU and Dice simultaneously. The deployed config has the *highest* focus ratio and nearly the
*lowest* pointing accuracy — direct evidence that the selection metric was wrong.

### Visual evidence

`gradcam_examples/rollout_vs_mask.png` — six images, four panels each (photo, deployed map, best variant,
expert mask). In several rows the lesion is rendered **cold (blue)** under the deployed configuration while
bright hotspots sit on blank skin. ISIC_0024467 and ISIC_0024959 are the clearest.

### Honest limits of this measurement

- Masks are **dermoscopic** HAM10000; the deployed classifier is trained on **clinical** photographs. The
  rollout comes from the frozen trunk, which is identical in both cases, so this validly measures the
  backbone's attention — but it is not a measurement on the clinical test split.
- Attention is **not** an explanation. High attention does not establish causal contribution to the
  prediction. These metrics bound localization only.

---

## 9. Error analysis

152 of 465 misclassified. Full table: `misclassified.csv`.

| Error category | Count | Share of errors |
|---|---|---|
| Same risk tier — advice still correct | 57 | 37.5% |
| ⛔ **Dangerous downgrade (High → Low)** | **19** | **12.5%** |
| Upgraded (errs safe) | 54 | 35.5% |
| Other downgrades | 22 | 14.5% |

**19 dangerous downgrades = 4.1% of the whole test set.** These are High-risk lesions reported as Low risk.
The dominant contributor is **MEL → NEV (12 cases)**: melanomas called ordinary moles. See
`misclassified_examples/dangerous_downgrades.png`.

**Most frequent confusions**, all clinically coherent:

| Confusion | Count | Same tier? |
|---|---|---|
| BCC → SCC | 25 | ✅ both High |
| NEV → MEL | 21 | ⚠️ false alarm (safe) |
| ACK → SCC | 18 | ⚠️ upgrade (safe) |
| SCC → BCC | 13 | ✅ both High |
| **MEL → NEV** | **12** | ⛔ **High → Low** |
| SEK → MEL | 12 | ⚠️ false alarm (safe) |

The model confuses lesions that dermatologists also confuse, and most errors keep the advice correct. The
MEL↔NEV pair is the exception and the one that matters.

**Overfitting / underfitting:** training history shows train loss 1.79 → 1.05 with validation macro-F1
still rising at epoch 14 — the model is *under*-trained in the frozen configuration rather than overfitted.
2,170 training images against a 0.2M-parameter head is the binding constraint.

**Domain shift:** training on clinical, evaluating on dermoscopic drops accuracy to 63.5%. Within-domain
dermoscopic training gives 69.2% — statistically the same as clinical's 70.2%, so the imaging modality is
*not* the limiting factor.

---

## 10. Robustness — ⚠️ **BRITTLE**

Synthetic perturbations of the full test split (n = 465). **A drop indicates brittleness; holding up is not
clinical validation.**

| Perturbation | Accuracy | Δ | MEL recall | MEL shown Low | Mean conf |
|---|---|---|---|---|---|
| none (baseline) | 0.673 | — | 0.747 | 24.0% | 0.673 |
| horizontal flip | 0.686 | +0.013 | 0.760 | 22.7% | 0.673 |
| rotate 90° | 0.671 | −0.002 | 0.747 | 24.0% | 0.671 |
| brightness ×0.6 | 0.654 | −0.019 | 0.720 | 26.7% | 0.646 |
| brightness ×1.4 | 0.634 | −0.039 | 0.627 | 20.0% | 0.651 |
| contrast ×0.5 | 0.634 | −0.039 | 0.560 | 42.7% | 0.624 |
| **rotate 10°** | 0.626 | −0.047 | **0.507** | **45.3%** | 0.644 |
| downscale to 64px | 0.488 | −0.185 | 0.520 | 48.0% | 0.647 |
| **JPEG quality 20** | 0.458 | −0.215 | 0.387 | **60.0%** | 0.664 |
| greyscale | 0.430 | −0.243 | 0.693 | 29.3% | 0.614 |
| **Gaussian blur r=2** | **0.404** | **−0.269** | **0.347** | **65.3%** | 0.657 |

Three findings:

1. **Blur is catastrophic.** Accuracy −0.269, melanoma recall halved, and **65.3% of melanomas shown as Low
   risk**. The quality gate must catch blur, because the model will not.
2. **A 10° rotation nearly halves melanoma recall** (0.747 → 0.507) while costing only 0.047 overall
   accuracy. Every handheld phone photo has some rotation. 90° and flips cost nothing, so this is
   interpolation sensitivity, not orientation sensitivity — and it points at missing rotation augmentation.
3. ⛔ **Confidence barely moves while accuracy collapses.** Under blur, accuracy falls to 0.404 while mean
   confidence stays at 0.657 (baseline 0.673). The abstention gate — which depends on confidence — **will
   not fire on degraded images**. Calibration measured on clean data does not survive perturbation.

Greyscale costing 0.243 confirms the model uses colour, which is clinically appropriate.

---

## 11. GPU and inference performance

| Property | Value |
|---|---|
| CUDA available | **True** |
| PyTorch | 2.12.1+cu126 |
| GPU | NVIDIA GeForce RTX 4050 Laptop, 6.4 GB |
| Model device | `cuda:0` |
| Single-image latency | **9.48 ms** |
| Batch-32 throughput | **149.1 img/s** |
| Peak VRAM | 598.7 MB |

**Finding, now resolved:** the environment had `torch 2.12.1+cpu` installed — a CPU-only build — so the
RTX 4050 sat unused and fine-tuning ran at 3.2 img/s. A CUDA build was installed into a separate venv at
`C:\Users\chint\gpuenv` (`--system-site-packages`, same torch version, `+cu126` chosen because driver
573.05 predates the CUDA 13 runtime). Throughput rose to **87 img/s during training — a 27× speedup**.

**Note:** these figures are for the *Python* path. The deployed app runs int8 ONNX under WASM in a Web
Worker on the user's CPU; the GPU is irrelevant to production latency.

---

## 12. Bug list

| ID | Sev | File | Problem | Fix |
|---|---|---|---|---|
| **BUG-01** | ⛔ CRITICAL | `scripts/export_onnx.py` `TrunkWrapper.forward` ~L133 | `max`-fusion from block 6 gives Pointing Game **0.127 vs 0.287 random**. Attention peak lands on ViT artifact tokens, not the lesion. | Switch to `a.mean(dim=1)` and `ROLLOUT_START_LAYER = 3`. Measured: Pointing 0.305, IoU 0.306, Dice 0.440. Requires ONNX re-export + parity re-run. |
| **BUG-02** | ⛔ CRITICAL | `scripts/tune_rollout.py` | Rollout was selected by **focus ratio alone**, which is blind to peak placement — it chose the worst-pointing variant. | Add Pointing Game / IoU against expert masks to the selection metric. |
| **BUG-03** | 🔴 HIGH | model / `configs/default.yaml` | SCC: precision **0.281**, PR-AUC 0.373, 140 train images. 45 false positives vs 18 true. Class weight 2.58× pushes predictions in without data to place them. | Collect SCC data; or merge SCC+BCC into "keratinocyte carcinoma" (both High risk, so no advice is lost); or cap the class weight. |
| **BUG-04** | 🔴 HIGH | model behaviour | **19 dangerous downgrades (4.1%)**, 12 of them MEL→NEV. | Asymmetric loss penalising High→Low errors; or lower the abstention threshold for MEL-adjacent predictions. |
| **BUG-05** | 🔴 HIGH | `src/model/fusion.py` + training | Metadata dominates. All-unknown metadata drops top-1 **67.3% → 56.6%** and MEL recall **74.7% → 33.3%**. Three optional form fields outweigh 768-d image features. | Retrain with **metadata dropout** (randomly zero it, p≈0.3) so image evidence must stand alone. |
| **BUG-06** | 🔴 HIGH | model robustness | Confidence does not fall when input degrades (blur: acc 0.404, conf 0.657). Abstention cannot fire on bad images. | Tighten the blur threshold in the quality gate; consider entropy- or TTA-variance-based uncertainty instead of max-softmax. |
| **BUG-07** | 🟡 MED | `web/lib/pipeline.js` | No EXIF orientation handling. 10° rotation halves MEL recall (0.747→0.507), so orientation errors are not benign. | Read EXIF and rotate before preprocessing; add rotation augmentation to training. |
| **BUG-08** | 🟡 MED | `src/model/train.py` | No rotation/scale augmentation in the frozen path (embeddings are cached once, so augmentation is impossible by construction). | Augment during embedding extraction, or move to the fine-tuned path where augmentation is already active. |
| **BUG-09** | 🟡 MED | `web/lib/model.js` | Heatmap percentile normalisation (p50→p99) **clips the top 1%**, hiding the artifact spikes rather than removing them — the display masks BUG-01 instead of fixing it. | Fix at source (BUG-01); optionally add a 3×3 median filter on the 14×14 grid. |
| **BUG-10** | 🟢 LOW | `scripts/finetune_backbone.py` | Two concurrent runs wrote the same checkpoint path; the CPU run overwrote the GPU run's better weights (val macro-F1 0.7124 → 0.6606). | Already mitigated by the `--tag` flag added during this work. |
| **BUG-11** | 🟢 LOW | docs | `PROJECT.md` carries pre-rollout metrics. | Regenerate from `audit_reports/evaluation_metrics.json`. |

---

## 13. Prioritised roadmap

**Immediate (hours, no retraining)**
1. **BUG-01** — switch rollout to `mean` fusion from block 3, re-export ONNX, re-run parity. Doubles
   localization quality; the heatmap is currently worse than random at pointing.
2. **BUG-02** — add Pointing Game to the rollout selection metric so this cannot recur.
3. **BUG-06** — tighten the blur threshold. The model stays confident on unusable photos.

**Short term (days, retraining)**
4. **BUG-05** — retrain with metadata dropout. Highest-value change: removes the single largest source of
   confident wrong answers.
5. **BUG-04** — asymmetric loss for High→Low errors.
6. **BUG-07/08** — EXIF handling plus rotation augmentation.

**Medium term**
7. **BUG-03** — SCC data collection, or merge into a keratinocyte-carcinoma class.
8. Deploy the fine-tuned backbone (71.2% top-1, 71.4% balanced) **only after** re-tuning the abstention
   threshold against its shifted probability distribution — fine-tuning made one known melanoma case worse
   (NEV 46.8% → 51.4%, crossing the threshold).

**Do not do**
- Replace PanDerm. Fine-tuned, it beats a matched ImageNet backbone by 4.5 points.
- Crop to the lesion before classification. Measured −0.048 macro-F1.
- Switch to dermoscopic imaging expecting a gain. Measured 69.2% vs 70.2% clinical — no difference.

---

## 14. Summary

**What works**
- PanDerm integration is correct, verified bit-identical to the official checkpoint.
- Preprocessing is correct and consistent between training and inference.
- Data splitting is clean — zero lesion leakage, 2 near-duplicates in 465.
- **Calibration is excellent** — overconfidence gap +0.0001.
- Abstention selects the right cases (74.9% accurate when confident, 35.6% when abstaining).
- Discrimination is strong — macro ROC-AUC 0.937, benign/malignant AUC 0.942, top-3 97.4%.
- Most errors preserve the correct risk tier.

**What is broken**
- ⛔ **The heatmap.** Pointing Game 0.127 against 0.287 random — the peak avoids the lesion.
- ⛔ **SCC classification.** Precision 0.281, a sink absorbing BCC and ACK.
- 🔴 **4.1% dangerous downgrades**, 12 melanomas called moles.
- 🔴 **Metadata dominance** — three form fields outweigh the image.
- 🔴 **Brittleness with flat confidence** — blur halves melanoma recall while confidence holds.

**Why**
- The heatmap fails because `max`-fusion amplifies ViT artifact tokens, and the tuning metric could not see
  peak placement.
- SCC fails from data scarcity (140 training images) amplified by a 2.58× class weight.
- Metadata dominates because 2,170 images are too few for the head to prefer noisy 768-d features over
  three clean low-dimensional ones.

**Evidence**
Everything above comes from `audit_reports/`: `evaluation_metrics.json`, `localization_metrics.json`,
`rollout_variants.json`, `robustness_results.json`, `misclassified.csv`, `classification_report.csv`,
`confusion_matrix.png`, `calibration.png`, `gradcam_examples/`, `misclassified_examples/`. Every figure
states its split and sample count. No metric in this report was estimated or carried over from a prior run.

**Fix first:** BUG-01. The explanation shown to users is currently worse than random at pointing to the
lesion, the fix is measured and ready, and it needs no retraining.
