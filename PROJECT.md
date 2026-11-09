# Skin Cancer Detection with Explainable Classification and Risk Assessment

A patient-facing skin-lesion screening prototype. Upload a phone photo plus a
few details and it returns a risk level, a heatmap of what the model looked at,
and a plain-language ABCDE explanation.

> ## ⚠️ Not a medical device
> A student research prototype. Not clinically validated, not a diagnosis, and
> its output must never be used to rule out skin cancer. If you are worried
> about a mark on your skin — or it is changing, bleeding, itching or not
> healing — see a doctor regardless of what this says.

---

## Links

| | |
|---|---|
| Live app | https://skin-cancer-risk-screening.vercel.app |
| Source | https://github.com/Rohan0639/skin-cancer-risk-app |
| Model weights | https://huggingface.co/rohan0639/panderm-base-lp-fp16 |

---

## Architecture

The whole system runs **in the visitor's browser**. Vercel serves static files
only; no Python backend, no server-side inference, and the photo never leaves
the device.

```
phone photo
   │
   ├─ 1. quality gate      blur · exposure · contrast · resolution · framing
   ├─ 2. segmentation      flat-fielded Otsu → component pick → hole fill
   ├─ 3. PanDerm           ViT-B/16 → 197 tokens × 768 dims   [ONNX, int8]
   ├─ 4. fusion head       CLS token + age/sex/site → logits  [JS]
   ├─ 5. relevance map     head applied per patch token → 14×14 heatmap
   ├─ 6. ABCDE             asymmetry · border · colour · diameter · evolving
   └─ 7. report            one recommendation; evidence may only escalate
```

### Why in-browser

Vercel serverless functions cap at 250 MB. The app needs ~95 MB of
dependencies plus a 172 MB model, and fetching weights at cold start exceeds
the 10 s timeout. Browser-side inference has no such limits — and gives
privacy for free.

---

## Tech stack

**Training (Python):** PyTorch · timm · OpenCV · scikit-learn · ONNX
**Serving (browser):** ONNX Runtime Web (WASM, self-hosted) · vanilla JS
**Backbone:** PanDerm `PanDerm_Base_LP` — Yan et al., *Nature Medicine* 2025
**Host:** Vercel (static)

---

## Repository layout

```
configs/default.yaml          every path, threshold and label mapping
data/
  prepare_isic.py             real clinical images from the ISIC Archive
  prepare_ham10000.py         HAM10000 → unified schema
  prepare_pad_ufes20.py       PAD-UFES-20 → unified schema
  make_demo_dataset.py        synthetic data for smoke-testing
src/
  config.py utils.py dataset.py data_prep.py segmentation.py
  pipeline.py evaluate.py check_backbone.py
  quality/image_quality.py    the quality gate
  backbone/                   PanDerm wrapper + ImageNet fallback
  model/                      fusion head, classifier, training
  explain/                    Grad-CAM, ABCDE
  tracking/change_detection.py
  report/report_generator.py
scripts/
  export_onnx.py              ViT trunk → ONNX, head → JSON
  quantize_onnx.py            int8 quantisation + accuracy measurement
  verify_web_model.py         browser model vs PyTorch parity gate
web/                          the deployed site (HTML + JS + model)
tests/                        225 tests
```

---

## Verified measurements

Everything below was measured, not assumed.

| Check | Result |
|---|---|
| ONNX fp32 trunk + JS head vs PyTorch | **30/30 agree, 0.00000 drift** |
| int8 quantisation cost | 29/30 agree (327 → 85 MB) |
| Python test suite | **225 passing** |
| PanDerm cold-start download | 72 s, loads, `is_fallback: False` |
| Peak RSS (PyTorch path) | 1.18 GB |

---

## Design decisions worth knowing

**Diameter is not scored without a scale.** A photo has no scale reference.
Converting pixels to millimetres via an assumed field of view flagged the "D"
criterion on *100% of lesions*. It now reports relative size and is marked
*not measurable* unless the user supplies the frame width.

**The attention check is relative, not absolute.** "Only 23% of attention fell
on the lesion" sounds bad until you note the lesion covers 14% of the frame —
that is 1.6× better than chance. The check uses the scale-free **focus ratio**,
with two thresholds: below 1.2 warns the user, below 1.0 (worse than chance)
also escalates the risk level.

**Advice escalates, never de-escalates.** Low confidence, detected change, ≥3
ABCDE flags, or a heatmap worse than chance each push the level one step
*towards* seeing a doctor.

**The contrast check rejected only benign lesions.** At `contrast_std_min: 15`
it failed 10% of images and *every one* was Low-risk — a pale mark on even skin
genuinely has low whole-image contrast. Lowered to 5.0, which still catches
genuinely blank frames (those score under 2).

**Model selection breaks ties on loss.** Macro-F1 saturates on small datasets,
and the earliest epoch to hit the ceiling wins — an under-trained, badly
calibrated model. Selecting on `(macro-F1, −loss)` fixed 52% → 99% confidence
at identical accuracy.

**Not Grad-CAM in the browser.** ONNX Runtime Web has no backward pass. The
heatmap applies the classifier to each patch token instead — the same question,
computed without gradients. Labelled as a relevance map, not Grad-CAM.

---

## Known limitations

1. **No validated performance numbers.** The deployed classifier was trained on
   synthetic images to verify the pipeline. It demonstrates the system end to
   end; it is not a measured clinical result.
2. **Skin-tone bias.** Dermatology datasets over-represent lighter skin. Expect
   worse performance on darker skin. Nothing here corrects for it.
3. **Three risk tiers are a triage simplification**, not a clinical grading
   system.
4. **Segmentation is classical, not learned** — and the browser build drops
   GrabCut refinement (OpenCV-only), so masks are coarser than in Python.
5. **int8 quantisation** costs roughly one flipped prediction in thirty.

---

## Status — work in progress

Lesion-**type** classification is **partially implemented**.

**Done**
- Downloaded **2,203 real clinical lesion photographs** from the ISIC Archive
  across six diagnoses — MEL 400, NEV 400, BCC 400, SEK 400, ACK 396, SCC 207
  — with real age, sex and body site. Clinical close-ups only, matching the
  phone photos the app receives.
- `data/prepare_isic.py`; diagnosis classes, patient-facing names and
  descriptions in the config; `risk_classifier.py` now predicts a diagnosis and
  derives the risk tier from it.

**Not done**
- `train.py`, `evaluate.py`, `pipeline.py` and the web UI still assume three
  risk classes
- Model not retrained on the real data
- ONNX not re-exported; nothing redeployed
- Interface not yet trimmed

**The live site is therefore unchanged** — still risk-only and
synthetic-trained. The repository is mid-refactor and will not train until the
remaining files are updated.

**To finish:** complete the refactor → retrain on ISIC → re-export and quantise
→ redeploy with a trimmed UI.

---

## Running it

```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python data/prepare_isic.py --per-class 400   # real data
python src/model/train.py
python src/evaluate.py
python -m pytest
```

Rebuild the web model:

```bash
python scripts/export_onnx.py --out web/model
python scripts/quantize_onnx.py
python scripts/verify_web_model.py
cd web && npx vercel deploy --prod
```

---

## Licences and attribution

**PanDerm** — Yan et al., "A general-purpose multimodal foundation model for
dermatology", *Nature Medicine* (2025).
https://github.com/SiyuanYan1/PanDerm — weights **CC-BY-NC**: non-commercial
academic research only, with attribution. This project complies on that basis.
Do not deploy commercially.

**ISIC Archive** — International Skin Imaging Collaboration, https://isic-archive.com
**HAM10000** — Tschandl et al., *Scientific Data* (2018), doi:10.1038/sdata.2018.161
**PAD-UFES-20** — Pacheco et al., *Data in Brief* (2020), doi:10.1016/j.dib.2020.106221
**Grad-CAM** — Selvaraju et al., *ICCV* (2017)

Datasets are not redistributed; `data/raw/` is gitignored.
