# Final deployment readiness — attention rollout fix

**Date:** 2026-11-21
**Change:** attention rollout `max`/block-6 → `mean`/block-3, plus heatmap normalisation
**Verdict:** ✅ **READY TO DEPLOY** — every critical test passes
**Deployed:** ❌ **No.** The live site still runs the old configuration.

---

## Verdict summary

| Task | Result |
|---|---|
| 1 — Classification parity | ✅ **PASS** — 465/465 identical, all stages `0.000e+00` |
| 2 — Quantization | ✅ **FIXED** — no `ConvInteger`, Conv fp32, regression guard added |
| 3 — Heatmap verification | ✅ **PASS** — all 5 checks |
| 4 — Browser testing | ✅ **PASS** — 9/9 identical, 0 console errors, camera works |
| 5 — Deployment readiness | ✅ **PASS** — hashes correct, cache bust verified |

**Localization is improved but NOT clinically validated.** See §7.

---

## 1. Classification parity ✅

Full chain, **all 465 held-out test images**, deployed int8 vs new int8:

| Stage | Max abs diff | Agreement |
|---|---|---|
| Pooled embedding (CLS token) | **0.000e+00** | — |
| Logits | **0.000e+00** | — |
| Softmax probabilities | **0.000e+00** | — |
| Predicted class | — | **465/465 (100%)** |
| Risk tier | — | **465/465 (100%)** |

Bit-identical. The rollout runs as a side branch — `x = blk(...)` is untouched — so the change cannot
alter a single prediction. This is measured, not argued.

### Supporting gates

| Check | Result |
|---|---|
| ONNX fp32 trunk vs PyTorch | 5.10e-05 |
| ONNX rollout vs PyTorch | 1.49e-08 |
| `verify_web_model.py` fp32 | 30/30 agree, drift 0.00000 |
| `verify_web_model.py` int8 | 29/30 agree, drift 0.10798 *(unchanged from baseline)* |

### Pooling consistency — checked, and clean

`web/lib/model.js` takes `tokens.slice(0, dim)` — the CLS token.
`src/backbone/panderm_wrapper.py` returns `self.model(images)`.

| Comparison | Max abs diff |
|---|---|
| Python backbone vs **CLS token** | **0.000e+00** |
| Python backbone vs mean-of-tokens | 1.445e+01 |

The browser and the Python pipeline pool identically. No train/serve skew.

*(An interim measurement in this audit mean-pooled over all 197 tokens and produced a spurious "1 in 9
predictions differ" figure. That was a bug in the audit script, not in the application. Corrected here.)*

---

## 2. Quantization ✅ fixed

`scripts/quantize_onnx.py` had three faults and could not rebuild the deployed artifact:

1. Defaulted to `web/model/panderm_risk.onnx` — a combined model that no longer exists.
2. Used `nodes_to_exclude` on head nodes instead of `op_types_to_quantize=["MatMul"]`.
3. Consequently emitted **`ConvInteger`**, an op **ONNX Runtime Web has no kernel for**. The model would
   load in the browser and then throw at first inference. Measured side effects: token drift **7.2**,
   embedding drift **0.18**.

### After the fix

| Property | Deployed (old) | New | Match |
|---|---|---|---|
| `Conv` (fp32) | 1 | 1 | ✅ |
| `MatMulInteger` | 48 | 48 | ✅ |
| `ConvInteger` | 0 | **0** | ✅ |
| Size | 84.5 MB | 85 MB | ✅ |

Keeping `Conv` in fp32 also preserves the property that `patch_embed` stays **bit-identical to PanDerm's
released weights**.

**Regression guard added.** `assert_web_compatible()` raises if a `ConvInteger` appears, or if the
patch-embedding Conv stops being fp32. This failure cannot ship silently again.

### int8 cost — measured, and negligible

| Pipeline | Top-1 | Top-3 | MEL recall | MEL shown Low |
|---|---|---|---|---|
| fp32 | 0.673 | 0.974 | 0.747 | 24.0% |
| **int8 (what the browser runs)** | **0.673** | 0.966 | **0.747** | 24.0% |

**Top-1 identical to four decimal places.** Individual predictions differ on 27/465 images (94.2%
agreement, max probability drift 0.521), but the disagreements cancel and net accuracy is unchanged.
Top-3 costs 0.008. **The published accuracy figures describe what users actually get.**

---

## 3. Heatmap verification ✅

| Check | Result | Detail |
|---|---|---|
| 14×14 attention map | ✅ | 196 patches, 14 per side, from 224/16 |
| No artificial clipping | ✅ | patches at ceiling: old 2.00 → new **1.00** |
| No arbitrary hotspot ties | ✅ | patches at max: old **2.00** → new **1.00** (unique) |
| Original-image alignment | ✅ | IoU **0.335** vs 0.214 random, on 120 non-square frames |
| RGB/RGBA handling | ✅ | RGBA throughout; no BGR path exists |

**The tie finding matters.** Under the old p99 ceiling, **exactly 2 patches sat at the maximum** — p99 of
196 values clips the top two. The "hottest pixel" was therefore decided by array order, not by attention.
The new normalisation leaves a single unique maximum.

**Alignment is established statistically, not by eye.** A mis-mapped 14×14 grid would collapse IoU to
chance, and would fail on non-square frames first. All 120 test images were 600×450 and IoU held at
0.335. The model sees a square 224×224 while the display preserves aspect ratio, but photo and heatmap are
stretched by identical factors, so the linear mapping holds.

---

## 4. Browser testing ✅

Production (live Vercel, old) vs staging (local 8794, new model + new JS):

| Case | Production | Staging | Match |
|---|---|---|---|
| original 600×450 | NOT CONFIDENT / High | NOT CONFIDENT / High | ✅ |
| upscaled 1200×900 | NOT CONFIDENT / High | NOT CONFIDENT / High | ✅ |
| downscaled 320×240 | NOT CONFIDENT / High | NOT CONFIDENT / High | ✅ |
| square 448×448 | NOT CONFIDENT / High | NOT CONFIDENT / High | ✅ |
| wide 16:9 800×450 | NOT CONFIDENT / High | NOT CONFIDENT / High | ✅ |
| tall 3:4 480×640 | NOT CONFIDENT / High | NOT CONFIDENT / High | ✅ |
| bright +40% | NOT CONFIDENT / High | NOT CONFIDENT / High | ✅ |
| dim −40% | NOT CONFIDENT / High | NOT CONFIDENT / High | ✅ |
| camera-like 1920 q85 | NOT CONFIDENT / High | NOT CONFIDENT / High | ✅ |

**9/9 identical. Zero console errors on either build.**

**Camera path** exercised through Chromium's fake capture device: `getUserMedia` started, capture
succeeded, result rendered with both canvases, no errors.

**Qualification, stated plainly:** all nine variants derive from one source image, which abstains at High
risk, so every row reads the same. This test establishes *no crash, no console error, no behavioural
change across input shapes* — it is weak evidence of discrimination. The strong parity evidence is Task 1:
465 distinct images at bit-identical logits.

---

## 5. Deployment readiness ✅

| Check | Result |
|---|---|
| New ONNX + new JS compatible | ✅ verified together on staging, 0 errors |
| Staged `app.js` hash matches staged model | ✅ `fc92e506` = `fc92e506` |
| `head.json` unchanged | ✅ byte-identical, hash `63f065b4` (head was not retrained) |
| Cache bust effective | ✅ URL changes `?v=b8a38550` → `?v=fc92e506` |
| Deployed artifacts untouched | ✅ `web/model/` clean in git; hash still `b8a38550` |
| Original checkpoints preserved | ✅ all intact |

Model files are served `immutable, max-age=31536000`. Without the hash change a returning browser would
keep the old model forever — pairing an old graph with new display code. The staged `app.js` carries the
correct new hash.

---

## 6. Localization outcome

Validated on **400 held-out masked images**, disjoint from the 300 used to select the configuration:

| Metric | Old | New | Random | Δ | 95% CI | Bootstraps favouring new |
|---|---|---|---|---|---|---|
| Pointing Game | 0.128 | **0.273** | 0.297 | +0.145 | [+0.098, +0.193] | 100% |
| IoU | 0.257 | **0.325** | 0.214 | +0.068 | [+0.059, +0.076] | 100% |
| Dice | 0.387 | **0.462** | 0.321 | +0.076 | [+0.067, +0.085] | 100% |

77% of individual images improved on IoU. Focus ratio **fell** (1.236 → 1.198) while all three real
metrics rose — confirming it was the wrong selection metric.

---

## 7. Remaining risks

**R1 — Pointing Game is still at chance.** 0.273 against a 0.297 random baseline. Region overlap is
genuinely better than chance (IoU 0.325 vs 0.214), but the single hottest pixel is not. The heatmap is a
rough region indicator; it must not be read as "the model looked *here*". ⚠️ Unresolved.

**R2 — Localization is NOT clinically validated.** Masks are dermoscopic HAM10000 while the deployed
classifier trains on clinical photographs. The rollout comes from the frozen trunk, which is shared, so
the measurement is valid for backbone attention — but no clinical-image masks exist, and nothing here
constitutes clinical validation.

**R3 — Attention is not causal attribution.** A high rollout value means the CLS token's attention reaches
that patch. It does not establish that the patch drove the prediction.

**R4 — Repo/deploy mismatch while staged.** `web/lib/model.js` and `scripts/export_onnx.py` are updated in
the repository while the deployed model still carries the old rollout. The new normalisation on the old
map is untested together — the p99 clip existed specifically to tame the old spikes. **Deploy the model
and the JS together, or revert the JS until you do.**

**R5 — Unrelated open issues** from the main audit, untouched by this change: SCC precision 0.281; 19
dangerous High→Low downgrades (4.1%); metadata dominance; confidence staying flat under image degradation.

---

## 8. Deployment instructions

Nothing below has been performed.

```bash
SRC="<scratch>/stage/model"          # staged artifacts
cd /c/Users/chint/OneDrive/Desktop/SCD

# 1. Copy the new model (head.json is unchanged; copying it is harmless)
cp "$SRC/panderm_vit_int8.onnx" web/model/

# 2. Bump the cache hash in web/app.js
#    MODEL_URL: ?v=b8a38550  ->  ?v=fc92e506
#    HEAD_URL:  ?v=63f065b4  (unchanged)

# 3. Confirm the reference matches the file
sha256sum web/model/panderm_vit_int8.onnx | cut -c1-8   # expect fc92e506
grep -oE "v=[a-f0-9]+" web/app.js

# 4. Drive it locally before shipping
python <scratch>/serve_any.py 8791 web
python <scratch>/browser_test.py

# 5. Deploy
cd web && npx vercel deploy --prod --yes --scope rohan0639s-projects

# 6. Verify live: hash served, heatmap renders, no console errors
curl -s https://skin-cancer-risk-screening.vercel.app/app.js | grep -oE "v=[a-f0-9]+"
```

**Rollback:** redeploy the previous Vercel deployment, or restore
`web/model/panderm_vit_int8.onnx` (hash `b8a38550`) and revert the `app.js` hash and the `model.js`
normalisation together.

---

## 9. Evidence

```
audit_reports/
  FINAL_DEPLOYMENT_READINESS.md      this document
  deploy_parity.json                 Task 1, 465 images, full chain
  heatmap_verification.json          Task 3, 5 checks
  browser_test.json                  Task 4, production vs staging
  int8_accuracy.json                 int8 vs fp32 accuracy + pooling check
  rollout_parity.json                old vs new int8 tokens
  rollout_fix_report.md              implementation and validation
  updated_localization_metrics.json  400-image held-out validation
  rollout_validation_per_image.csv   per-image results
  rollout_comparison.png             old vs new vs random
```

Scripts: `audit_deploy_parity.py`, `audit_heatmap_verify.py`, `audit_rollout_validate.py`,
`audit_rollout_fix_verify.py`, `quantize_onnx.py` (fixed).
