# Vercel production update — attention rollout fix

**Date:** 2026-11-21
**Change:** attention rollout `max`/block-6 → `mean`/block-3, plus heatmap display normalisation
**Status:** ✅ **DEPLOYED AND VERIFIED LIVE**

---

## 1. Deployment summary

| Field | Value |
|---|---|
| Existing project updated | ✅ `skin-cancer-risk-screening` |
| Project ID | `prj_0lYvnnQz3iWJXyX0DM2KwdCJjks0` |
| Org ID | `team_2wYNuGIOcCFXEw7qwsw5Cl19` |
| **Deployment ID** | **`dpl_AKPwcNXs2uqcHE1xUqTDdt271Epo`** |
| Deployment URL | `skin-cancer-risk-screening-9sv6tfwiw-rohan0639s-projects.vercel.app` |
| **Production URL** | **https://skin-cancer-risk-screening.vercel.app** |
| Inspector | `vercel.com/rohan0639s-projects/skin-cancer-risk-screening/AKPwcNXs2uqcHE1xUqTDdt271Epo` |
| Ready state | `READY`, target `production` |
| New project created | ❌ No |
| Domain changed | ❌ No |
| Previous deployments deleted | ❌ No |

The project was already linked in `web/.vercel/project.json`, so `vercel deploy --prod` updated it in
place. `vercel link` was never run.

### Model versions

| Artifact | Before | After |
|---|---|---|
| `panderm_vit_int8.onnx` | `b8a38550` | **`fc92e506`** |
| `head.json` | `63f065b4` | `63f065b4` *(unchanged — head not retrained)* |
| `app.js` MODEL_URL | `?v=b8a38550` | `?v=fc92e506` |
| `app.js` HEAD_URL | `?v=63f065b4` | `?v=63f065b4` |

---

## 2. Pre-deployment verification

Run against the exact files that were deployed, with the new model and new normalisation together for
the first time:

| Check | Result |
|---|---|
| ONNX web compatibility | ✅ `Conv(fp32) 1`, `MatMulInteger 48`, **`ConvInteger 0`** |
| Model hash on disk | ✅ `fc92e506` |
| `head.json` identical to deployed | ✅ byte-identical |
| Classification parity | ✅ 465/465, embeddings/logits/softmax `0.000e+00` |
| Browser suite (9 input variants) | ✅ 9/9 identical to production |
| Console errors | ✅ 0 |
| Camera capture | ✅ working |

Classification parity was not re-run after the copy: the file placed in `web/model/` is byte-identical
(hash `fc92e506`) to the one that scored 465/465.

---

## 3. Live verification

Against **https://skin-cancer-risk-screening.vercel.app** after deployment:

### Artifact level

| Check | Result |
|---|---|
| Served model hash | ✅ `fc92e506` |
| `app.js` references | ✅ `v=fc92e506`, `v=63f065b4` |
| `model.js` normalisation fix | ✅ present |

### Browser level

| Case | Result | Canvases | Heatmap |
|---|---|---|---|
| original 600×450 | NOT CONFIDENT / High risk | 2 | ✅ |
| upscaled 1200×900 | NOT CONFIDENT / High risk | 2 | ✅ |
| tall 3:4 480×640 | NOT CONFIDENT / High risk | 2 | ✅ |
| dim −40% | NOT CONFIDENT / High risk | 2 | ✅ |
| **camera capture** | NOT CONFIDENT / Moderate risk | 2 | ✅ |

**5/5 rendering correctly. 0 console errors.**

### Cache bust confirmed

The live browser requested:

```
panderm_vit_int8.onnx?v=fc92e506      <- the NEW model
head.json?v=63f065b4
model.js
```

This is the check that matters most. Model files are served `immutable, max-age=31536000`. Without the
hash change a returning browser would keep `b8a38550` indefinitely and pair an old attention graph with
new display code — classification would look correct while only the heatmap was stale. The request log
proves the new model is being fetched.

---

## 4. Deployment issue encountered

**The first deploy attempt failed with `Error: Not authorized`.**

`vercel whoami` returned `rohan0639` and `vercel ls` worked normally, so authentication was valid. An
immediate retry with the identical command succeeded. Before retrying, production was confirmed still
serving `b8a38550`, so the failed attempt applied nothing and left no partial state.

Transient, on Vercel's side. Recorded here because the deployment was not a clean single command.

---

## 5. Rollback capability

**Preserved by two independent routes.**

⚠️ `web/model/*.onnx` is **gitignored** (`.gitignore:44`), so git does **not** preserve the previous
model. A manual backup was taken before overwriting:

```
C:/Users/chint/scd_model_backup/panderm_vit_int8_b8a38550.onnx   (hash verified b8a38550)
C:/Users/chint/scd_model_backup/head_63f065b4.json
```

**Route 1 — Vercel (fastest).** Previous production deployments remain `Ready` and were not deleted:

| Age | Deployment | State |
|---|---|---|
| ~6h | `skin-cancer-risk-screening-9fqr0fxjs-…` | Ready, Production |
| ~7h | `skin-cancer-risk-screening-f8zqz1z2c-…` | Ready, Production |

Promote `9fqr0fxjs` from the Vercel dashboard or with `vercel rollback`.

**Route 2 — local rebuild:**

```bash
cp C:/Users/chint/scd_model_backup/panderm_vit_int8_b8a38550.onnx web/model/panderm_vit_int8.onnx
# revert web/app.js MODEL_URL to ?v=b8a38550
git checkout web/lib/model.js          # restores the p50-p99 normalisation
cd web && npx vercel deploy --prod --yes --scope rohan0639s-projects
```

**Revert the model and `model.js` together.** The p99 clip existed to tame the old rollout's artifact
spikes; the new normalisation on the old map is an untested combination.

---

## 6. What changed for users, and what did not

**Changed — the heatmap only.** Validated on 400 held-out images with expert masks, disjoint from the
300 used to select the configuration:

| Metric | Old | New | Random | 95% CI of change |
|---|---|---|---|---|
| Pointing Game | 0.128 | 0.273 | 0.297 | [+0.098, +0.193] |
| IoU | 0.257 | **0.325** | 0.214 | [+0.059, +0.076] |
| Dice | 0.387 | **0.462** | 0.321 | [+0.067, +0.085] |

**Unchanged — every prediction.** Classification is bit-identical: 465/465 agreement, embeddings, logits
and softmax all differing by `0.000e+00`. Accuracy (67.3%), melanoma recall (74.7%), the abstention rate
(19.4%) and the dangerous-downgrade rate (4.1%) are exactly as before. No weights were retrained.

---

## 7. Remaining issues

**Not fixed by this change.**

| # | Issue | Status |
|---|---|---|
| 1 | **Pointing Game 0.273 still below the 0.297 random baseline.** Region overlap now beats chance; peak placement does not. The heatmap is a rough region indicator, not "the model looked *here*". | ⚠️ Open |
| 2 | **Localization is NOT clinically validated.** Masks are dermoscopic HAM10000; the classifier trains on clinical photographs. Attention is not causal attribution. | ⚠️ By design |
| 3 | **SCC precision 0.281** — a sink absorbing BCC and ACK; 140 training images. | 🔴 Open |
| 4 | **19 dangerous downgrades (4.1%)**, 12 of them melanomas called moles. | 🔴 Open |
| 5 | **Metadata dominance** — all-unknown metadata drops melanoma recall 74.7% → 33.3%. | 🔴 Open |
| 6 | **Confidence stays flat as images degrade** — under blur, accuracy falls to 0.404 while confidence holds at 0.657, so abstention cannot fire on bad photos. | 🔴 Open |
| 7 | `scripts/quantize_onnx.py` fixed in the repo but **not yet committed**. | 🟡 Pending |

Items 3–6 are from the main audit and need retraining, not a deployment. See
`PANDERM_DIAGNOSTIC_REPORT.md` §13 for the prioritised roadmap.

---

## 8. Repository state

Deployed but **not yet committed**:

```
M scripts/export_onnx.py       rollout: mean fusion, block 3
M scripts/quantize_onnx.py     MatMul-only + ConvInteger guard
M web/app.js                   MODEL_URL -> ?v=fc92e506
M web/lib/model.js             normalisation: p50 floor, true max, no gamma
?? audit_reports/              all audit evidence
?? scripts/audit_*.py          audit tooling
```

`web/model/panderm_vit_int8.onnx` is gitignored and therefore not tracked; the backup in §5 is the only
local copy of the previous model.

---

## 9. Evidence

```
audit_reports/
  VERCEL_UPDATE_REPORT.md            this document
  live_verification.json             Task 6, live browser results
  FINAL_DEPLOYMENT_READINESS.md      pre-deployment sign-off
  deploy_parity.json                 465-image classification parity
  heatmap_verification.json          grid, ties, clipping, alignment
  browser_test.json                  pre-deploy browser suite
  int8_accuracy.json                 int8 vs fp32 + pooling consistency
  updated_localization_metrics.json  400-image held-out validation
  rollout_comparison.png             old vs new vs random
```
