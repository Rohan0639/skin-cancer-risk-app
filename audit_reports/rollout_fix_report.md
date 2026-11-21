# Attention rollout fix — implementation and validation

**Date:** 2026-11-21
**Change:** attention rollout switched from max-fusion/block-6 to mean-fusion/block-3
**Status:** ⚠️ **implemented and validated, NOT deployed.** The live site still runs the old configuration.

---

## 1. Summary

| | Result |
|---|---|
| Classification path | **bit-identical** — tokens and embeddings differ by `0.000e+00` |
| Localization, held-out | Pointing **0.128 → 0.273**, IoU **0.257 → 0.325**, Dice **0.387 → 0.462** |
| Statistical confidence | 100% of 2,000 paired bootstraps favour the new config on all three metrics |
| **Remaining limitation** | **Pointing 0.273 is still marginally below the 0.297 random baseline** |
| Side findings | 2 further bugs found while implementing (§4, §5) |

The fix is a large, statistically certain improvement. It does **not** make the heatmap good — see §7.

---

## 2. Code changes

Three files. No model weights, no preprocessing, no classification logic touched.

### `scripts/export_onnx.py`

```python
# ROLLOUT_START_LAYER = 6   →   ROLLOUT_START_LAYER = 3
# a = a.max(dim=1).values + eye   →   a = a.mean(dim=1) + eye
```

`max` over heads propagates whichever head spikes on a ViT artifact token — isolated background patches
the transformer uses as global scratch space (Darcet et al. 2023, *Vision Transformers Need Registers*),
which carry very high attention and no local signal. `mean` averages them out. Block 3 restores the
earlier spatial blocks that block 6 was skipping.

The constant's comment block, which previously justified the old values on focus ratio, was rewritten to
record why that metric was the wrong selector.

### `web/lib/model.js`

Removed the p99 upper clip and the 1.2 gamma from the heatmap normalisation (see §5).

### `scripts/audit_rollout_fix_verify.py` (new)

Quantizes the re-exported trunk and compares it against the deployed model.

---

## 3. Numerical parity — classification unchanged

The rollout is a side branch: `x = blk(...)` is untouched, so tokens should be unchanged. Verified rather
than assumed, on **40 held-out test images**, old deployed int8 vs new int8:

| Quantity | Max abs diff |
|---|---|
| `tokens` (1, 197, 768) | **0.000e+00** |
| Pooled embedding (1, 768) | **0.000e+00** |
| `rollout` (1, 196), normalised | 8.234e-01 *(expected — this is the fixed branch)* |

**Bit-identical.** Every logit, probability and prediction the app produces is unchanged.

### Export and parity gate

| Check | Result |
|---|---|
| ONNX trunk vs PyTorch | 5.10e-05 |
| ONNX rollout vs PyTorch | 1.49e-08 |
| `verify_web_model.py` fp32 | **30/30 agree**, max prob drift 0.00000 |
| `verify_web_model.py` int8 | 29/30 agree, max prob drift 0.10798 |

Identical to the pre-fix baseline, as expected given the classification path did not move.

### Quantization

| | Deployed (old) | New |
|---|---|---|
| `Conv` (fp32) | 1 | 1 |
| `MatMulInteger` | 48 | 48 |
| `ConvInteger` | 0 | 0 |
| Size | 84.5 MB | 85 MB |

---

## 4. ⚠️ Second bug found: `scripts/quantize_onnx.py` does not reproduce the deployed artifact

The committed quantization script uses `nodes_to_exclude` on head nodes. Running it on this trunk produced:

- `ConvInteger: 1` — **ONNX Runtime Web has no kernel for this op.** The model would load in the browser
  and then fail at inference.
- Token drift of **7.2** and embedding drift of **0.18** against the deployed model, which moves
  predictions.

The deployed artifact was in fact built with `op_types_to_quantize=["MatMul"]`, which leaves the
`patch_embed` Conv in fp32 — and is why that layer is still bit-identical to PanDerm's release.

**The committed script cannot rebuild the deployed model.** `audit_rollout_fix_verify.py` uses the correct
setting; `scripts/quantize_onnx.py` should be corrected or removed before anyone relies on it.

This was caught only because the parity check compared against the deployed artifact rather than trusting
the pipeline. It had been latent since the original export.

---

## 5. ⚠️ Third finding: the display normalisation was hiding the fix

`web/lib/model.js` normalised the map with a p50 floor **and a p99 ceiling**, then a 1.2 gamma. The p99
clip was added to tame the old configuration's artifact spikes. It flattens everything above p99 to
exactly 1.0, so the brightest point becomes a tie broken arbitrarily by array order.

Measured on 150 held-out masked images, new configuration:

| Normalisation | Pointing | IoU | Dice | Focus |
|---|---|---|---|---|
| p50–p99, γ1.2 *(previous)* | 0.220 | 0.330 | 0.465 | 2.067 |
| p50–p100, γ1.2 | 0.267 | 0.333 | 0.468 | 2.042 |
| **p50–p100, γ1.0** *(adopted)* | **0.273** | 0.332 | 0.467 | 1.953 |
| p30–p100, γ1.0 | 0.273 | 0.330 | 0.467 | 1.783 |

The old clip was costing **0.047 pointing accuracy** on the new map. Adopted: p50 floor, true max ceiling,
no gamma — which matches the unnormalised map exactly, so the display now loses nothing.

**A further warning about focus ratio:** the p50 floor inflates it from ~1.22 to ~2.07 by zeroing the
bottom half of the map. Any focus-ratio figure measured *after* display normalisation is inflated. This is
a second, independent way that metric misled.

---

## 6. Validation on a held-out set

The variant sweep chose mean/block-3 by scoring eight configurations on 200 images. Reporting that 0.305
as the result would be reporting a **selection score**: with eight candidates, the winner is partly lucky.

**Design:** 2,570 dermoscopic images carry expert masks. All 300 images used during selection were
excluded. 400 were sampled from the remaining 2,270 (`random_state=99`), disjoint from selection.

| Metric | Old (max/b6) | New (mean/b3) | Random | Δ |
|---|---|---|---|---|
| **Pointing Game** | 0.128 | **0.273** | **0.297** | **+0.145** |
| **IoU** | 0.257 | **0.325** | 0.214 | **+0.068** |
| **Dice** | 0.387 | **0.462** | 0.321 | **+0.076** |
| Focus ratio | 1.236 | 1.198 | 1.000 | −0.038 |
| Above-chance focus | 47.8% | 57.8% | — | +10.0pp |

### Paired bootstrap (2,000 resamples, new − old)

| Metric | Δ | 95% CI | Bootstraps favouring new |
|---|---|---|---|
| Pointing Game | +0.145 | [+0.098, +0.193] | **100.0%** |
| IoU | +0.068 | [+0.059, +0.076] | **100.0%** |
| Dice | +0.076 | [+0.067, +0.085] | **100.0%** |

Every confidence interval excludes zero. **77% of individual images improved on IoU.**

Note the focus ratio **fell** (1.236 → 1.198) while all three real metrics rose — direct confirmation that
focus ratio is anti-correlated with localization quality here, and that selecting on it was the original
error.

Per-image results: `rollout_validation_per_image.csv`. Figure: `rollout_comparison.png`.

---

## 7. What this does **not** fix

**Pointing Game 0.273 remains below the 0.297 random baseline.** The selection-set figure of 0.305
appeared to clear chance; on held-out data it does not. That gap is ordinary selection optimism, and it is
the reason a separate validation set was required.

So, stated plainly:

- **Region overlap is now genuinely better than chance.** IoU 0.325 vs 0.214 and Dice 0.462 vs 0.321 are
  real, large margins.
- **Peak placement is still about chance.** The hottest single pixel is no more likely to be on the lesion
  than a random pixel. The heatmap is useful as a rough region indicator and should not be read as "the
  model looked *here*".

The honest description of the current state is: *the explanation went from actively misleading to roughly
uninformative-but-not-wrong at its peak, with meaningfully correct region coverage.*

### Further limitations

1. **Domain.** Masks are dermoscopic HAM10000; the deployed classifier trains on clinical photographs. The
   rollout comes from the frozen trunk, which is shared, so this validly measures backbone attention — but
   it is not measured on the clinical test split, because no clinical masks exist.
2. **Attention is not causal attribution.** A high rollout value says the CLS token's attention reaches
   that patch. It does not establish that the patch drove the prediction. No claim in this report should
   be read as a causal explanation of the model's output.
3. **Pointing Game's random baseline equals mean lesion area** (0.297 ≈ 0.321 here). It is only meaningful
   read against that number, never on its own.
4. **Artifact tokens are reduced, not removed.** Mean fusion dilutes them; it does not eliminate them.
   Approaches that would: register tokens (requires retraining the backbone), or a median filter on the
   14×14 grid (measured Pointing 0.230 in the sweep — worse than mean/b3, so not adopted).

---

## 8. Deployment — not performed

As instructed, nothing was deployed. Current state:

| Artifact | State |
|---|---|
| Live site | **old configuration still serving** |
| `web/model/panderm_vit_int8.onnx` | unchanged (old rollout) |
| `web/lib/model.js` | **updated** (normalisation fix) — not deployed |
| `scripts/export_onnx.py` | **updated** (rollout fix) — not deployed |
| New int8 model | staged at `<scratch>/stage/model/panderm_vit_int8.onnx` (85 MB) |
| All original checkpoints | preserved and untouched |

### To deploy

1. Copy the staged `panderm_vit_int8.onnx` into `web/model/`.
2. Update the content hash in `web/app.js` (`MODEL_URL = './model/panderm_vit_int8.onnx?v=<hash>'`) —
   the files are served immutable, so without this browsers keep the old model forever.
3. Drive the app locally and confirm the heatmap renders and aligns.
4. Deploy, then verify on the live URL.

**One caution:** `web/lib/model.js` and `scripts/export_onnx.py` are now updated in the repository while
the deployed model still carries the old rollout. The new normalisation on the old map is untested
together — the p99 clip was there specifically to tame the old spikes. **Deploy the model and the JS
together, or revert the JS until you do.**

---

## 9. Files

```
audit_reports/
  rollout_fix_report.md                 this document
  rollout_comparison.png                old vs new vs random, focus histogram, per-image change
  updated_localization_metrics.json     validation, normalisation sweep, parity
  rollout_validation_per_image.csv      400 held-out images, both configs
  rollout_parity.json                   old vs new int8 comparison
  rollout_variants.json                 original 8-way selection sweep
```
