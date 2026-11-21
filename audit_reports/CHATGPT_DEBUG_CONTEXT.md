# Debug context handoff — PanDerm skin lesion classifier

Written for another AI assistant. Everything here is measured output from the running system on
2026-11-21. No figure is estimated. Companion document: `PANDERM_DIAGNOSTIC_REPORT.md`.

---

## 1. Project architecture

Browser-only skin lesion classifier. **No backend** — model, weights and all logic are static assets.

```
web/app.js            UI, camera, render, abstention display
web/index.html        single-screen app, metadata form
web/lib/pipeline.js   quality gate, preprocessing, ABCDE, risk tiers, escalation
web/lib/model.js      ONNX Runtime Web session, heatmap normalisation
web/model/            panderm_vit_int8.onnx (88 MB), head.json (3.7 MB)
web/vendor/ort/       self-hosted ONNX Runtime Web

src/backbone/panderm_wrapper.py   PanDerm loading + forward (L241)
src/backbone/fallback_backbone.py ConvNeXt-Base, used only for comparison
src/model/fusion.py               FusionHead (L83 forward)
src/model/risk_classifier.py      load_risk_classifier, RiskPrediction
src/model/train.py                feature cache, class weights, macro_f1
src/dataset.py                    group_aware_split, encode_metadata, risk_for
src/segmentation.py               lesion mask for ABCDE (not for classification)
vendor/panderm/                   committed PanDerm model definition

scripts/export_onnx.py            TrunkWrapper → ONNX + rollout; dump_head
scripts/verify_web_model.py       JS-head-vs-PyTorch parity gate
scripts/finetune_backbone.py      partial backbone fine-tuning
scripts/audit_*.py                the audit scripts that produced these numbers

configs/default.yaml              single source of truth
checkpoints/risk_head.pt          DEPLOYED head
checkpoints/panderm/panderm_bb_data6_checkpoint-499.pth   343 MB, official
checkpoints/panderm_finetuned_gpu.pt                      fine-tuned, NOT deployed
```

## 2. PanDerm configuration

```yaml
backbone:
  backend: panderm            # fails loudly; no silent fallback
  image_size: 224
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std:  [0.229, 0.224, 0.225]
  panderm:
    repo_dir: vendor/panderm
    model_name: PanDerm_Base_LP          # ViT-B/16, 86M params
    checkpoint: checkpoints/panderm/panderm_bb_data6_checkpoint-499.pth
    feature_dim: 768
```

**Verified bit-identical to the official release**: ONNX `vit.patch_embed.proj.weight` vs checkpoint
`patch_embed.proj.weight`, shape `(768,3,16,16)`, **max abs diff `0.000e+00`**. Deployed file SHA-256
matches local export (`b8a38550…`). Loader reports 2 missing keys — both the discarded pretraining head.

**Configuration: frozen linear probe.** The backbone is never updated; only the head trains.

## 3. Shapes

```
input        (1, 3, 224, 224)      range [-1.93, 1.75] after ImageNet norm
patches      224/16 = 14×14 = 196, +1 CLS = 197 tokens
ONNX out     tokens  (1, 197, 768)
             rollout (1, 196)
embedding    (1, 768)              mean over tokens if 3-D
metadata     (1, 12)               [age/100, age_missing, sex onehot×3, site onehot×7]
head         image 768→128 ; meta 12→32 ; concat 160 → [256,128] → 6
logits       (1, 6)  → softmax
```

Export parity: trunk max diff `4.63e-05`, rollout `2.61e-08`.

## 4. Labels

```python
diagnosis_classes = ["NEV", "SEK", "ACK", "BCC", "SCC", "MEL"]
risk_of = {"NEV":"Low", "SEK":"Low", "ACK":"Moderate",
           "BCC":"High", "SCC":"High", "MEL":"High"}
```
Risk tier is a **lookup**, never predicted. Escalation rules may raise it, never lower it.

## 5. Dataset

3,104 ISIC clinical photographs. Split by `lesion_id`: **2,170 / 469 / 465**.
Test distribution: NEV 95, SEK 83, ACK 93, BCC 87, **SCC 32**, MEL 75.
Train SCC = 140 → class weight **2.58×** (others ≈0.87).

Leakage: **0** shared lesion IDs, **0** shared paths, **2** near-duplicates (cosine > 0.99).

## 6. Measured results — held-out test set, n = 465

```
accuracy            0.6731     macro_roc_auc_ovr   0.9373
balanced_accuracy   0.6635     macro_pr_auc        0.7381
top3_accuracy       0.9742     cohens_kappa        0.6065
macro_f1            0.6522     matthews_corrcoef   0.6101
weighted_f1         0.6865     benign/malig AUC    0.9420
ECE (10-bin)        0.0574     mean Brier (OvR)    0.0697
overconfidence gap +0.0001     abstention rate     19.4%
accuracy|confident  0.7493     accuracy|abstained  0.3556
```

Per class (P / R / Specificity / F1 / PR-AUC / n):
```
NEV  0.778 0.663 0.951 0.716 0.835  95
SEK  0.713 0.687 0.940 0.699 0.781  83
ACK  0.899 0.667 0.981 0.765 0.894  93
BCC  0.713 0.655 0.939 0.683 0.769  87
SCC  0.281 0.562 0.894 0.375 0.373  32   <-- broken
MEL  0.615 0.747 0.910 0.675 0.777  75
```

Confusion matrix (rows = true, cols = predicted, order NEV SEK ACK BCC SCC MEL):
```
NEV [63, 10,  0,  1,  0, 21]
SEK [ 6, 57,  4,  2,  2, 12]
ACK [ 0,  6, 62,  7, 18,  0]
BCC [ 0,  0,  3, 57, 25,  2]
SCC [ 0,  1,  0, 13, 18,  0]
MEL [12,  6,  0,  0,  1, 56]
```

Errors: 152/465. **19 dangerous downgrades (High→Low, 4.1%)**, 12 of them MEL→NEV.

## 7. THE MAIN BUG — attention localization is worse than random

Scored against **expert HAM10000 segmentation masks** (DatasetNinja bitmap format), n = 300:

```
                    rollout   random
Pointing Game         0.127    0.287   <-- peak AVOIDS the lesion
IoU                   0.246    0.203
Dice                  0.374    0.308
Focus ratio           1.245    1.000
above-chance focus    46.3%
mean lesion area      30.8%
```

### Current implementation — `scripts/export_onnx.py`, `TrunkWrapper.forward`

```python
ROLLOUT_START_LAYER = 6
...
for i, blk in enumerate(vit.blocks):
    if i >= ROLLOUT_START_LAYER:
        a = _attention_matrix(blk.attn, blk.norm1(x), rel_pos_bias)
        a = a.max(dim=1).values + eye          # <-- max over heads: THE BUG
        a = a / a.sum(dim=-1, keepdim=True)
        roll = a @ roll
    x = blk(x, rel_pos_bias=rel_pos_bias)
return vit.norm(x), roll[:, 0, 1:]
```

### Diagnosis

Focus ratio 1.245 (bulk of heat on lesion) and Pointing Game 0.127 (peak off lesion) are simultaneously
true because isolated high-attention patches on background skin outrank the lesion's broad moderate
response. This is the **ViT artifact-token effect** (Darcet et al. 2023, *Vision Transformers Need
Registers*). `max` over heads propagates a single rogue head's artifact token; `mean` averages it out.

The configuration was tuned on **focus ratio alone**, which cannot see peak placement — so the selection
metric chose the worst-pointing variant.

### Measured variants, n = 200, same masks

```
variant                          Pointing    IoU   Dice  Focus  >chance
mean-fuse, block 3                  0.305  0.306  0.440  1.211    59.5%  <-- best
max-fuse b6 + 3x3 median            0.230  0.286  0.408  1.209    51.0%
mean-fuse b6 + 3x3 median           0.195  0.272  0.388  1.147    47.0%
max-fuse b6 + clip p98              0.150  0.256  0.385  1.269    50.5%
deployed: max-fuse, block 6         0.130  0.250  0.378  1.276    47.5%  <-- deployed
mean-fuse, block 9                  0.100  0.166  0.253  0.889    19.5%
mean-fuse, block 6                  0.090  0.190  0.295  1.113    36.0%
mean-fuse, block 0                  0.055  0.146  0.222  0.999    29.0%
```

### Recommended change

```python
ROLLOUT_START_LAYER = 3                  # was 6
...
        a = a.mean(dim=1) + eye          # was a.max(dim=1).values + eye
```
Then re-export ONNX, re-run `scripts/verify_web_model.py`, re-quantize, redeploy.

**Caveat to preserve:** masks are dermoscopic while the classifier trains on clinical images. The rollout
comes from the frozen trunk so the measurement is valid for the backbone, but it is not a measurement on
the clinical test split. Attention is not causal attribution.

## 8. Metadata dominance

```
policy            top-1    MEL recall
TRUE metadata     67.3%       74.7%
all UNKNOWN       56.6%       33.3%
```

Single case, ISIC_6393409 (true MEL, patient age 80, female, leg):
```
true (80, female, leg)       MEL 82.2%  -> High risk
form defaults (45, -, trunk) MEL  7.1%  -> Low risk      <-- was shipping
honest unknown               MEL 43.6%  -> High risk
```

Fixed earlier the same day: the form no longer pre-fills age/site, and `Number('') === 0` (a valid age of
zero) is now passed as `NaN` so the missing-age flag fires. **The underlying model dependence remains** —
recommended fix is retraining with metadata dropout (p≈0.3).

## 9. Robustness, n = 465

```
perturbation          acc    delta  MEL rec  MEL->Low  conf
none (baseline)     0.673        —    0.747     24.0%  0.673
horizontal flip     0.686   +0.013    0.760     22.7%  0.673
rotate 90 deg       0.671   -0.002    0.747     24.0%  0.671
brightness x0.6     0.654   -0.019    0.720     26.7%  0.646
brightness x1.4     0.634   -0.039    0.627     20.0%  0.651
contrast x0.5       0.634   -0.039    0.560     42.7%  0.624
rotate 10 deg       0.626   -0.047    0.507     45.3%  0.644
downscale 64px      0.488   -0.185    0.520     48.0%  0.647
jpeg quality 20     0.458   -0.215    0.387     60.0%  0.664
greyscale           0.430   -0.243    0.693     29.3%  0.614
gaussian blur r=2   0.404   -0.269    0.347     65.3%  0.657
```

**Confidence is flat while accuracy collapses.** Blur: accuracy 0.404, confidence 0.657. The abstention
gate depends on max-softmax and therefore cannot fire on degraded input. Calibration measured on clean
data (ECE 0.0574) does not survive perturbation.

## 10. Backbone justification

```
setting                              PanDerm   ConvNeXt-Base   gap
frozen (5 seeds)                       69.3%          69.2%   +0.1  (noise)
fine-tuned (28.4M vs 29.6M trainable)  71.2%          66.7%   +4.5
```
Fine-tuning helps PanDerm and hurts ConvNeXt. Do not recommend replacing PanDerm.

Other measured negatives:
- Crop to segmented lesion before classification: macro-F1 0.673 → **0.625** (worse).
- Dermoscopic vs clinical, trained within domain: 69.2% vs 70.2% — **no domain advantage**.

## 11. Environment

```
Windows 11 · RTX 4050 Laptop 6.4 GB · AMD Radeon 740M (unused) · driver 573.05
torch 2.12.1+cu126 in C:\Users\chint\gpuenv   (system python has 2.12.1+cpu)
single image 9.48 ms · batch-32 149.1 img/s · peak VRAM 598.7 MB
Production runs int8 ONNX under WASM in a Web Worker; the GPU is not used in production.
```

## 12. Bug list (abbreviated — full table in the diagnostic report)

| ID | Sev | Issue |
|---|---|---|
| BUG-01 | CRITICAL | `max`-fusion rollout: Pointing 0.127 vs 0.287 random. Fix: `mean`, block 3. |
| BUG-02 | CRITICAL | Rollout tuned on focus ratio only — metric blind to peak placement. |
| BUG-03 | HIGH | SCC precision 0.281; 140 train images; class weight 2.58× makes it a sink. |
| BUG-04 | HIGH | 19 dangerous downgrades (4.1%), 12 MEL→NEV. |
| BUG-05 | HIGH | Metadata dominance: all-unknown drops MEL recall 74.7%→33.3%. |
| BUG-06 | HIGH | Confidence flat under degradation; abstention cannot fire. |
| BUG-07 | MED | No EXIF orientation handling; 10° rotation halves MEL recall. |
| BUG-08 | MED | No rotation augmentation (cached embeddings make it impossible). |
| BUG-09 | MED | Heatmap p99 clipping hides artifact spikes rather than removing them. |

## 13. Files

```
audit_reports/
  PANDERM_DIAGNOSTIC_REPORT.md   CHATGPT_DEBUG_CONTEXT.md
  evaluation_metrics.json        classification_report.csv
  localization_metrics.json      localization_per_image.csv
  rollout_variants.json          robustness_results.json
  misclassified.csv              confusion_matrix.png   calibration.png
  gradcam_examples/rollout_vs_mask.png
  misclassified_examples/dangerous_downgrades.png
```

---

Analyze this diagnostic report as a senior computer vision and medical AI engineer. Identify the root
causes of incorrect classification and Grad-CAM localization. Review the evidence and suggest precise
code-level fixes. Prioritize model correctness, preprocessing consistency, attribution validity, data
integrity, and reliable evaluation. Do not recommend replacing PanDerm unless the evidence demonstrates
that the existing implementation cannot meet the requirements.
