"""Full evaluation of the DEPLOYED model for the technical audit.

Everything here runs against the frozen head that is actually in production
(checkpoints/risk_head.pt), on the lesion-grouped held-out test split. No
figure in the audit report comes from anywhere else.

Produces, under audit_reports/:
  evaluation_metrics.json    every metric, with sample counts
  classification_report.csv  per-class precision/recall/F1/support
  confusion_matrix.png       counts and row-normalised
  calibration.png            reliability curve + confidence histogram
  misclassified.csv          every error, with confidence and category
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, brier_score_loss, cohen_kappa_score,
    confusion_matrix, matthews_corrcoef, roc_auc_score,
)

from src.config import load_config  # noqa: E402
from src.dataset import (  # noqa: E402
    TARGET_COLUMN, build_metadata_matrix, diagnosis_classes,
    group_aware_split, load_unified_csv, risk_for,
)
from src.model.risk_classifier import load_risk_classifier  # noqa: E402
from src.utils import get_logger, load_image_rgb  # noqa: E402

logger = get_logger("audit")
OUT = Path("audit_reports")


def expected_calibration_error(conf, correct, bins=10):
    """ECE with equal-width bins, plus the per-bin data for the curve."""
    edges = np.linspace(0, 1, bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not m.sum():
            rows.append((float((lo + hi) / 2), None, None, 0))
            continue
        acc, avg = float(correct[m].mean()), float(conf[m].mean())
        ece += (m.sum() / len(conf)) * abs(acc - avg)
        rows.append((float((lo + hi) / 2), acc, avg, int(m.sum())))
    return float(ece), rows


def main() -> int:
    cfg = load_config()
    classes = diagnosis_classes(cfg)
    n_cls = len(classes)
    c2i = {c: i for i, c in enumerate(classes)}

    df = load_unified_csv(cfg)
    tr, va, te = group_aware_split(df, cfg)
    logger.info("split train=%d val=%d test=%d", len(tr), len(va), len(te))

    model = load_risk_classifier(cfg=cfg, strict=True).eval()
    device = next(model.head.parameters()).device
    tf = model.backbone.build_transform(train=False)

    # ---- leakage checks before any metric is believed ------------------------
    tr_ids = set(tr.get("lesion_id", pd.Series(dtype=object)).dropna())
    te_ids = set(te.get("lesion_id", pd.Series(dtype=object)).dropna())
    shared_lesions = sorted(tr_ids & te_ids)
    shared_paths = sorted(set(tr.image_path) & set(te.image_path))

    # ---- inference -----------------------------------------------------------
    X = torch.stack([tf(Image.fromarray(load_image_rgb(cfg.resolve(p))))
                     for p in te.image_path])
    M = torch.from_numpy(build_metadata_matrix(te, cfg).astype(np.float32))
    feats, probs = [], []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            f = model.backbone(X[i:i + 32].to(device)).float()
            feats.append(f.cpu().numpy())
            probs.append(torch.softmax(model.head(f, M[i:i + 32].to(device)),
                                       1).cpu().numpy())
    F_test = np.concatenate(feats)
    P = np.concatenate(probs)
    y = np.array([c2i[v] for v in te[TARGET_COLUMN]])
    pred = P.argmax(1)
    conf = P.max(1)
    order = np.argsort(-P, 1)

    # ---- near-duplicate check in embedding space -----------------------------
    Ftr = []
    with torch.no_grad():
        Xtr = [tf(Image.fromarray(load_image_rgb(cfg.resolve(p)))) for p in tr.image_path]
        for i in range(0, len(Xtr), 32):
            Ftr.append(model.backbone(torch.stack(Xtr[i:i + 32]).to(device))
                       .float().cpu().numpy())
    Ftr = np.concatenate(Ftr)
    A = F_test / np.clip(np.linalg.norm(F_test, axis=1, keepdims=True), 1e-8, None)
    B = Ftr / np.clip(np.linalg.norm(Ftr, axis=1, keepdims=True), 1e-8, None)
    sim = A @ B.T
    nn_sim = sim.max(1)
    near_dupes = int((nn_sim > 0.99).sum())

    # ---- metrics -------------------------------------------------------------
    cm = confusion_matrix(y, pred, labels=range(n_cls))
    per_class = {}
    for i, c in enumerate(classes):
        tp = int(cm[i, i]); fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp); tn = int(cm.sum() - tp - fn - fp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class[c] = {
            "support": int((y == i).sum()),
            "precision": prec, "recall_sensitivity": rec,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
            "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
            "roc_auc_ovr": float(roc_auc_score((y == i).astype(int), P[:, i])),
            "pr_auc": float(average_precision_score((y == i).astype(int), P[:, i])),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    correct = (pred == y).astype(float)
    ece, cal_rows = expected_calibration_error(conf, correct)
    brier = float(np.mean([
        brier_score_loss((y == i).astype(int), P[:, i]) for i in range(n_cls)]))

    MAL = {"MEL", "BCC", "SCC", "ACK"}
    mi = [i for i, c in enumerate(classes) if c in MAL]
    pm = P[:, mi].sum(1)
    ym = np.isin(y, mi).astype(int)

    metrics = {
        "model_evaluated": "frozen PanDerm_Base_LP + FusionHead (checkpoints/risk_head.pt) "
                           "- the deployed configuration",
        "split": {"train": len(tr), "val": len(va), "test": len(te),
                  "grouping": "by lesion_id"},
        "leakage_checks": {
            "lesion_ids_shared_train_test": len(shared_lesions),
            "identical_image_paths_shared": len(shared_paths),
            "test_images_with_train_cosine_gt_0.99": near_dupes,
            "max_test_to_train_cosine": float(nn_sim.max()),
            "mean_test_to_train_cosine": float(nn_sim.mean()),
        },
        "class_distribution_test": {c: int((y == i).sum()) for i, c in enumerate(classes)},
        "overall": {
            "n_samples": int(len(y)),
            "accuracy": float((pred == y).mean()),
            "balanced_accuracy": float(np.mean([
                (pred[y == i] == i).mean() for i in range(n_cls) if (y == i).sum()])),
            "top3_accuracy": float(np.mean([y[i] in order[i, :3] for i in range(len(y))])),
            "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
            "weighted_f1": float(np.average([v["f1"] for v in per_class.values()],
                                            weights=[v["support"] for v in per_class.values()])),
            "macro_precision": float(np.mean([v["precision"] for v in per_class.values()])),
            "macro_recall": float(np.mean([v["recall_sensitivity"] for v in per_class.values()])),
            "macro_roc_auc_ovr": float(roc_auc_score(y, P, multi_class="ovr", average="macro")),
            "macro_pr_auc": float(np.mean([v["pr_auc"] for v in per_class.values()])),
            "cohens_kappa": float(cohen_kappa_score(y, pred)),
            "matthews_corrcoef": float(matthews_corrcoef(y, pred)),
        },
        "binary_benign_vs_malignant": {
            "definition": "malignant/premalignant = MEL, BCC, SCC, ACK",
            "roc_auc": float(roc_auc_score(ym, pm)),
            "pr_auc": float(average_precision_score(ym, pm)),
            "accuracy_at_0.5": float(np.mean((pm > 0.5) == ym)),
        },
        "calibration": {
            "expected_calibration_error_10bin": ece,
            "mean_brier_score_ovr": brier,
            "mean_confidence": float(conf.mean()),
            "accuracy": float(correct.mean()),
            "overconfidence_gap": float(conf.mean() - correct.mean()),
            "bins": [{"bin_centre": b, "accuracy": a, "mean_confidence": m, "n": n}
                     for b, a, m, n in cal_rows],
        },
        "abstention": {
            "threshold": 0.5,
            "rate": float(np.mean(conf < 0.5)),
            "accuracy_when_confident": float(correct[conf >= 0.5].mean()),
            "accuracy_when_abstaining": float(correct[conf < 0.5].mean())
            if (conf < 0.5).sum() else None,
        },
        "per_class": per_class,
        "confusion_matrix": {"labels": classes, "counts": cm.tolist()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2),
                                                 encoding="utf-8")

    pd.DataFrame(per_class).T.to_csv(OUT / "classification_report.csv")

    # ---- confusion matrix figure --------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (mat, ttl, fmt) in zip(axes, [
            (cm, "Counts", "d"),
            (cm / np.clip(cm.sum(1, keepdims=True), 1, None), "Row-normalised (recall)", ".2f")]):
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(n_cls)); ax.set_xticklabels(classes, rotation=45)
        ax.set_yticks(range(n_cls)); ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(ttl)
        hi = mat.max()
        for a in range(n_cls):
            for b in range(n_cls):
                ax.text(b, a, format(mat[a, b], fmt), ha="center", va="center",
                        fontsize=8, color="white" if mat[a, b] > hi * 0.55 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Confusion matrix - held-out test set (n={len(y)})")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", dpi=140); plt.close(fig)

    # ---- calibration figure --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bc = [r[0] for r in cal_rows if r[1] is not None]
    ba = [r[1] for r in cal_rows if r[1] is not None]
    bm = [r[2] for r in cal_rows if r[1] is not None]
    axes[0].plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    axes[0].plot(bm, ba, "o-", color="#0b6e7a", label="model")
    axes[0].set_xlabel("Mean confidence"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"Reliability curve (ECE={ece:.3f})")
    axes[0].legend(); axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[1].hist(conf, bins=20, color="#0b6e7a", alpha=.85)
    axes[1].axvline(0.5, color="#a62a22", ls="--", label="abstention threshold")
    axes[1].set_xlabel("Top-class confidence"); axes[1].set_ylabel("Images")
    axes[1].set_title("Confidence distribution"); axes[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "calibration.png", dpi=140); plt.close(fig)

    # ---- misclassification table --------------------------------------------
    rows = []
    for i in range(len(y)):
        if pred[i] == y[i]:
            continue
        t, p = classes[y[i]], classes[pred[i]]
        tr_risk, pr_risk = risk_for(cfg, t), risk_for(cfg, p)
        if tr_risk == "High" and pr_risk == "Low":
            cat = "CRITICAL: dangerous downgrade"
        elif tr_risk in ("High", "Moderate") and pr_risk == "Low":
            cat = "downgraded to Low"
        elif tr_risk == pr_risk:
            cat = "same risk tier (advice still correct)"
        else:
            cat = "upgraded (errs safe)"
        rows.append({
            "image_id": Path(str(te.iloc[i].image_path)).stem,
            "true": t, "predicted": p, "confidence": round(float(conf[i]), 4),
            "true_prob": round(float(P[i, y[i]]), 4),
            "true_risk": tr_risk, "predicted_risk": pr_risk,
            "abstained": bool(conf[i] < 0.5), "category": cat,
            "true_in_top3": bool(y[i] in order[i, :3]),
        })
    mis = pd.DataFrame(rows).sort_values("confidence", ascending=False)
    mis.to_csv(OUT / "misclassified.csv", index=False)

    print(f"\n{'=' * 70}")
    print(f"  DEPLOYED MODEL - held-out test set, n={len(y)}")
    print(f"{'=' * 70}")
    o = metrics["overall"]
    for k in ("accuracy", "balanced_accuracy", "top3_accuracy", "macro_f1",
              "weighted_f1", "macro_roc_auc_ovr", "macro_pr_auc",
              "cohens_kappa", "matthews_corrcoef"):
        print(f"  {k:24s} {o[k]:.4f}")
    print(f"  {'ECE (10-bin)':24s} {ece:.4f}")
    print(f"  {'mean Brier (OvR)':24s} {brier:.4f}")
    print(f"  {'overconfidence gap':24s} {metrics['calibration']['overconfidence_gap']:+.4f}")
    print(f"\n  leakage: {len(shared_lesions)} shared lesion_ids, "
          f"{len(shared_paths)} shared paths, {near_dupes} near-dupes (cos>0.99)")
    print(f"  misclassified: {len(mis)} / {len(y)}")
    print(f"    dangerous downgrades (High->Low): "
          f"{int((mis.category == 'CRITICAL: dangerous downgrade').sum())}")
    print(f"    same-tier errors (advice ok):     "
          f"{int((mis.category == 'same risk tier (advice still correct)').sum())}")
    print(f"\n  wrote {OUT}/evaluation_metrics.json, classification_report.csv,")
    print(f"        confusion_matrix.png, calibration.png, misclassified.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
