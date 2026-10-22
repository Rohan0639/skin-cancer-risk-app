"""Evaluate the trained risk head on the held-out test split.

Every number this script prints is measured, never assumed. It scores the exact
split recorded in ``outputs/splits.csv`` during training, so the test images
were never seen by the head.

Outputs (paths configurable under ``evaluate:``):

* ``outputs/metrics.json``          -- all metrics plus full run provenance
* ``outputs/per_class_metrics.csv`` -- precision/recall/F1/support per class
* ``outputs/confusion_matrix.png``  -- counts and row-normalised rates
* ``outputs/roc_curves.png``        -- one-vs-rest ROC curve per class

The JSON records the backbone used and the data sources, so a run on the
synthetic demo data can never be mistaken for a real result.

Usage::

    python src/evaluate.py
    python src/evaluate.py --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from src.config import Config, load_config  # noqa: E402
from src.dataset import build_metadata_matrix, load_unified_csv  # noqa: E402
from src.model.risk_classifier import load_risk_classifier  # noqa: E402
from src.model.train import SPLIT_FILENAME, extract_features  # noqa: E402
from src.utils import ensure_dir, get_logger, resolve_device, set_seed  # noqa: E402

logger = get_logger("evaluate")


def load_split(cfg: Config, split: str) -> pd.DataFrame:
    """Return the rows belonging to ``split``, joined back to the unified CSV.

    Args:
        cfg: Loaded project config.
        split: ``"train"``, ``"val"`` or ``"test"``.

    Returns:
        A unified-schema dataframe restricted to that split.

    Raises:
        FileNotFoundError: If training has not been run, so no split exists.
        ValueError: If the split is empty.
    """
    split_path = cfg.path("paths.output_dir") / SPLIT_FILENAME
    if not split_path.exists():
        raise FileNotFoundError(
            f"No split file at {split_path}. Run: python src/model/train.py"
        )

    assignments = pd.read_csv(split_path)
    wanted = set(assignments.loc[assignments["split"] == split, "image_path"])
    if not wanted:
        raise ValueError(f"Split {split!r} is empty in {split_path}.")

    df = load_unified_csv(cfg)
    subset = df[df["image_path"].isin(wanted)].reset_index(drop=True)
    if subset.empty:
        raise ValueError(
            f"None of the {split} images are present in the unified CSV. "
            "Re-run data preparation and training."
        )
    logger.info("Evaluating on the %s split: %d images", split, len(subset))
    return subset


def predict_split(
    model,
    df: pd.DataFrame,
    cfg: Config,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over ``df``.

    Returns:
        ``(y_true, probabilities)`` where ``y_true`` holds class indices and
        ``probabilities`` has shape ``(N, n_classes)``.
    """
    classes = list(model.classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    features = extract_features(df, model.backbone, cfg, device)
    metadata = build_metadata_matrix(df, cfg)

    model.eval()
    with torch.no_grad():
        logits = model.forward_from_features(
            torch.from_numpy(features).to(device),
            torch.from_numpy(metadata).to(device),
        )
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    y_true = np.array([class_to_idx[r] for r in df["risk_label"]], dtype=np.int64)
    return y_true, probabilities


def compute_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: List[str],
) -> Dict[str, Any]:
    """Compute the full metric suite.

    ROC-AUC is one-vs-rest. It is reported per class and macro-averaged, and is
    omitted (``None``) for any class absent from the split, since AUC is
    undefined without both positives and negatives.

    Returns:
        A JSON-serialisable dict of metrics.
    """
    y_pred = probabilities.argmax(axis=1)
    n_classes = len(classes)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )

    per_class_auc: Dict[str, Optional[float]] = {}
    for idx, name in enumerate(classes):
        binary_truth = (y_true == idx).astype(int)
        if binary_truth.min() == binary_truth.max():
            per_class_auc[name] = None  # only one class present
            continue
        per_class_auc[name] = float(roc_auc_score(binary_truth, probabilities[:, idx]))

    present = sorted(set(y_true.tolist()))
    macro_auc: Optional[float] = None
    if len(present) > 1:
        try:
            macro_auc = float(roc_auc_score(
                y_true, probabilities, multi_class="ovr", average="macro",
                labels=list(range(n_classes)),
            ))
        except ValueError as exc:
            logger.warning("Macro ROC-AUC unavailable: %s", exc)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), average="weighted", zero_division=0)

    # Clinically the costliest error is calling a High-risk lesion Low, so
    # report that rate explicitly rather than burying it in the matrix.
    high_idx = classes.index("High") if "High" in classes else None
    low_idx = classes.index("Low") if "Low" in classes else None
    high_missed_as_low: Optional[float] = None
    if high_idx is not None and low_idx is not None:
        n_high = int((y_true == high_idx).sum())
        if n_high > 0:
            high_missed_as_low = float(
                ((y_true == high_idx) & (y_pred == low_idx)).sum() / n_high
            )

    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro": {"precision": float(macro_p), "recall": float(macro_r),
                  "f1": float(macro_f1), "roc_auc_ovr": macro_auc},
        "weighted": {"precision": float(weighted_p), "recall": float(weighted_r),
                     "f1": float(weighted_f1)},
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "roc_auc_ovr": per_class_auc[name],
            }
            for i, name in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(n_classes))).tolist(),
        "confusion_matrix_labels": classes,
        "high_risk_missed_as_low_rate": high_missed_as_low,
        "classification_report": classification_report(
            y_true, y_pred, labels=list(range(n_classes)), target_names=classes,
            zero_division=0),
    }


def plot_confusion_matrix(matrix: np.ndarray, classes: List[str], out_path: Path) -> Path:
    """Plot raw counts alongside row-normalised recall rates."""
    ensure_dir(out_path.parent)
    counts = np.asarray(matrix, dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    normalised = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, fmt in (
        (axes[0], counts, "Confusion matrix (counts)", "{:.0f}"),
        (axes[1], normalised, "Row-normalised (recall per true class)", "{:.2f}"),
    ):
        image = ax.imshow(data, cmap="Blues", vmin=0,
                          vmax=data.max() if data.max() > 0 else 1)
        ax.set_title(title)
        ax.set_xlabel("Predicted risk")
        ax.set_ylabel("True risk")
        ax.set_xticks(range(len(classes)), classes, rotation=20)
        ax.set_yticks(range(len(classes)), classes)
        threshold = data.max() / 2 if data.max() > 0 else 0.5
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                        color="white" if data[i, j] > threshold else "black")
        fig.colorbar(image, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote confusion matrix -> %s", out_path)
    return out_path


def plot_roc_curves(y_true: np.ndarray, probabilities: np.ndarray,
                    classes: List[str], out_path: Path) -> Optional[Path]:
    """Plot one-vs-rest ROC curves, skipping classes absent from the split."""
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    plotted = 0

    for idx, name in enumerate(classes):
        binary_truth = (y_true == idx).astype(int)
        if binary_truth.min() == binary_truth.max():
            continue
        fpr, tpr, _ = roc_curve(binary_truth, probabilities[:, idx])
        auc = roc_auc_score(binary_truth, probabilities[:, idx])
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        logger.warning("No ROC curves plotted -- only one class present in the split.")
        return None

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote ROC curves -> %s", out_path)
    return out_path


def print_summary(metrics: Dict[str, Any], provenance: Dict[str, Any]) -> None:
    """Print a readable console summary of the evaluation."""
    print()
    print("=" * 72)
    print("  EVALUATION RESULTS")
    print("=" * 72)
    print(f"  split            : {provenance['split']}  ({metrics['n_samples']} images)")
    print(f"  backbone         : {provenance['backbone']}")
    print(f"  data sources     : {', '.join(provenance['sources'])}")
    if provenance.get("synthetic_data"):
        print()
        print("  *** SYNTHETIC DEMO DATA -- these numbers verify that the code")
        print("  *** runs. They are NOT a measure of clinical performance and")
        print("  *** must not be reported as one.")
    print("-" * 72)
    print(f"  accuracy          : {metrics['accuracy']:.4f}")
    print(f"  balanced accuracy : {metrics['balanced_accuracy']:.4f}")
    macro = metrics["macro"]
    auc_text = f"{macro['roc_auc_ovr']:.4f}" if macro["roc_auc_ovr"] is not None else "n/a"
    print(f"  macro precision   : {macro['precision']:.4f}")
    print(f"  macro recall      : {macro['recall']:.4f}")
    print(f"  macro F1          : {macro['f1']:.4f}")
    print(f"  macro ROC-AUC(ovr): {auc_text}")
    if metrics["high_risk_missed_as_low_rate"] is not None:
        print(f"  High called Low   : {metrics['high_risk_missed_as_low_rate']:.4f}  "
              "(the costliest error)")
    print("-" * 72)
    print(metrics["classification_report"])
    print("=" * 72)
    print()


def run_evaluation(cfg: Config, split: str = "test",
                   checkpoint: Optional[Path] = None) -> Dict[str, Any]:
    """Evaluate the trained model and write all artefacts.

    Returns:
        The metrics dict, as written to ``outputs/metrics.json``.
    """
    set_seed(int(cfg.get("project.seed", 42)))
    device = resolve_device(cfg.get("project.device", "auto"))

    df = load_split(cfg, split)
    model = load_risk_classifier(checkpoint, cfg=cfg, device=device, strict=True)
    classes = list(model.classes)

    y_true, probabilities = predict_split(model, df, cfg, device)
    metrics = compute_metrics(y_true, probabilities, classes)

    sources = sorted(df["source"].unique())
    provenance = {
        "split": split,
        "backbone": model.backbone.describe(),
        "backbone_is_fallback": bool(model.backbone.is_fallback),
        "sources": sources,
        "synthetic_data": any(s == "synthetic_demo" for s in sources),
        "checkpoint": str(checkpoint) if checkpoint else str(
            cfg.path("paths.checkpoint_dir")
            / str(cfg.get("train.checkpoint_name", "risk_head.pt"))),
        "class_order": classes,
    }
    metrics["provenance"] = provenance
    if provenance["synthetic_data"]:
        metrics["WARNING"] = (
            "Measured on SYNTHETIC demo images. These numbers verify the code "
            "path only and are NOT clinical performance."
        )

    ensure_dir(cfg.path("paths.output_dir"))
    json_path = cfg.path("evaluate.metrics_json")
    ensure_dir(json_path.parent)
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Wrote metrics -> %s", json_path)

    rows = []
    for name, values in metrics["per_class"].items():
        rows.append({"class": name, **values})
    csv_path = cfg.path("evaluate.metrics_csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Wrote per-class metrics -> %s", csv_path)

    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), classes,
                          cfg.path("evaluate.confusion_matrix_png"))
    plot_roc_curves(y_true, probabilities, classes, cfg.path("evaluate.roc_curve_png"))

    print_summary(metrics, provenance)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to a YAML config.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Which split to score (default: test).")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint .pt to load.")
    parser.add_argument("--backend", choices=["panderm", "fallback", "auto"], default=None,
                        help="Override backbone.backend.")
    args = parser.parse_args()

    overrides = {"backbone.backend": args.backend} if args.backend else None
    cfg = load_config(args.config, overrides=overrides)

    try:
        run_evaluation(cfg, split=args.split,
                       checkpoint=Path(args.checkpoint) if args.checkpoint else None)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
