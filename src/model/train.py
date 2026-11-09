"""Train the risk-classification head.

The backbone is frozen, so by default this script embeds every image **once**,
caches the embeddings to an ``.npz``, and then trains the small fusion MLP on
those cached vectors. On a CPU laptop that turns a multi-hour job into a
one-off extraction pass followed by seconds of training.

Set ``train.cache_features: false`` to run the slower path that re-embeds every
image each epoch with augmentation applied -- worth it on a GPU, not on a CPU.

The train/val/test assignment is written to ``outputs/splits.csv`` so
``evaluate.py`` scores the exact held-out set this run never learned from.

Usage::

    python src/model/train.py
    python src/model/train.py --epochs 30 --backend fallback
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.backbone import build_backbone  # noqa: E402
from src.config import Config, load_config  # noqa: E402
from src.dataset import (  # noqa: E402
    TARGET_COLUMN,
    LesionDataset,
    diagnosis_classes,
    build_metadata_matrix,
    class_distribution,
    group_aware_split,
    load_unified_csv,
)
from src.model.risk_classifier import RiskClassifier  # noqa: E402
from src.utils import ensure_dir, get_logger, load_image_rgb, resolve_device, set_seed  # noqa: E402

logger = get_logger("train")

SPLIT_FILENAME = "splits.csv"
HISTORY_FILENAME = "training_history.json"


# ---------------------------------------------------------------------------
# Feature caching
# ---------------------------------------------------------------------------
def extract_features(
    df: pd.DataFrame,
    backbone,
    cfg: Config,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Embed every image in ``df`` with the frozen backbone.

    Args:
        df: Unified-schema dataframe.
        backbone: A constructed :class:`FeatureBackbone`.
        cfg: Loaded project config.
        device: Device to run the backbone on.
        batch_size: Images per forward pass.

    Returns:
        A ``(len(df), feature_dim)`` float32 array, row-aligned with ``df``.
    """
    from PIL import Image

    transform = backbone.build_transform(train=False)
    backbone.eval()

    features = np.zeros((len(df), backbone.feature_dim), dtype=np.float32)
    paths = [Config.resolve(p) for p in df["image_path"]]
    start = time.time()

    with torch.no_grad():
        for begin in range(0, len(paths), batch_size):
            chunk = paths[begin:begin + batch_size]
            tensors = [transform(Image.fromarray(load_image_rgb(p))) for p in chunk]
            batch = torch.stack(tensors).to(device)
            features[begin:begin + len(chunk)] = backbone(batch).cpu().numpy()

            done = begin + len(chunk)
            if done % (batch_size * 10) == 0 or done == len(paths):
                rate = done / max(time.time() - start, 1e-6)
                logger.info("  embedded %d/%d images (%.1f img/s)", done, len(paths), rate)

    return features


def load_or_build_feature_cache(
    df: pd.DataFrame,
    backbone,
    cfg: Config,
    device: torch.device,
    force: bool = False,
) -> np.ndarray:
    """Return cached embeddings for ``df``, computing and saving them if needed.

    The cache is keyed by image path *and* backbone name, so switching from the
    fallback backbone to PanDerm correctly invalidates it.

    Args:
        df: Unified-schema dataframe.
        backbone: The frozen backbone.
        cfg: Loaded project config.
        device: Device for extraction.
        force: Recompute even if a valid cache exists.

    Returns:
        A ``(len(df), feature_dim)`` float32 array row-aligned with ``df``.
    """
    cache_path = (cfg.path("paths.checkpoint_dir")
                  / str(cfg.get("train.feature_cache_name", "features.npz")))

    if cache_path.exists() and not force:
        try:
            cached = np.load(cache_path, allow_pickle=True)
            same_backbone = str(cached["backbone_name"]) == backbone.name
            cached_paths = list(cached["image_paths"])
            wanted_paths = list(df["image_path"])
            if same_backbone and cached_paths == wanted_paths:
                logger.info("Reusing feature cache %s (%d x %d)", cache_path.name,
                            *cached["features"].shape)
                return cached["features"].astype(np.float32)
            logger.info("Feature cache is stale (backbone or file list changed); "
                        "recomputing.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read feature cache (%s); recomputing.", exc)

    logger.info("Extracting features for %d images with %s ...", len(df), backbone.name)
    features = extract_features(df, backbone, cfg, device)

    ensure_dir(cache_path.parent)
    np.savez_compressed(
        cache_path,
        features=features,
        image_paths=np.array(list(df["image_path"]), dtype=object),
        backbone_name=backbone.name,
    )
    logger.info("Wrote feature cache -> %s", cache_path)
    return features


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def compute_class_weights(df: pd.DataFrame, classes: List[str]) -> torch.Tensor:
    """Return inverse-frequency class weights for the loss.

    Both real datasets are heavily imbalanced (HAM10000 is ~67% ordinary nevi),
    so without weighting the model learns to predict "Low" for everything and
    scores well on accuracy while being clinically useless.
    """
    counts = class_distribution(df, classes)
    total = sum(counts.values())
    weights = []
    for name in classes:
        count = counts.get(name, 0)
        weights.append(total / (len(classes) * count) if count > 0 else 0.0)
    tensor = torch.tensor(weights, dtype=torch.float32)
    logger.info("Class counts %s -> loss weights %s",
                counts, [round(float(w), 3) for w in tensor])
    return tensor


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Compute macro-averaged F1 without pulling in sklearn.

    Used as the early-stopping criterion: it weights every risk class equally,
    so a model that ignores the rare High class cannot score well.
    """
    scores = []
    for cls in range(num_classes):
        tp = float(np.sum((y_pred == cls) & (y_true == cls)))
        fp = float(np.sum((y_pred == cls) & (y_true != cls)))
        fn = float(np.sum((y_pred != cls) & (y_true == cls)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 else 0.0)
    return float(np.mean(scores))


def _evaluate_cached(
    model: RiskClassifier,
    features: torch.Tensor,
    metadata: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    num_classes: int,
) -> Tuple[float, float, float]:
    """Score the head on precomputed features.

    Returns:
        ``(loss, accuracy, macro_f1)``.
    """
    model.eval()
    with torch.no_grad():
        logits = model.forward_from_features(features, metadata)
        loss = float(criterion(logits, labels).item())
        preds = logits.argmax(dim=1).cpu().numpy()
    truth = labels.cpu().numpy()
    accuracy = float((preds == truth).mean())
    return loss, accuracy, macro_f1(truth, preds, num_classes)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------
def train_on_cached_features(
    model: RiskClassifier,
    splits: Dict[str, pd.DataFrame],
    features_by_split: Dict[str, np.ndarray],
    cfg: Config,
    device: torch.device,
) -> List[Dict[str, float]]:
    """Train the fusion head on precomputed embeddings.

    Returns:
        Per-epoch history records.
    """
    classes = diagnosis_classes(cfg)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    epochs = int(cfg.get("train.epochs", 15))
    batch_size = int(cfg.get("train.batch_size", 32))
    patience = int(cfg.get("train.early_stopping_patience", 5))

    tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for name, df in splits.items():
        tensors[name] = (
            torch.from_numpy(features_by_split[name]).to(device),
            torch.from_numpy(build_metadata_matrix(df, cfg)).to(device),
            torch.tensor([class_to_idx[r] for r in df[TARGET_COLUMN]],
                         dtype=torch.long, device=device),
        )

    weights = (compute_class_weights(splits["train"], classes).to(device)
               if bool(cfg.get("train.class_weighting", True)) else None)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(cfg.get("train.learning_rate", 1e-3)),
        weight_decay=float(cfg.get("train.weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    train_x, train_m, train_y = tensors["train"]
    n_train = train_x.shape[0]
    checkpoint_path = (ensure_dir(cfg.path("paths.checkpoint_dir"))
                       / str(cfg.get("train.checkpoint_name", "risk_head.pt")))

    history: List[Dict[str, float]] = []
    # Selection key is (macro-F1, -loss). Macro-F1 alone saturates easily on
    # small or easy datasets, and then the *earliest* epoch to reach the
    # ceiling wins -- typically an under-trained, badly calibrated model that
    # emits near-uniform probabilities. Breaking ties on loss keeps improving
    # calibration after accuracy has plateaued.
    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    has_val = len(splits["val"]) > 0
    if not has_val:
        logger.warning("Validation split is empty; selecting on training macro-F1 "
                       "instead. Prepare more data for a trustworthy run.")

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for begin in range(0, n_train, batch_size):
            idx = permutation[begin:begin + batch_size]
            # BatchNorm in the head needs >1 sample; skip a trailing single.
            if idx.numel() < 2:
                continue
            optimizer.zero_grad()
            logits = model.forward_from_features(train_x[idx], train_m[idx])
            loss = criterion(logits, train_y[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)

        _, train_acc, train_f1 = _evaluate_cached(
            model, train_x, train_m, train_y, criterion, len(classes))
        if has_val:
            val_x, val_m, val_y = tensors["val"]
            val_loss, val_acc, val_f1 = _evaluate_cached(
                model, val_x, val_m, val_y, criterion, len(classes))
        else:
            val_loss, val_acc, val_f1 = float("nan"), float("nan"), train_f1

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "train_macro_f1": train_f1, "val_loss": val_loss, "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        logger.info(
            "epoch %2d/%d | train loss %.4f acc %.3f f1 %.3f | val loss %.4f acc %.3f f1 %.3f",
            epoch, epochs, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1,
        )

        selection_f1 = val_f1 if has_val else train_f1
        selection_loss = val_loss if has_val else train_loss
        improved = (selection_f1 > best_f1
                    or (selection_f1 == best_f1 and selection_loss < best_loss))
        if improved:
            best_f1, best_loss, best_epoch = selection_f1, selection_loss, epoch
            epochs_without_improvement = 0
            model.save(checkpoint_path, extra={
                "selected_epoch": epoch,
                "selection_macro_f1": selection_f1,
                "selection_loss": selection_loss,
                "selection_split": "val" if has_val else "train",
            })
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs).",
                            epoch, patience)
                break

    logger.info("Best macro-F1 %.4f (loss %.4f) at epoch %d -> %s",
                best_f1, best_loss, best_epoch, checkpoint_path)
    return history


def train_end_to_end(
    model: RiskClassifier,
    splits: Dict[str, pd.DataFrame],
    cfg: Config,
    device: torch.device,
) -> List[Dict[str, float]]:
    """Train with per-epoch augmentation, re-embedding images every epoch.

    Slower but allows augmentation to vary across epochs. Selected by setting
    ``train.cache_features: false``.

    Returns:
        Per-epoch history records.
    """
    from torch.utils.data import DataLoader

    classes = diagnosis_classes(cfg)
    epochs = int(cfg.get("train.epochs", 15))
    batch_size = int(cfg.get("train.batch_size", 32))
    patience = int(cfg.get("train.early_stopping_patience", 5))
    augment = bool(cfg.get("train.augment", True))

    loaders = {}
    for name, df in splits.items():
        if len(df) == 0:
            continue
        dataset = LesionDataset(
            df, cfg,
            transform=model.backbone.build_transform(train=(augment and name == "train")),
        )
        loaders[name] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(name == "train"),
            num_workers=int(cfg.get("train.num_workers", 0)), drop_last=(name == "train"),
        )

    weights = (compute_class_weights(splits["train"], classes).to(device)
               if bool(cfg.get("train.class_weighting", True)) else None)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(cfg.get("train.learning_rate", 1e-3)),
        weight_decay=float(cfg.get("train.weight_decay", 1e-4)),
    )

    checkpoint_path = (ensure_dir(cfg.path("paths.checkpoint_dir"))
                       / str(cfg.get("train.checkpoint_name", "risk_head.pt")))
    history: List[Dict[str, float]] = []
    # Same (macro-F1, -loss) selection key as the cached path; see there.
    best_f1, best_loss, best_epoch, stale = -1.0, float("inf"), -1, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for images, meta, labels in loaders["train"]:
            images, meta, labels = images.to(device), meta.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images, meta), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_preds: List[int] = []
        val_true: List[int] = []
        with torch.no_grad():
            for images, meta, labels in loaders.get("val", loaders["train"]):
                logits = model(images.to(device), meta.to(device))
                val_preds.extend(logits.argmax(dim=1).cpu().tolist())
                val_true.extend(labels.tolist())
        val_f1 = macro_f1(np.array(val_true), np.array(val_preds), len(classes))
        val_acc = float(np.mean(np.array(val_true) == np.array(val_preds)))

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_acc": val_acc, "val_macro_f1": val_f1})
        logger.info("epoch %2d/%d | train loss %.4f | val acc %.3f f1 %.3f",
                    epoch, epochs, train_loss, val_acc, val_f1)

        if val_f1 > best_f1 or (val_f1 == best_f1 and train_loss < best_loss):
            best_f1, best_loss, best_epoch, stale = val_f1, train_loss, epoch, 0
            model.save(checkpoint_path, extra={"selected_epoch": epoch,
                                               "selection_macro_f1": val_f1,
                                               "selection_loss": train_loss})
        else:
            stale += 1
            if stale >= patience:
                logger.info("Early stopping at epoch %d.", epoch)
                break

    logger.info("Best macro-F1 %.4f (loss %.4f) at epoch %d -> %s",
                best_f1, best_loss, best_epoch, checkpoint_path)
    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_training(cfg: Config, force_features: bool = False) -> Path:
    """Prepare splits, train the head and persist the checkpoint and history.

    Returns:
        Path to the saved checkpoint.
    """
    set_seed(int(cfg.get("project.seed", 42)))
    device = resolve_device(cfg.get("project.device", "auto"))
    logger.info("Device: %s", device)

    df = load_unified_csv(cfg)
    sources = sorted(df["source"].unique())
    logger.info("Loaded %d rows from %s", len(df), sources)
    if any(s == "synthetic_demo" for s in sources):
        logger.warning(
            "This run includes SYNTHETIC demo data. Any metric it produces is a "
            "smoke test of the code, NOT a measure of clinical performance."
        )

    train_df, val_df, test_df = group_aware_split(df, cfg)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, part in splits.items():
        logger.info("  %-5s %4d images  %s", name, len(part),
                    class_distribution(part, diagnosis_classes(cfg)))

    # Persist the split so evaluate.py scores the same held-out set.
    output_dir = ensure_dir(cfg.path("paths.output_dir"))
    split_rows = pd.concat(
        [part.assign(split=name) for name, part in splits.items()], ignore_index=True
    )
    split_path = output_dir / SPLIT_FILENAME
    split_rows[["image_path", "diagnosis", "risk_label", "source", "split"]].to_csv(split_path, index=False)
    logger.info("Wrote split assignment -> %s", split_path)

    backbone = build_backbone(cfg).to(device)
    model = RiskClassifier(backbone, cfg, freeze_backbone=True).to(device)

    if bool(cfg.get("train.cache_features", True)):
        logger.info("Feature caching ON -- augmentation is disabled in this mode "
                    "(embeddings are computed once with the eval transform).")
        all_features = load_or_build_feature_cache(df, backbone, cfg, device,
                                                   force=force_features)
        index_of = {p: i for i, p in enumerate(df["image_path"])}
        features_by_split = {
            name: all_features[[index_of[p] for p in part["image_path"]]]
            if len(part) else np.zeros((0, backbone.feature_dim), dtype=np.float32)
            for name, part in splits.items()
        }
        history = train_on_cached_features(model, splits, features_by_split, cfg, device)
    else:
        logger.info("Feature caching OFF -- re-embedding every epoch (slow on CPU).")
        history = train_end_to_end(model, splits, cfg, device)

    history_path = output_dir / HISTORY_FILENAME
    history_path.write_text(json.dumps({
        "backbone": backbone.describe(),
        "backbone_is_fallback": backbone.is_fallback,
        "sources": sources,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "history": history,
    }, indent=2), encoding="utf-8")
    logger.info("Wrote training history -> %s", history_path)

    return (cfg.path("paths.checkpoint_dir")
            / str(cfg.get("train.checkpoint_name", "risk_head.pt")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to a YAML config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--backend", choices=["panderm", "fallback", "auto"], default=None,
                        help="Override backbone.backend.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable feature caching (slower, allows augmentation).")
    parser.add_argument("--force-features", action="store_true",
                        help="Recompute the feature cache even if it looks valid.")
    args = parser.parse_args()

    overrides: Dict[str, object] = {}
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["train.batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["train.learning_rate"] = args.lr
    if args.backend is not None:
        overrides["backbone.backend"] = args.backend
    if args.no_cache:
        overrides["train.cache_features"] = False

    cfg = load_config(args.config, overrides=overrides)
    run_training(cfg, force_features=args.force_features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
