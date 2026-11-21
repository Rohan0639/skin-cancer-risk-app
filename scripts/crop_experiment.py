"""Does cropping to the segmented lesion before classification help?

The pipeline segments the lesion for the ABCDE measurements but then feeds
PanDerm the *whole frame*. Two consequences show up in the failure cases:

* A photo with several lesions produces one CLS embedding pooled over all of
  them, so a melanoma beside three benign moles is averaged into "mole".
* A lesion covering ~10% of a 224x224 input survives as roughly 70x70 pixels,
  so the fine colour and border detail the diagnosis depends on is already
  gone when the model sees it.

This embeds every image a second time, cropped to the segmentation bounding
box with padding, then trains the same head on each feature set over the same
split and seeds. Any difference is attributable to the crop alone.

Usage::

    python scripts/crop_experiment.py --seeds 5 --pad 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backbone import build_backbone  # noqa: E402
from src.config import Config, load_config  # noqa: E402
from src.dataset import (  # noqa: E402
    TARGET_COLUMN,
    build_metadata_matrix,
    diagnosis_classes,
    group_aware_split,
    load_unified_csv,
    risk_for,
)
from src.model.fusion import FusionHead  # noqa: E402
from src.model.train import compute_class_weights, load_or_build_feature_cache, macro_f1  # noqa: E402
from src.segmentation import segment_lesion  # noqa: E402
from src.utils import ensure_dir, get_logger, load_image_rgb, resolve_device, set_seed  # noqa: E402

logger = get_logger("crop_experiment")


def crop_to_lesion(rgb: np.ndarray, pad: float) -> np.ndarray:
    """Crop to the segmented lesion's bounding box, padded and squared.

    The box is expanded to a square before padding so the resize to 224 does
    not stretch the lesion, which would distort exactly the asymmetry and
    border cues the diagnosis depends on. Falls back to the full frame when
    segmentation finds nothing.
    """
    seg = segment_lesion(rgb)
    h, w = rgb.shape[:2]
    if not seg.found:
        return rgb
    y0, x0, y1, x1 = seg.bbox
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    half = max(y1 - y0, x1 - x0) / 2 * (1 + pad)
    half = max(half, 16)
    y0 = int(max(0, cy - half)); y1 = int(min(h, cy + half))
    x0 = int(max(0, cx - half)); x1 = int(min(w, cx + half))
    if y1 - y0 < 8 or x1 - x0 < 8:
        return rgb
    return rgb[y0:y1, x0:x1]


def extract_cropped(df, backbone, cfg: Config, device, pad: float, cache: Path) -> np.ndarray:
    """Embed every image after cropping to its lesion, with caching."""
    from PIL import Image

    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=True)
            if list(z["image_paths"]) == list(df["image_path"]) and \
                    float(z["pad"]) == pad and str(z["backbone_name"]) == backbone.name:
                logger.info("Reusing crop cache %s", cache.name)
                return z["features"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read crop cache (%s); recomputing.", exc)

    transform = backbone.build_transform(train=False)
    backbone.eval()
    paths = [Config.resolve(p) for p in df["image_path"]]
    feats = np.zeros((len(paths), backbone.feature_dim), dtype=np.float32)
    start, batch, fallbacks = time.time(), 16, 0

    with torch.no_grad():
        for i in range(0, len(paths), batch):
            chunk = []
            for p in paths[i:i + batch]:
                rgb = load_image_rgb(p)
                c = crop_to_lesion(rgb, pad)
                if c.shape == rgb.shape:
                    fallbacks += 1
                chunk.append(transform(Image.fromarray(np.ascontiguousarray(c))))
            feats[i:i + len(chunk)] = backbone(torch.stack(chunk).to(device)).cpu().numpy()
            done = i + len(chunk)
            if done % (batch * 20) == 0 or done == len(paths):
                logger.info("  cropped %d/%d (%.1f img/s)", done, len(paths),
                            done / max(time.time() - start, 1e-6))

    logger.info("segmentation fell back to the full frame on %d/%d images",
                fallbacks, len(paths))
    ensure_dir(cache.parent)
    np.savez_compressed(cache, features=feats, pad=pad,
                        image_paths=np.array(list(df["image_path"]), dtype=object),
                        backbone_name=backbone.name)
    return feats


def train_eval(X, M, y, n_classes, weights, device, seed, hp, epochs=80):
    """Train a head and return (val F1, test F1, test predictions)."""
    set_seed(seed)
    head = FusionHead(feature_dim=X["train"].shape[1], metadata_dim=M["train"].shape[1],
                      num_classes=n_classes, image_proj_dim=hp["proj"],
                      metadata_embed_dim=32, hidden_dims=hp["hidden"],
                      dropout=hp["dropout"], use_metadata=True).to(device)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=hp["smooth"])
    opt = torch.optim.AdamW(head.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = X["train"].shape[0]
    best, best_state, stale = -1.0, None, 0
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            if idx.numel() < 2:
                continue
            opt.zero_grad()
            crit(head(X["train"][idx], M["train"][idx]), y["train"][idx]).backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            vp = head(X["val"], M["val"]).argmax(1).cpu().numpy()
        f1 = macro_f1(y["val"].cpu().numpy(), vp, n_classes)
        if f1 > best:
            best, stale = f1, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= 10:
                break
    if best_state:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        tp = head(X["test"], M["test"]).argmax(1).cpu().numpy()
    return best, macro_f1(y["test"].cpu().numpy(), tp, n_classes), tp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--pad", type=float, default=0.25)
    args = parser.parse_args()

    cfg = load_config()
    device = resolve_device(cfg.get("project.device", "auto"))
    classes = diagnosis_classes(cfg)
    n_classes = len(classes)

    df = load_unified_csv(cfg)
    tr, va, te = group_aware_split(df, cfg)
    splits = {"train": tr, "val": va, "test": te}
    logger.info("split train=%d val=%d test=%d", len(tr), len(va), len(te))

    backbone = build_backbone(cfg).to(device)
    full = load_or_build_feature_cache(df, backbone, cfg, device)
    cropped = extract_cropped(df, backbone, cfg, device, args.pad,
                              cfg.path("paths.checkpoint_dir") / "features_cropped.npz")

    idx = {p: i for i, p in enumerate(df["image_path"])}
    c2i = {c: i for i, c in enumerate(classes)}
    rows = {k: [idx[p] for p in part["image_path"]] for k, part in splits.items()}
    M = {k: torch.from_numpy(build_metadata_matrix(part, cfg)).to(device)
         for k, part in splits.items()}
    y = {k: torch.tensor([c2i[v] for v in part[TARGET_COLUMN]], dtype=torch.long,
                         device=device) for k, part in splits.items()}
    weights = compute_class_weights(tr, classes).to(device)

    hp = dict(proj=int(cfg.get("model.image_proj_dim")),
              hidden=list(cfg.get("model.hidden_dims")),
              dropout=float(cfg.get("model.dropout")),
              lr=float(cfg.get("train.learning_rate")),
              wd=float(cfg.get("train.weight_decay")),
              smooth=float(cfg.get("train.label_smoothing", 0.0)))

    mel = c2i["MEL"]
    y_test = y["test"].cpu().numpy()
    results: Dict[str, Tuple[float, float, float, float]] = {}

    variants = {
        "full frame (current)": {k: full[rows[k]] for k in splits},
        "lesion crop": {k: cropped[rows[k]] for k in splits},
        "both concatenated": {k: np.concatenate([full[rows[k]], cropped[rows[k]]], axis=1)
                              for k in splits},
    }

    for name, feats in variants.items():
        X = {k: torch.from_numpy(v).to(device) for k, v in feats.items()}
        vs, ts, preds = [], [], []
        for s in range(args.seeds):
            v, t, p = train_eval(X, M, y, n_classes, weights, device, s, hp)
            vs.append(v); ts.append(t); preds.append(p)
        # Melanomas whose predicted class maps to the Low tier.
        low_rates = []
        for p in preds:
            m = y_test == mel
            if m.sum():
                low_rates.append(float(np.mean(
                    [risk_for(cfg, classes[c]) == "Low" for c in p[m]])))
        results[name] = (float(np.mean(vs)), float(np.mean(ts)),
                         float(np.std(ts)), float(np.mean(low_rates)))
        logger.info("%s done", name)

    print()
    print("=" * 78)
    print(f"  same split, same head, {args.seeds} seeds"
          f"   ({len(tr)} train / {len(va)} val / {len(te)} test)")
    print("-" * 78)
    print(f"  {'features':24s} {'val F1':>8s} {'test F1':>16s} {'MEL shown Low':>16s}")
    base = results["full frame (current)"]
    for k, (v, t, sd, low) in results.items():
        d = "" if k == "full frame (current)" else f"  ({t - base[1]:+.4f})"
        print(f"  {k:24s} {v:8.4f} {t:8.4f} ±{sd:.3f}{d:>10s} {low:>12.1%}")
    print("=" * 78)

    out = Path("outputs/crop_experiment.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: {"val_macro_f1": v, "test_macro_f1": t,
                                   "test_std": sd, "melanoma_shown_low": low}
                               for k, (v, t, sd, low) in results.items()}, indent=2),
                   encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
