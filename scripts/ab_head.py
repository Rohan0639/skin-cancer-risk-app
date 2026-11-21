"""Controlled A/B of head hyperparameters on one fixed dataset.

Earlier comparisons moved the dataset and the hyperparameters at the same
time, so a drop in macro-F1 could not be attributed to either. This trains
several configurations on the *same* cached features and the *same* split,
over several seeds, and reports validation and test macro-F1 for each.

Test scores are printed for information. Selection must still be made on
validation, and the winner re-measured once on test.

Usage::

    python scripts/ab_head.py --seeds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backbone import build_backbone  # noqa: E402
from src.config import load_config  # noqa: E402
from src.dataset import (  # noqa: E402
    TARGET_COLUMN,
    build_metadata_matrix,
    diagnosis_classes,
    group_aware_split,
    load_unified_csv,
)
from src.model.fusion import FusionHead  # noqa: E402
from src.model.train import (  # noqa: E402
    compute_class_weights,
    load_or_build_feature_cache,
    macro_f1,
)
from src.utils import get_logger, resolve_device, set_seed  # noqa: E402

logger = get_logger("ab_head")

CONFIGS: Dict[str, dict] = {
    "original (pre-sweep)": dict(proj=256, hidden=[256, 128], dropout=0.3,
                                 lr=1e-3, wd=1e-4, smooth=0.0),
    "swept (current)": dict(proj=128, hidden=[256, 128], dropout=0.5,
                            lr=3e-3, wd=1e-2, smooth=0.1),
    "swept, no smoothing": dict(proj=128, hidden=[256, 128], dropout=0.5,
                                lr=3e-3, wd=1e-2, smooth=0.0),
    "wider, light reg": dict(proj=512, hidden=[256, 128], dropout=0.35,
                             lr=1e-3, wd=1e-4, smooth=0.0),
}


def run(feats, metas, labels, classes, weights, hp, device, seed, epochs=80):
    """Train one configuration; return (best val F1, test F1 at that point)."""
    set_seed(seed)
    head = FusionHead(
        feature_dim=feats["train"].shape[1], metadata_dim=metas["train"].shape[1],
        num_classes=len(classes), image_proj_dim=hp["proj"],
        metadata_embed_dim=32, hidden_dims=hp["hidden"],
        dropout=hp["dropout"], use_metadata=True).to(device)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=hp["smooth"])
    opt = torch.optim.AdamW(head.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = feats["train"].shape[0]
    best_val, best_test, stale = -1.0, 0.0, 0
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            if idx.numel() < 2:
                continue
            opt.zero_grad()
            crit(head(feats["train"][idx], metas["train"][idx]), labels["train"][idx]).backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            v = head(feats["val"], metas["val"]).argmax(1).cpu().numpy()
            t = head(feats["test"], metas["test"]).argmax(1).cpu().numpy()
        vf = macro_f1(labels["val"].cpu().numpy(), v, len(classes))
        if vf > best_val:
            best_val, stale = vf, 0
            best_test = macro_f1(labels["test"].cpu().numpy(), t, len(classes))
        else:
            stale += 1
            if stale >= 10:
                break
    return best_val, best_test


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config()
    device = resolve_device(cfg.get("project.device", "auto"))
    classes = diagnosis_classes(cfg)

    df = load_unified_csv(cfg)
    tr, va, te = group_aware_split(df, cfg)
    splits = {"train": tr, "val": va, "test": te}
    logger.info("fixed split  train=%d val=%d test=%d", len(tr), len(va), len(te))

    backbone = build_backbone(cfg).to(device)
    allf = load_or_build_feature_cache(df, backbone, cfg, device)
    idx = {p: i for i, p in enumerate(df["image_path"])}
    c2i = {c: i for i, c in enumerate(classes)}

    feats, metas, labels = {}, {}, {}
    for name, part in splits.items():
        rows = [idx[p] for p in part["image_path"]]
        feats[name] = torch.from_numpy(allf[rows]).to(device)
        metas[name] = torch.from_numpy(build_metadata_matrix(part, cfg)).to(device)
        labels[name] = torch.tensor([c2i[v] for v in part[TARGET_COLUMN]],
                                    dtype=torch.long, device=device)
    weights = compute_class_weights(tr, classes).to(device)

    print()
    print("=" * 72)
    print(f"  same data, same split, {args.seeds} seeds each")
    print(f"  {len(tr)} train / {len(va)} val / {len(te)} test")
    print("-" * 72)
    print(f"  {'config':24s} {'val macro-F1':>16s} {'test macro-F1':>16s}")
    for name, hp in CONFIGS.items():
        res = [run(feats, metas, labels, classes, weights, hp, device, s)
               for s in range(args.seeds)]
        v = np.array([r[0] for r in res])
        t = np.array([r[1] for r in res])
        print(f"  {name:24s}  {v.mean():.4f} ±{v.std():.3f}   {t.mean():.4f} ±{t.std():.3f}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
