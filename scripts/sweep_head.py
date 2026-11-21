"""Sweep fusion-head hyperparameters on cached embeddings.

The backbone is frozen, so every configuration trains on the same precomputed
features: a full sweep costs seconds, not hours. Selection is on the
**validation** split; the test split is scored once at the end for the winner
only, so the search cannot leak into the reported number.

Usage::

    python scripts/sweep_head.py --trials 40
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backbone import build_backbone  # noqa: E402
from src.config import Config, load_config  # noqa: E402
from src.dataset import (  # noqa: E402
    TARGET_COLUMN,
    build_metadata_matrix,
    class_distribution,
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

logger = get_logger("sweep_head")


def train_once(
    feats: Dict[str, torch.Tensor],
    metas: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    classes: List[str],
    weights: torch.Tensor | None,
    hp: dict,
    device: torch.device,
    seed: int,
) -> Tuple[float, FusionHead]:
    """Train one head configuration; return its best validation macro-F1."""
    set_seed(seed)
    head = FusionHead(
        feature_dim=feats["train"].shape[1],
        metadata_dim=metas["train"].shape[1],
        num_classes=len(classes),
        image_proj_dim=hp["proj"],
        metadata_embed_dim=hp["meta"],
        hidden_dims=hp["hidden"],
        dropout=hp["dropout"],
        use_metadata=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=hp["smooth"])
    opt = torch.optim.AdamW(head.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"])

    n = feats["train"].shape[0]
    best_f1, best_state, stale = -1.0, None, 0

    for _ in range(hp["epochs"]):
        head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, hp["batch"]):
            idx = perm[i:i + hp["batch"]]
            if idx.numel() < 2:
                continue
            opt.zero_grad()
            loss = criterion(head(feats["train"][idx], metas["train"][idx]),
                             labels["train"][idx])
            loss.backward()
            opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            pred = head(feats["val"], metas["val"]).argmax(1).cpu().numpy()
        f1 = macro_f1(labels["val"].cpu().numpy(), pred, len(classes))
        if f1 > best_f1:
            best_f1, stale = f1, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= 8:
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    return best_f1, head


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=3,
                        help="Seeds averaged per config, to avoid picking noise.")
    args = parser.parse_args()

    cfg = load_config()
    device = resolve_device(cfg.get("project.device", "auto"))
    classes = diagnosis_classes(cfg)

    df = load_unified_csv(cfg)
    train_df, val_df, test_df = group_aware_split(df, cfg)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    logger.info("train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    backbone = build_backbone(cfg).to(device)
    all_feats = load_or_build_feature_cache(df, backbone, cfg, device)
    index = {p: i for i, p in enumerate(df["image_path"])}

    c2i = {c: i for i, c in enumerate(classes)}
    feats, metas, labels = {}, {}, {}
    for name, part in splits.items():
        rows = [index[p] for p in part["image_path"]]
        feats[name] = torch.from_numpy(all_feats[rows]).to(device)
        metas[name] = torch.from_numpy(build_metadata_matrix(part, cfg)).to(device)
        labels[name] = torch.tensor([c2i[v] for v in part[TARGET_COLUMN]],
                                    dtype=torch.long, device=device)

    weights = compute_class_weights(train_df, classes).to(device)

    grid = {
        "proj": [128, 256, 512],
        "hidden": [[128], [256, 128], [512, 256]],
        "dropout": [0.2, 0.35, 0.5],
        "lr": [3e-4, 1e-3, 3e-3],
        "wd": [1e-4, 1e-2],
        "smooth": [0.0, 0.1],
        "meta": [32],
        "batch": [64],
        "epochs": [60],
    }
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    rng = np.random.default_rng(0)
    rng.shuffle(combos)
    combos = combos[:args.trials]
    logger.info("Evaluating %d configs x %d seeds ...", len(combos), args.seeds)

    results = []
    for n, combo in enumerate(combos, 1):
        hp = dict(zip(keys, combo))
        f1s = [train_once(feats, metas, labels, classes, weights, hp, device, s)[0]
               for s in range(args.seeds)]
        mean_f1 = float(np.mean(f1s))
        results.append((mean_f1, float(np.std(f1s)), hp))
        if n % 10 == 0 or n == len(combos):
            logger.info("  %d/%d  best so far %.4f", n, len(combos),
                        max(r[0] for r in results))

    results.sort(key=lambda r: -r[0])
    print()
    print("=" * 76)
    print("  top configurations by validation macro-F1 (mean over seeds)")
    print("-" * 76)
    for f1, sd, hp in results[:6]:
        print(f"  {f1:.4f} ±{sd:.3f}  proj={hp['proj']:<4} hidden={str(hp['hidden']):<12}"
              f" drop={hp['dropout']} lr={hp['lr']:<7} wd={hp['wd']} smooth={hp['smooth']}")

    best_hp = results[0][2]
    baseline = {"proj": int(cfg.get("model.image_proj_dim")),
                "hidden": list(cfg.get("model.hidden_dims")),
                "dropout": float(cfg.get("model.dropout")),
                "lr": float(cfg.get("train.learning_rate")),
                "wd": float(cfg.get("train.weight_decay")),
                "smooth": 0.0, "meta": 32, "batch": 64, "epochs": 60}
    base_f1 = float(np.mean([
        train_once(feats, metas, labels, classes, weights, baseline, device, s)[0]
        for s in range(args.seeds)]))

    print("-" * 76)
    print(f"  current config   val macro-F1 {base_f1:.4f}")
    print(f"  best found       val macro-F1 {results[0][0]:.4f}")
    print("=" * 76)

    out = Path("outputs/head_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"baseline_val_macro_f1": base_f1,
         "best_val_macro_f1": results[0][0],
         "best": best_hp,
         "top": [{"val_macro_f1": f, "std": s, "hp": h} for f, s, h in results[:10]]},
        indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
