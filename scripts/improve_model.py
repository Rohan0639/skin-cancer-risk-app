"""Measure three cheap improvements over the frozen-backbone baseline.

The backbone is frozen, so the only things left that do not cost hours of CPU
fine-tuning are applied around it:

1. **Feature-space augmentation.** Lesion photos have no canonical
   orientation, so horizontal and vertical flips are label-preserving. Each
   flip is embedded once and treated as an extra training row, quadrupling the
   training set without touching the backbone.
2. **Test-time augmentation.** The same four views are embedded at inference
   and their probabilities averaged, which cancels some of the orientation
   sensitivity a single view carries.
3. **Seed ensembling.** Heads train in seconds, so several are trained from
   different seeds and their probabilities averaged.

Each is measured separately and cumulatively, on the same fixed split, so a
gain can be attributed rather than assumed. Selection stays on validation;
test is reported alongside for information.

Usage::

    python scripts/improve_model.py --seeds 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
)
from src.model.fusion import FusionHead  # noqa: E402
from src.model.train import compute_class_weights, macro_f1  # noqa: E402
from src.utils import ensure_dir, get_logger, load_image_rgb, resolve_device, set_seed  # noqa: E402

logger = get_logger("improve_model")

#: Label-preserving views. A lesion has no canonical orientation, so a flip is
#: the same lesion; brightness or colour jitter would not be safe here because
#: colour is itself a diagnostic signal.
VIEWS = ["id", "h", "v", "hv"]


def apply_view(rgb: np.ndarray, view: str) -> np.ndarray:
    if view == "h":
        return rgb[:, ::-1]
    if view == "v":
        return rgb[::-1, :]
    if view == "hv":
        return rgb[::-1, ::-1]
    return rgb


def extract_views(df, backbone, cfg: Config, device, cache: Path) -> Dict[str, np.ndarray]:
    """Embed every image under each view, caching the result."""
    from PIL import Image

    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=True)
            if list(z["image_paths"]) == list(df["image_path"]) and \
                    str(z["backbone_name"]) == backbone.name:
                logger.info("Reusing view cache %s", cache.name)
                return {v: z[v] for v in VIEWS}
            logger.info("View cache stale; recomputing.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read view cache (%s); recomputing.", exc)

    transform = backbone.build_transform(train=False)
    backbone.eval()
    paths = [Config.resolve(p) for p in df["image_path"]]
    out = {v: np.zeros((len(paths), backbone.feature_dim), dtype=np.float32) for v in VIEWS}
    start = time.time()
    batch = 16

    with torch.no_grad():
        for begin in range(0, len(paths), batch):
            chunk = [load_image_rgb(p) for p in paths[begin:begin + batch]]
            for view in VIEWS:
                tensors = [transform(Image.fromarray(np.ascontiguousarray(apply_view(r, view))))
                           for r in chunk]
                feats = backbone(torch.stack(tensors).to(device)).cpu().numpy()
                out[view][begin:begin + len(chunk)] = feats
            done = begin + len(chunk)
            if done % (batch * 20) == 0 or done == len(paths):
                rate = done * len(VIEWS) / max(time.time() - start, 1e-6)
                logger.info("  %d/%d images x%d views (%.1f emb/s)",
                            done, len(paths), len(VIEWS), rate)

    ensure_dir(cache.parent)
    np.savez_compressed(cache, image_paths=np.array(list(df["image_path"]), dtype=object),
                        backbone_name=backbone.name, **out)
    logger.info("Wrote %s", cache)
    return out


def train_head(X, M, y, n_classes, weights, device, seed, hp, epochs=80):
    """Train one head; return it and its best validation macro-F1."""
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
            pred = head(X["val"], M["val"]).argmax(1).cpu().numpy()
        f1 = macro_f1(y["val"].cpu().numpy(), pred, n_classes)
        if f1 > best:
            best, stale = f1, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= 10:
                break
    if best_state:
        head.load_state_dict(best_state)
    return head, best


def probs_for(head, feats_by_view, meta, views: List[str], device) -> np.ndarray:
    """Average softmax probabilities over the requested views."""
    head.eval()
    acc = None
    with torch.no_grad():
        for v in views:
            p = torch.softmax(head(feats_by_view[v], meta), dim=1).cpu().numpy()
            acc = p if acc is None else acc + p
    return acc / len(views)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config()
    device = resolve_device(cfg.get("project.device", "auto"))
    classes = diagnosis_classes(cfg)
    n_classes = len(classes)

    df = load_unified_csv(cfg)
    tr, va, te = group_aware_split(df, cfg)
    splits = {"train": tr, "val": va, "test": te}
    logger.info("split  train=%d val=%d test=%d", len(tr), len(va), len(te))

    backbone = build_backbone(cfg).to(device)
    views = extract_views(df, backbone, cfg,
                          device, cfg.path("paths.checkpoint_dir") / "features_views.npz")

    idx = {p: i for i, p in enumerate(df["image_path"])}
    c2i = {c: i for i, c in enumerate(classes)}
    rows = {k: [idx[p] for p in part["image_path"]] for k, part in splits.items()}

    meta = {k: torch.from_numpy(build_metadata_matrix(part, cfg)).to(device)
            for k, part in splits.items()}
    labels = {k: torch.tensor([c2i[v] for v in part[TARGET_COLUMN]],
                              dtype=torch.long, device=device)
              for k, part in splits.items()}
    by_view = {k: {v: torch.from_numpy(views[v][rows[k]]).to(device) for v in VIEWS}
               for k in splits}
    weights = compute_class_weights(tr, classes).to(device)

    hp = dict(proj=int(cfg.get("model.image_proj_dim")),
              hidden=list(cfg.get("model.hidden_dims")),
              dropout=float(cfg.get("model.dropout")),
              lr=float(cfg.get("train.learning_rate")),
              wd=float(cfg.get("train.weight_decay")),
              smooth=float(cfg.get("train.label_smoothing", 0.0)))

    def build(augmented: bool):
        """Training tensors, optionally with the flipped views stacked in."""
        if not augmented:
            return ({"train": by_view["train"]["id"], "val": by_view["val"]["id"]},
                    {"train": meta["train"], "val": meta["val"]},
                    {"train": labels["train"], "val": labels["val"]})
        Xt = torch.cat([by_view["train"][v] for v in VIEWS], dim=0)
        Mt = torch.cat([meta["train"]] * len(VIEWS), dim=0)
        yt = torch.cat([labels["train"]] * len(VIEWS), dim=0)
        return ({"train": Xt, "val": by_view["val"]["id"]},
                {"train": Mt, "val": meta["val"]},
                {"train": yt, "val": labels["val"]})

    y_val = labels["val"].cpu().numpy()
    y_test = labels["test"].cpu().numpy()
    results: Dict[str, Tuple[float, float]] = {}

    def score(heads, use_tta: bool) -> Tuple[float, float]:
        vs = VIEWS if use_tta else ["id"]
        pv = np.mean([probs_for(h, by_view["val"], meta["val"], vs, device) for h in heads], axis=0)
        pt = np.mean([probs_for(h, by_view["test"], meta["test"], vs, device) for h in heads], axis=0)
        return (macro_f1(y_val, pv.argmax(1), n_classes),
                macro_f1(y_test, pt.argmax(1), n_classes))

    for augmented in (False, True):
        X, M, y = build(augmented)
        heads = [train_head(X, M, y, n_classes, weights, device, s, hp)[0]
                 for s in range(args.seeds)]
        tag = "aug" if augmented else "base"
        results[f"{tag}, single head"] = score(heads[:1], False)
        results[f"{tag}, single + TTA"] = score(heads[:1], True)
        results[f"{tag}, ensemble"] = score(heads, False)
        results[f"{tag}, ensemble + TTA"] = score(heads, True)
        logger.info("%s done", tag)

    print()
    print("=" * 74)
    print(f"  frozen backbone, {len(tr)} train / {len(va)} val / {len(te)} test"
          f", {args.seeds} seeds")
    print("-" * 74)
    print(f"  {'setting':28s} {'val macro-F1':>14s} {'test macro-F1':>15s}")
    base = results["base, single head"]
    for k, (v, t) in results.items():
        delta = t - base[1]
        mark = f"  ({delta:+.4f})" if k != "base, single head" else ""
        print(f"  {k:28s} {v:>14.4f} {t:>15.4f}{mark}")
    print("=" * 74)

    out = Path("outputs/improvement_ablation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {k: {"val_macro_f1": v, "test_macro_f1": t} for k, (v, t) in results.items()},
        indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
