"""Sweep attention-rollout variants and pick the best by measured focus ratio.

Plain rollout lands on the lesion at 1.33x chance (median) and beats chance on
68% of test images -- better than everything else tried, but still poor on
roughly one image in three. Rollout has three standard knobs that change
localisation a lot:

* **head fusion** -- averaging heads dilutes a few sharply-focused ones with
  many diffuse ones; ``max`` keeps the sharpest, ``min`` the most conservative.
* **discard ratio** -- zeroing the weakest attention entries before
  renormalising removes the diffuse background floor that otherwise
  accumulates over twelve layers of matrix products.
* **layer range** -- early blocks attend almost uniformly; starting the
  product later can sharpen the result.

Everything here is gradient-free, so whatever wins can run in ONNX Runtime Web.

Usage::

    python scripts/tune_rollout.py --n 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_saliency import (  # noqa: E402
    attention_maps,
    focus_ratio,
    trunk_forward,
)
from src.config import Config, load_config  # noqa: E402
from src.dataset import encode_metadata  # noqa: E402
from src.model.risk_classifier import load_risk_classifier  # noqa: E402
from src.segmentation import segment_lesion  # noqa: E402
from src.utils import get_logger, load_image_rgb  # noqa: E402

logger = get_logger("tune_rollout")


def rollout_variant(attns: List[torch.Tensor], fusion: str, discard: float,
                    start: int) -> np.ndarray:
    """Attention rollout with configurable head fusion, pruning and depth.

    Args:
        attns: Per-block attention, each ``(1, heads, N, N)``.
        fusion: ``"mean"``, ``"max"`` or ``"min"`` across heads.
        discard: Fraction of the weakest attention entries to zero per row
            before renormalising.
        start: Index of the first block to include in the product.

    Returns:
        The CLS row over patches, length ``N - 1``.
    """
    result = None
    for a in attns[start:]:
        if fusion == "max":
            a = a.max(dim=1).values
        elif fusion == "min":
            a = a.min(dim=1).values
        else:
            a = a.mean(dim=1)

        if discard > 0:
            flat = a.flatten(start_dim=-2)
            k = int(flat.shape[-1] * discard)
            if k > 0:
                idx = flat.topk(k, dim=-1, largest=False).indices
                flat = flat.scatter(-1, idx, 0.0)
                a = flat.view_as(a)

        n = a.shape[-1]
        a = a + torch.eye(n, device=a.device, dtype=a.dtype)
        a = a / a.sum(dim=-1, keepdim=True)
        result = a if result is None else a @ result
    return result[0, 0, 1:].cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()

    import pandas as pd
    from PIL import Image

    cfg = load_config()
    model = load_risk_classifier(cfg=cfg, strict=True).eval()
    vit = model.backbone.model
    transform = model.backbone.build_transform(train=False)

    splits = pd.read_csv(cfg.path("paths.output_dir") / "splits.csv")
    test = set(splits[splits.split == "test"].image_path)
    df = pd.read_csv(cfg.path("paths.unified_csv"))
    df = df[df.image_path.isin(test)].sample(n=args.n, random_state=1)

    variants: List[Tuple[str, float, int]] = []
    for fusion in ("mean", "max", "min"):
        for discard in (0.0, 0.5, 0.9):
            for start in (0, 6):
                variants.append((fusion, discard, start))

    scores: Dict[str, List[float]] = {}
    examined = 0

    for row in df.to_dict("records"):
        rgb = load_image_rgb(Config.resolve(row["image_path"]))
        seg = segment_lesion(rgb)
        if not seg.found:
            continue
        image = transform(Image.fromarray(rgb)).unsqueeze(0)
        attns = attention_maps(vit, image)
        mask = seg.mask > 0
        examined += 1

        for fusion, discard, start in variants:
            key = f"{fusion:4s} discard={discard:.1f} from L{start}"
            heat = rollout_variant(attns, fusion, discard, start)
            fr = focus_ratio(heat, mask)
            if fr is not None:
                scores.setdefault(key, []).append(fr)

    ranked = sorted(scores.items(), key=lambda kv: -float(np.median(kv[1])))
    print()
    print("=" * 68)
    print(f"  attention-rollout variants, {examined} test images")
    print("  ranked by median focus ratio (1.0 = chance)")
    print("-" * 68)
    for key, vals in ranked:
        v = np.array(vals)
        print(f"  {key:30s} median {np.median(v):5.2f}  mean {v.mean():5.2f}"
              f"  >1: {(v > 1).mean():4.0%}")
    print("=" * 68)
    best = ranked[0]
    print(f"  best: {best[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
