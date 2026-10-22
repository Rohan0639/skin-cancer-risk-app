"""Smoke-check the backbone layer.

Verifies that whichever backend is configured produces a fixed-size embedding
for a real image, and reports which one actually got used. Run this after
setting up PanDerm to confirm it is being picked up.

Usage::

    python src/check_backbone.py                      # honour the config
    python src/check_backbone.py --backend fallback   # force the fallback
    python src/check_backbone.py --backend panderm    # require PanDerm
    python src/check_backbone.py --image path/to.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backbone import BackboneUnavailable, build_backbone  # noqa: E402
from src.config import load_config  # noqa: E402
from src.utils import get_logger, load_image_rgb, resolve_device  # noqa: E402

logger = get_logger("check_backbone")


def _sample_image(explicit: Optional[str]) -> np.ndarray:
    """Return an image to test with: the given path, a prepared one, or noise."""
    if explicit:
        return load_image_rgb(explicit)

    cfg = load_config()
    for candidate in sorted((cfg.path("paths.processed_dir")).rglob("*.png"))[:1]:
        logger.info("Using prepared image: %s", candidate.name)
        return load_image_rgb(candidate)

    logger.info("No prepared images found; using a random noise image.")
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to a YAML config.")
    parser.add_argument("--backend", choices=["panderm", "fallback", "auto"], default=None,
                        help="Override backbone.backend for this run.")
    parser.add_argument("--image", default=None, help="Image to embed.")
    parser.add_argument("--batch", type=int, default=3, help="Batch size to test.")
    args = parser.parse_args()

    overrides = {"backbone.backend": args.backend} if args.backend else None
    cfg = load_config(args.config, overrides=overrides)
    device = resolve_device(cfg.get("project.device", "auto"))

    try:
        backbone = build_backbone(cfg).to(device)
    except BackboneUnavailable as exc:
        logger.error("Backbone unavailable: %s", exc)
        return 1

    rgb = _sample_image(args.image)
    transform = backbone.build_transform(train=False)

    from PIL import Image

    tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
    batch = tensor.repeat(args.batch, 1, 1, 1)

    with torch.no_grad():
        single = backbone(tensor)
        multi = backbone(batch)

    print()
    print("=" * 68)
    print(f"  backend requested : {cfg.get('backbone.backend')}")
    print(f"  backbone loaded   : {backbone.describe()}")
    print(f"  is_fallback       : {backbone.is_fallback}")
    print(f"  device            : {device}")
    print(f"  input tensor      : {tuple(tensor.shape)}")
    print(f"  single embedding  : {tuple(single.shape)}")
    print(f"  batch embedding   : {tuple(multi.shape)}")
    print(f"  declared dim      : {backbone.feature_dim}")
    print(f"  embedding norm    : {float(single.norm()):.4f}")
    print(f"  grad-cam layer(s) : "
          f"{[type(m).__name__ for m in backbone.gradcam_target_layers()]}")
    print("=" * 68)

    ok = True
    if single.shape != (1, backbone.feature_dim):
        logger.error("Single embedding shape %s != (1, %d)",
                     tuple(single.shape), backbone.feature_dim)
        ok = False
    if multi.shape != (args.batch, backbone.feature_dim):
        logger.error("Batch embedding shape %s != (%d, %d)",
                     tuple(multi.shape), args.batch, backbone.feature_dim)
        ok = False
    if not torch.isfinite(single).all():
        logger.error("Embedding contains NaN or Inf values.")
        ok = False

    print("  RESULT: PASS" if ok else "  RESULT: FAIL")
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
