"""Generate a small SYNTHETIC lesion dataset for smoke-testing the pipeline.

    ############################################################
    #  THIS IS NOT REAL MEDICAL DATA.                          #
    #  These are procedurally drawn blobs. Any metric measured  #
    #  on them says something about whether the CODE runs, and  #
    #  NOTHING about clinical performance. Never quote a number #
    #  from this dataset as a result.                           #
    ############################################################

Its purpose is to let ``train.py``, ``evaluate.py``, the explainers and the
Streamlit app be exercised end-to-end on a laptop without first downloading
~5 GB of HAM10000 and PAD-UFES-20. Rows are tagged ``source=synthetic_demo``
so every downstream report can flag them.

Lesions are drawn with class-correlated morphology, which also gives the ABCDE
heuristics something meaningful to measure:

===========  ==========  ================  ==================  =========
risk         symmetry    border            colour              size
===========  ==========  ================  ==================  =========
Low          round       smooth            single tan/brown    small
Moderate     mild skew   mildly ragged     two-tone            medium
High         asymmetric  ragged / notched  3+ tones inc. dark  large
===========  ==========  ================  ==================  =========

Usage::

    python data/make_demo_dataset.py --per-class 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config  # noqa: E402
from src.data_prep import rebuild_unified, relative_to_repo, summarise  # noqa: E402
from src.dataset import UNIFIED_COLUMNS, write_unified_csv  # noqa: E402
from src.utils import ensure_dir, get_logger  # noqa: E402

logger = get_logger("make_demo_dataset")

SOURCE_NAME = "synthetic_demo"

#: Per-class drawing parameters: (radius range, border roughness, n colour tones,
#: asymmetry strength).
CLASS_STYLE: Dict[str, Tuple[Tuple[float, float], float, int, float]] = {
    "Low":      ((0.14, 0.22), 0.04, 1, 0.02),
    "Moderate": ((0.20, 0.30), 0.13, 2, 0.10),
    "High":     ((0.26, 0.40), 0.28, 3, 0.24),
}

#: Plausible lesion tones in RGB, ordered light -> dark.
TONES: List[Tuple[int, int, int]] = [
    (196, 152, 116),   # light tan
    (150, 104, 72),    # brown
    (104, 66, 44),     # dark brown
    (58, 38, 30),      # near-black
    (128, 60, 60),     # reddish
    (74, 66, 104),     # blue-grey
]


def _skin_background(size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a mottled skin-toned background with soft lighting variation."""
    base = np.array(
        [rng.uniform(198, 236), rng.uniform(162, 196), rng.uniform(140, 172)],
        dtype=np.float32,
    )
    img = np.tile(base, (size, size, 1))

    # Low-frequency illumination gradient (a phone flash is never uniform).
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    grad = 1.0 + 0.18 * (rng.uniform(-1, 1) * (xx - 0.5) + rng.uniform(-1, 1) * (yy - 0.5))
    img *= grad[..., None]

    # Fine skin texture / sensor noise.
    img += rng.normal(0.0, 4.0, img.shape).astype(np.float32)
    return img


def _lesion_mask(size: int, radii: Tuple[float, float], roughness: float,
                 asymmetry: float, rng: np.random.Generator) -> np.ndarray:
    """Build a soft-edged lesion mask using a radially perturbed disc.

    The radius at each angle is a smooth random function, so higher
    ``roughness`` yields a genuinely notched contour rather than added noise.

    Returns:
        An HxW float32 mask in [0, 1].
    """
    cy = size / 2 + rng.uniform(-0.04, 0.04) * size
    cx = size / 2 + rng.uniform(-0.04, 0.04) * size
    base_r = rng.uniform(*radii) * size

    # Radius as a function of angle: a few low-order harmonics.
    n_angles = 512
    theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    radius = np.ones(n_angles, dtype=np.float32)
    for harmonic in (2, 3, 5, 7):
        radius += roughness * rng.uniform(-1, 1) * np.cos(harmonic * theta + rng.uniform(0, 6.28))
    # Asymmetry: stretch one half of the lesion.
    radius *= 1.0 + asymmetry * np.cos(theta + rng.uniform(0, 6.28))
    radius = np.clip(radius, 0.35, 1.8) * base_r

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dy, dx = yy - cy, xx - cx
    dist = np.sqrt(dy * dy + dx * dx)
    ang = np.mod(np.arctan2(dy, dx), 2 * np.pi)

    idx = np.clip((ang / (2 * np.pi) * n_angles).astype(np.int32), 0, n_angles - 1)
    boundary = radius[idx]

    # Soft edge -- real lesions do not have a 1-pixel transition.
    edge = max(2.0, 0.05 * base_r)
    return np.clip((boundary - dist) / edge + 0.5, 0.0, 1.0).astype(np.float32)


def _paint_lesion(img: np.ndarray, mask: np.ndarray, n_tones: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Composite ``n_tones`` colour regions inside ``mask`` onto ``img``.

    Single-tone (low-risk) lesions are restricted to the natural tan/brown end
    of the palette; the alarming red and blue-grey tones only appear on
    multi-tone lesions, matching the ABCDE "colour variation" cue.
    """
    size = img.shape[0]
    palette = TONES[:3] if n_tones == 1 else TONES
    tones = [palette[i] for i in rng.choice(len(palette), size=n_tones, replace=False)]

    out = img.copy()
    if n_tones == 1:
        colour = np.array(tones[0], dtype=np.float32)
        out = out * (1 - mask[..., None]) + colour * mask[..., None]
    else:
        # Assign each pixel to the nearest of n_tones random seed points, so
        # colour regions form contiguous patches rather than speckle.
        seeds = rng.uniform(0.25, 0.75, size=(n_tones, 2)) * size
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        dists = np.stack(
            [np.sqrt((yy - sy) ** 2 + (xx - sx) ** 2) for sy, sx in seeds], axis=0
        )
        owner = np.argmin(dists, axis=0)
        for i, tone in enumerate(tones):
            region = (mask * (owner == i)).astype(np.float32)
            colour = np.array(tone, dtype=np.float32)
            out = out * (1 - region[..., None]) + colour * region[..., None]

    out += rng.normal(0.0, 3.0, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def render_lesion(risk: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Render one synthetic lesion photo for the given risk class."""
    radii, roughness, n_tones, asymmetry = CLASS_STYLE[risk]
    background = _skin_background(size, rng)
    mask = _lesion_mask(size, radii, roughness, asymmetry, rng)
    return _paint_lesion(background, mask, n_tones, rng)


def rephotograph(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate re-photographing the *same* lesion on a different day.

    Applies only nuisance variation -- camera angle, distance, and lighting
    colour -- and never alters the lesion itself. This is what makes the demo
    dataset useful for two things at once: the change-detection module should
    report "no change" across such a pair, and the group-aware split has
    genuine near-duplicates that would leak between train and test if images
    were split individually.
    """
    size = image.shape[0]
    angle = float(rng.uniform(-15, 15))
    scale = float(rng.uniform(0.88, 1.12))
    shift = rng.uniform(-0.03, 0.03, size=2) * size

    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
    matrix[0, 2] += shift[0]
    matrix[1, 2] += shift[1]
    out = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    # Different bulb / time of day: a colour cast plus an exposure change.
    cast = rng.uniform(0.90, 1.10, size=3).astype(np.float32)
    exposure = float(rng.uniform(0.88, 1.12))
    out = out.astype(np.float32) * cast * exposure
    out += rng.normal(0.0, 3.0, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def generate(cfg: Config, per_class: int, size: int, merge: bool = True) -> Path:
    """Generate the synthetic dataset and write its unified CSV.

    Args:
        cfg: Loaded project config.
        per_class: Number of images to render per risk class.
        size: Output image edge length in pixels.
        merge: Whether to rebuild ``data/processed/unified.csv`` afterwards.

    Returns:
        Path to ``data/processed/synthetic_demo.csv``.
    """
    from PIL import Image

    seed = int(cfg.get("project.seed", 42))
    rng = np.random.default_rng(seed)
    classes = list(cfg.get("labels.risk_classes"))
    sex_vocab = [s for s in cfg.get("metadata.sex_vocab") if s != "unknown"]
    site_vocab = [s for s in cfg.get("metadata.site_vocab") if s != "unknown"]

    img_dir = ensure_dir(cfg.path("paths.processed_dir") / "synthetic_images")
    rows = []

    for risk in classes:
        for i in range(per_class):
            # Two photos per lesion, so the group-aware split has real groups
            # to work with (mirrors HAM10000's repeated lesion images).
            lesion_id = f"{SOURCE_NAME}_{risk.lower()}_{i:04d}"
            age = float(np.clip(rng.normal(62 if risk == "High" else 45, 15), 8, 95))
            sex = str(rng.choice(sex_vocab))
            site = str(rng.choice(site_vocab))

            # Shot 0 is the lesion; shot 1 is the SAME lesion re-photographed
            # under different lighting and from a slightly different angle.
            base = render_lesion(risk, size, rng)
            for shot in range(2):
                array = base if shot == 0 else rephotograph(base, rng)
                out_path = img_dir / f"{lesion_id}_{shot}.png"
                Image.fromarray(array).save(out_path)
                rows.append({
                    "image_path": relative_to_repo(out_path),
                    "age": round(age, 1),
                    "sex": sex,
                    "site": site,
                    "diagnosis": f"synthetic_{risk.lower()}",
                    "risk_label": risk,
                    "source": SOURCE_NAME,
                    "lesion_id": lesion_id,
                })

    df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    summarise(df, "SYNTHETIC DEMO (not real data)")

    out_csv = cfg.path("paths.processed_dir") / f"{SOURCE_NAME}.csv"
    write_unified_csv(df, out_csv)
    logger.warning(
        "Generated SYNTHETIC data. Metrics measured on it are smoke-test "
        "output only and must never be reported as clinical performance."
    )
    if merge:
        rebuild_unified(cfg)
    return out_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to a YAML config.")
    parser.add_argument("--per-class", type=int, default=40,
                        help="Lesions per risk class (2 photos each). Default 40.")
    parser.add_argument("--size", type=int, default=256, help="Image edge length in px.")
    parser.add_argument("--no-merge", action="store_true",
                        help="Skip rebuilding data/processed/unified.csv.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate(cfg, args.per_class, args.size, merge=not args.no_merge)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
