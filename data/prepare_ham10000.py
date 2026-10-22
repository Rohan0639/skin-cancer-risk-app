"""Prepare HAM10000 into the project's unified CSV schema.

HAM10000 (Tschandl, Rosendahl & Kittler, *Scientific Data* 2018,
doi:10.1038/sdata.2018.161) is a set of 10,015 dermoscopic images with 7-class
diagnosis labels and light patient metadata.

Expected input layout under ``paths.ham10000_raw`` (folder names vary between
mirrors -- images are located by filename, so any nesting works)::

    data/raw/ham10000/
      HAM10000_metadata.csv
      HAM10000_images_part_1/*.jpg
      HAM10000_images_part_2/*.jpg

Usage::

    python data/prepare_ham10000.py --per-class 150     # fast local subset
    python data/prepare_ham10000.py --full              # everything
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config  # noqa: E402
from src.data_prep import (  # noqa: E402
    index_images,
    rebuild_unified,
    relative_to_repo,
    subsample,
    summarise,
)
from src.dataset import (  # noqa: E402
    UNIFIED_COLUMNS,
    normalise_age,
    normalise_sex,
    normalise_site,
    write_unified_csv,
)
from src.utils import get_logger  # noqa: E402

logger = get_logger("prepare_ham10000")

SOURCE_NAME = "ham10000"
METADATA_CANDIDATES = ("HAM10000_metadata.csv", "HAM10000_metadata", "metadata.csv")


def find_metadata_csv(root: Path) -> Optional[Path]:
    """Locate the HAM10000 metadata CSV anywhere under ``root``."""
    for name in METADATA_CANDIDATES:
        for candidate in root.rglob(name):
            if candidate.is_file():
                return candidate
    for candidate in root.rglob("*.csv"):
        try:
            header = pd.read_csv(candidate, nrows=0).columns
        except Exception:  # noqa: BLE001
            continue
        if {"image_id", "dx"}.issubset(set(header)):
            return candidate
    return None


def prepare(cfg: Config, per_class: Optional[int], limit: Optional[int],
            merge: bool = True) -> Optional[Path]:
    """Build ``data/processed/ham10000.csv`` from the raw download.

    Args:
        cfg: Loaded project config.
        per_class: Cap on rows per risk class (``None`` = keep all).
        limit: Cap on total rows after per-class sampling.
        merge: Whether to rebuild the merged unified CSV afterwards.

    Returns:
        Path to the written per-dataset CSV, or ``None`` if the raw data is
        absent (a warning is logged rather than raising, so the script is safe
        to run before downloading).
    """
    raw_root = cfg.path("paths.ham10000_raw")
    if not raw_root.exists():
        logger.error(
            "HAM10000 not found at %s.\n"
            "  Download it from the ISIC archive or Kaggle "
            "('kmader/skin-cancer-mnist-ham10000') and unzip it there.",
            raw_root,
        )
        return None

    meta_csv = find_metadata_csv(raw_root)
    if meta_csv is None:
        logger.error("No HAM10000_metadata.csv found under %s", raw_root)
        return None
    logger.info("Reading metadata: %s", meta_csv)

    meta = pd.read_csv(meta_csv)
    required = {"image_id", "dx"}
    if not required.issubset(meta.columns):
        logger.error("Metadata is missing required columns %s", required - set(meta.columns))
        return None

    images = index_images(raw_root)
    if not images:
        logger.error("No image files found under %s", raw_root)
        return None

    risk_map = dict(cfg.get("labels.ham10000_map"))
    site_map = dict(cfg.get("metadata.ham10000_site_map"))
    sex_vocab = list(cfg.get("metadata.sex_vocab"))
    site_vocab = list(cfg.get("metadata.site_vocab"))

    rows = []
    unmapped_dx: set[str] = set()
    missing_images = 0

    for record in meta.to_dict("records"):
        image_id = str(record["image_id"]).strip()
        dx = str(record["dx"]).strip().lower()

        risk = risk_map.get(dx)
        if risk is None:
            unmapped_dx.add(dx)
            continue

        path = images.get(image_id.lower())
        if path is None:
            missing_images += 1
            continue

        rows.append({
            "image_path": relative_to_repo(path),
            "age": normalise_age(record.get("age"), cfg),
            "sex": normalise_sex(record.get("sex"), sex_vocab),
            "site": normalise_site(record.get("localization"), site_map, site_vocab),
            "diagnosis": dx,
            "risk_label": risk,
            "source": SOURCE_NAME,
            # Fall back to the image id so every row has a group key.
            "lesion_id": str(record.get("lesion_id") or f"{SOURCE_NAME}_{image_id}"),
        })

    if unmapped_dx:
        logger.warning("Skipped unmapped diagnoses %s -- add them to labels.ham10000_map.",
                       sorted(unmapped_dx))
    if missing_images:
        logger.warning("Skipped %d metadata row(s) with no matching image file.", missing_images)
    if not rows:
        logger.error("No usable HAM10000 rows were produced.")
        return None

    df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
    df = subsample(df, per_class, limit, int(cfg.get("project.seed", 42)))
    summarise(df, "HAM10000")

    out_path = cfg.path("paths.processed_dir") / f"{SOURCE_NAME}.csv"
    write_unified_csv(df, out_path)
    if merge:
        rebuild_unified(cfg)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to a YAML config.")
    parser.add_argument("--per-class", type=int, default=200,
                        help="Max rows per risk class (default 200; use --full for all).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Hard cap on total rows after per-class sampling.")
    parser.add_argument("--full", action="store_true",
                        help="Use the entire dataset (overrides --per-class/--limit).")
    parser.add_argument("--no-merge", action="store_true",
                        help="Skip rebuilding data/processed/unified.csv.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    per_class = None if args.full else args.per_class
    limit = None if args.full else args.limit

    result = prepare(cfg, per_class, limit, merge=not args.no_merge)
    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
