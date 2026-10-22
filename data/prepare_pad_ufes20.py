"""Prepare PAD-UFES-20 into the project's unified CSV schema.

PAD-UFES-20 (Pacheco et al., *Data in Brief* 2020) contains 2,298 **smartphone
clinical** images of 1,373 lesions with rich patient metadata. It is the closer
match to how this tool is actually used -- HAM10000 is dermoscopic, taken
through a contact lens, which a patient does not have.

Download: https://data.mendeley.com/datasets/zr7vgbcyr2/1

Expected input layout under ``paths.pad_ufes_raw`` (images are located by
filename, so any nesting works)::

    data/raw/pad_ufes_20/
      metadata.csv
      images/imgs_part_1/*.png
      images/imgs_part_2/*.png
      images/imgs_part_3/*.png

Usage::

    python data/prepare_pad_ufes20.py --per-class 150    # fast local subset
    python data/prepare_pad_ufes20.py --full             # everything
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

logger = get_logger("prepare_pad_ufes20")

SOURCE_NAME = "pad_ufes_20"


def find_metadata_csv(root: Path) -> Optional[Path]:
    """Locate the PAD-UFES-20 metadata CSV anywhere under ``root``."""
    for candidate in root.rglob("metadata.csv"):
        if candidate.is_file():
            return candidate
    for candidate in root.rglob("*.csv"):
        try:
            header = set(pd.read_csv(candidate, nrows=0).columns)
        except Exception:  # noqa: BLE001
            continue
        if {"img_id", "diagnostic"}.issubset(header):
            return candidate
    return None


def prepare(cfg: Config, per_class: Optional[int], limit: Optional[int],
            merge: bool = True) -> Optional[Path]:
    """Build ``data/processed/pad_ufes_20.csv`` from the raw download.

    Args:
        cfg: Loaded project config.
        per_class: Cap on rows per risk class (``None`` = keep all).
        limit: Cap on total rows after per-class sampling.
        merge: Whether to rebuild the merged unified CSV afterwards.

    Returns:
        Path to the written per-dataset CSV, or ``None`` if the raw data is
        absent.
    """
    raw_root = cfg.path("paths.pad_ufes_raw")
    if not raw_root.exists():
        logger.error(
            "PAD-UFES-20 not found at %s.\n"
            "  Download it from https://data.mendeley.com/datasets/zr7vgbcyr2/1 "
            "and unzip it there.",
            raw_root,
        )
        return None

    meta_csv = find_metadata_csv(raw_root)
    if meta_csv is None:
        logger.error("No metadata.csv found under %s", raw_root)
        return None
    logger.info("Reading metadata: %s", meta_csv)

    meta = pd.read_csv(meta_csv)
    required = {"img_id", "diagnostic"}
    if not required.issubset(meta.columns):
        logger.error("Metadata is missing required columns %s", required - set(meta.columns))
        return None

    images = index_images(raw_root)
    if not images:
        logger.error("No image files found under %s", raw_root)
        return None

    risk_map = {str(k).upper(): v for k, v in dict(cfg.get("labels.pad_ufes_map")).items()}
    site_map = dict(cfg.get("metadata.pad_ufes_site_map"))
    sex_vocab = list(cfg.get("metadata.sex_vocab"))
    site_vocab = list(cfg.get("metadata.site_vocab"))

    rows = []
    unmapped_dx: set[str] = set()
    missing_images = 0

    for record in meta.to_dict("records"):
        img_id = str(record["img_id"]).strip()
        dx = str(record["diagnostic"]).strip().upper()

        risk = risk_map.get(dx)
        if risk is None:
            unmapped_dx.add(dx)
            continue

        # Try the full filename first, then the stem.
        path = images.get(img_id.lower()) or images.get(Path(img_id).stem.lower())
        if path is None:
            missing_images += 1
            continue

        rows.append({
            "image_path": relative_to_repo(path),
            "age": normalise_age(record.get("age"), cfg),
            # PAD-UFES-20 calls the column "gender".
            "sex": normalise_sex(record.get("gender", record.get("sex")), sex_vocab),
            "site": normalise_site(record.get("region"), site_map, site_vocab),
            "diagnosis": dx,
            "risk_label": risk,
            "source": SOURCE_NAME,
            "lesion_id": str(record.get("lesion_id") or f"{SOURCE_NAME}_{Path(img_id).stem}"),
        })

    if unmapped_dx:
        logger.warning("Skipped unmapped diagnoses %s -- add them to labels.pad_ufes_map.",
                       sorted(unmapped_dx))
    if missing_images:
        logger.warning("Skipped %d metadata row(s) with no matching image file.", missing_images)
    if not rows:
        logger.error("No usable PAD-UFES-20 rows were produced.")
        return None

    df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
    # lesion_id is only unique within the dataset; prefix it to stay unique
    # after merging with HAM10000.
    df["lesion_id"] = SOURCE_NAME + "_" + df["lesion_id"].astype(str)
    df = subsample(df, per_class, limit, int(cfg.get("project.seed", 42)))
    summarise(df, "PAD-UFES-20")

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
