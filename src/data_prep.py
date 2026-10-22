"""Helpers shared by the dataset preparation scripts under ``data/``."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import Config, REPO_ROOT
from src.dataset import UNIFIED_COLUMNS, write_unified_csv
from src.utils import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def index_images(root: Path) -> Dict[str, Path]:
    """Recursively index image files under ``root`` keyed by lowercase stem.

    Both datasets ship their images split across several sibling folders whose
    names vary between mirrors, so we index by filename rather than assuming a
    directory layout.

    Returns:
        Mapping from lowercase filename stem (e.g. ``"isic_0027419"``) to the
        file's path. Also includes full lowercase filenames as keys, so a
        metadata column containing ``"PAT_8_15_820.png"`` resolves too.
    """
    index: Dict[str, Path] = {}
    if not root.exists():
        return index
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.stem.lower(), path)
            index.setdefault(path.name.lower(), path)
    logger.info("Indexed %d image file(s) under %s", len(set(index.values())), root)
    return index


def relative_to_repo(path: Path) -> str:
    """Return ``path`` relative to the repo root when possible, else absolute.

    Storing relative paths keeps the CSVs portable between machines.
    """
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def subsample(
    df: pd.DataFrame,
    per_class: Optional[int],
    limit: Optional[int],
    seed: int,
) -> pd.DataFrame:
    """Take a reproducible subset for fast local iteration.

    Args:
        df: Unified-schema dataframe.
        per_class: If set, keep at most this many rows per risk class. This is
            the preferred knob -- it keeps rare classes present, which a plain
            random ``limit`` would not on a heavily imbalanced dataset.
        limit: If set, cap the total number of rows after per-class sampling.
        seed: RNG seed.

    Returns:
        The (possibly reduced) dataframe, shuffled.
    """
    out = df
    if per_class is not None and per_class > 0:
        out = (
            out.groupby("risk_label", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), per_class), random_state=seed),
                   include_groups=True)
            .reset_index(drop=True)
        )
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if limit is not None and limit > 0:
        out = out.head(limit).reset_index(drop=True)
    return out


def rebuild_unified(cfg: Config) -> Optional[Path]:
    """Merge every per-dataset CSV in ``paths.processed_dir`` into the unified CSV.

    Each ``prepare_*.py`` script writes ``data/processed/<source>.csv``; this
    concatenates all of them so training can run on one dataset or both without
    any code change.

    Returns:
        The unified CSV path, or ``None`` if no per-dataset CSVs exist yet.
    """
    processed = cfg.path("paths.processed_dir")
    unified_path = cfg.path("paths.unified_csv")
    parts: List[pd.DataFrame] = []
    for csv_path in sorted(processed.glob("*.csv")):
        if csv_path.resolve() == unified_path.resolve():
            continue
        try:
            part = pd.read_csv(csv_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping unreadable %s (%s)", csv_path.name, exc)
            continue
        if all(c in part.columns for c in UNIFIED_COLUMNS):
            parts.append(part[UNIFIED_COLUMNS])
            logger.info("Merging %s (%d rows)", csv_path.name, len(part))
        else:
            logger.warning("Skipping %s -- not in unified schema.", csv_path.name)

    if not parts:
        logger.warning("No per-dataset CSVs found in %s; nothing to merge.", processed)
        return None

    merged = pd.concat(parts, ignore_index=True)
    merged = merged.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    write_unified_csv(merged, unified_path)
    return unified_path


def summarise(df: pd.DataFrame, name: str) -> None:
    """Log a short class/metadata breakdown of a prepared dataset."""
    logger.info("--- %s: %d rows ---", name, len(df))
    for label, count in df["risk_label"].value_counts().items():
        logger.info("  risk=%-9s %5d (%.1f%%)", label, count, 100.0 * count / len(df))
    logger.info("  sites : %s", dict(df["site"].value_counts()))
    ages = pd.to_numeric(df["age"], errors="coerce")
    if ages.notna().any():
        logger.info("  age   : mean=%.1f  min=%.0f  max=%.0f",
                    float(np.nanmean(ages)), float(np.nanmin(ages)), float(np.nanmax(ages)))
