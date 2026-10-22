"""Unified dataset schema, metadata encoding and the torch ``Dataset``.

Both source datasets (HAM10000 and PAD-UFES-20) are normalised into one CSV
with the schema in :data:`UNIFIED_COLUMNS`, so a single model can be trained
across either or both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import Config
from src.utils import get_logger, load_image_rgb

logger = get_logger(__name__)

#: Required columns of the unified CSV, in canonical order.
UNIFIED_COLUMNS: List[str] = [
    "image_path",   # path to the image, relative to the repo root
    "age",          # float years, may be NaN
    "sex",          # one of metadata.sex_vocab
    "site",         # one of metadata.site_vocab
    "diagnosis",    # original dataset label, preserved for traceability
    "risk_label",   # one of labels.risk_classes
    "source",       # "ham10000" | "pad_ufes_20" | "synthetic_demo"
    "lesion_id",    # groups images of the same lesion; prevents split leakage
]


# ---------------------------------------------------------------------------
# Metadata normalisation
# ---------------------------------------------------------------------------
def normalise_sex(value: Any, vocab: Sequence[str]) -> str:
    """Map a raw sex/gender field onto the canonical vocabulary.

    Handles the spellings used by both datasets (``male``/``M``/``m``,
    ``female``/``F``) and returns ``"unknown"`` for anything else, including
    NaN and empty strings.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    text = str(value).strip().lower()
    if text in {"m", "male"}:
        return "male" if "male" in vocab else "unknown"
    if text in {"f", "female"}:
        return "female" if "female" in vocab else "unknown"
    return text if text in vocab else "unknown"


def normalise_site(value: Any, site_map: Dict[str, str], vocab: Sequence[str]) -> str:
    """Map a raw anatomical-site field onto the canonical vocabulary.

    ``site_map`` comes from the config and is matched case-insensitively, so a
    single map works for HAM10000's lowercase values and PAD-UFES-20's
    uppercase ones.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    raw = str(value).strip()
    lowered = {str(k).strip().lower(): v for k, v in site_map.items()}
    mapped = lowered.get(raw.lower())
    if mapped is None:
        mapped = raw.lower().replace(" ", "_")
    return mapped if mapped in vocab else "unknown"


def normalise_age(value: Any, cfg: Config) -> float:
    """Return a numeric age, substituting the configured impute value if absent."""
    impute = float(cfg.get("metadata.age.impute", 50.0))
    try:
        age = float(value)
    except (TypeError, ValueError):
        return impute
    if np.isnan(age) or age < 0:
        return impute
    return age


def encode_metadata(age: Any, sex: Any, site: Any, cfg: Config) -> np.ndarray:
    """Encode patient metadata into a fixed-length float32 feature vector.

    Layout: ``[scaled_age, age_missing_flag, one_hot(sex), one_hot(site)]``.
    The explicit missing-age flag lets the model distinguish an imputed age
    from a genuine one.

    Args:
        age: Age in years, or ``None``/NaN if unknown.
        sex: Raw sex string.
        site: Raw anatomical site string.
        cfg: Loaded project config.

    Returns:
        A 1-D float32 array of length :func:`metadata_dim`.
    """
    sex_vocab = list(cfg.get("metadata.sex_vocab"))
    site_vocab = list(cfg.get("metadata.site_vocab"))
    site_map = dict(cfg.get("metadata.ham10000_site_map", {}))
    site_map.update(dict(cfg.get("metadata.pad_ufes_site_map", {})))

    age_min = float(cfg.get("metadata.age.min", 0.0))
    age_max = float(cfg.get("metadata.age.max", 100.0))

    missing = 0.0
    try:
        age_val = float(age)
        if np.isnan(age_val):
            raise ValueError
    except (TypeError, ValueError):
        age_val = float(cfg.get("metadata.age.impute", 50.0))
        missing = 1.0

    span = max(age_max - age_min, 1e-6)
    scaled_age = float(np.clip((age_val - age_min) / span, 0.0, 1.0))

    sex_key = normalise_sex(sex, sex_vocab)
    site_key = normalise_site(site, site_map, site_vocab)

    sex_oh = np.zeros(len(sex_vocab), dtype=np.float32)
    sex_oh[sex_vocab.index(sex_key)] = 1.0
    site_oh = np.zeros(len(site_vocab), dtype=np.float32)
    site_oh[site_vocab.index(site_key)] = 1.0

    return np.concatenate(
        [np.array([scaled_age, missing], dtype=np.float32), sex_oh, site_oh]
    ).astype(np.float32)


def metadata_dim(cfg: Config) -> int:
    """Return the length of the vector produced by :func:`encode_metadata`."""
    return 2 + len(cfg.get("metadata.sex_vocab")) + len(cfg.get("metadata.site_vocab"))


def build_metadata_matrix(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Encode a dataframe's metadata into a ``(len(df), metadata_dim)`` matrix.

    Rows are in dataframe order, so the result aligns with a feature matrix
    extracted from the same dataframe.
    """
    if df.empty:
        return np.zeros((0, metadata_dim(cfg)), dtype=np.float32)
    return np.stack(
        [encode_metadata(r["age"], r["sex"], r["site"], cfg)
         for r in df.to_dict("records")]
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Unified CSV I/O
# ---------------------------------------------------------------------------
def write_unified_csv(rows: pd.DataFrame, out_path: Path) -> Path:
    """Validate and write a unified-schema dataframe to ``out_path``.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = [c for c in UNIFIED_COLUMNS if c not in rows.columns]
    if missing:
        raise ValueError(f"Unified CSV is missing required columns: {missing}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows[UNIFIED_COLUMNS].to_csv(out_path, index=False)
    logger.info("Wrote %d rows -> %s", len(rows), out_path)
    return out_path


def load_unified_csv(cfg: Config, path: Optional[Path] = None) -> pd.DataFrame:
    """Load the unified CSV and drop rows whose image file is missing.

    Raises:
        FileNotFoundError: If the CSV itself does not exist.
    """
    csv_path = Path(path) if path is not None else cfg.path("paths.unified_csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Unified dataset not found at {csv_path}. "
            "Run a script under data/ (e.g. prepare_ham10000.py) first."
        )
    df = pd.read_csv(csv_path)
    missing = [c for c in UNIFIED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")

    resolved = df["image_path"].map(lambda p: Config.resolve(p))
    exists = resolved.map(lambda p: Path(p).exists())
    if not exists.all():
        logger.warning("Dropping %d row(s) whose image file is missing.", int((~exists).sum()))
        df = df[exists].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No usable rows in {csv_path} -- all image files are missing.")
    return df


def group_aware_split(
    df: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test without letting one lesion span two splits.

    HAM10000 contains several images of the same physical lesion. Splitting
    per-image would leak near-duplicates into the test set and inflate scores,
    so splits are made over ``lesion_id`` groups instead.

    Returns:
        ``(train_df, val_df, test_df)``.
    """
    seed = int(cfg.get("project.seed", 42))
    val_frac = float(cfg.get("train.val_fraction", 0.15))
    test_frac = float(cfg.get("train.test_fraction", 0.15))

    rng = np.random.default_rng(seed)
    # Stratify groups by their majority risk label so rare classes survive.
    group_label = df.groupby("lesion_id")["risk_label"].agg(
        lambda s: s.value_counts().idxmax()
    )
    train_groups: List[Any] = []
    val_groups: List[Any] = []
    test_groups: List[Any] = []

    for label in sorted(group_label.unique()):
        groups = group_label[group_label == label].index.to_numpy()
        rng.shuffle(groups)
        n = len(groups)
        n_test = max(1, int(round(n * test_frac))) if n > 2 else 0
        n_val = max(1, int(round(n * val_frac))) if n > 2 else 0
        # Guarantee at least one training group per class.
        while n_test + n_val >= n and (n_test + n_val) > 0:
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
        test_groups.extend(groups[:n_test])
        val_groups.extend(groups[n_test:n_test + n_val])
        train_groups.extend(groups[n_test + n_val:])

    def subset(groups: Sequence[Any]) -> pd.DataFrame:
        return df[df["lesion_id"].isin(set(groups))].reset_index(drop=True)

    train_df, val_df, test_df = subset(train_groups), subset(val_groups), subset(test_groups)
    logger.info(
        "Split by lesion group -> train=%d  val=%d  test=%d images",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# torch Dataset
# ---------------------------------------------------------------------------
class LesionDataset:
    """Torch-style dataset yielding ``(image_tensor, metadata, label)`` tuples.

    Implemented without importing torch at module scope so that lightweight
    consumers (the data-prep scripts, tests of the encoding helpers) do not pay
    the torch import cost.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        transform: Optional[Any] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform
        self.classes: List[str] = list(cfg.get("labels.risk_classes"))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        import torch

        row = self.df.iloc[index]
        image = load_image_rgb(Config.resolve(row["image_path"]))

        if self.transform is not None:
            from PIL import Image

            tensor = self.transform(Image.fromarray(image))
        else:
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        meta = torch.from_numpy(
            encode_metadata(row["age"], row["sex"], row["site"], self.cfg)
        )
        label = torch.tensor(self.class_to_idx[row["risk_label"]], dtype=torch.long)
        return tensor, meta, label


def class_distribution(df: pd.DataFrame, classes: Sequence[str]) -> Dict[str, int]:
    """Return a ``{class_name: count}`` mapping including zero-count classes."""
    counts = df["risk_label"].value_counts().to_dict()
    return {c: int(counts.get(c, 0)) for c in classes}
