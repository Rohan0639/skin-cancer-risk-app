"""Shared helpers: logging, seeding, device selection, image loading."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False

ImageLike = Union[str, os.PathLike, np.ndarray, Any]


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, idempotently."""
    global _configured
    if _configured:
        return
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt="%H:%M:%S")
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured."""
    setup_logging()
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass


def resolve_device(preference: str = "auto") -> "Any":
    """Return a ``torch.device`` honouring ``preference``.

    Args:
        preference: ``"auto"`` picks CUDA when available, otherwise CPU.
            ``"cpu"`` and ``"cuda"`` force the respective device.
    """
    import torch

    pref = (preference or "auto").lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        get_logger(__name__).warning("CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(pref)


def ensure_dir(path: Union[str, os.PathLike]) -> Path:
    """Create ``path`` as a directory if needed and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_image_rgb(source: ImageLike) -> np.ndarray:
    """Load any supported image source into an HxWx3 uint8 RGB array.

    Accepts a filesystem path, a PIL ``Image``, a NumPy array (RGB, RGBA or
    grayscale), or an open binary file object such as a Streamlit upload.

    Raises:
        FileNotFoundError: If a path is given that does not exist.
        ValueError: If the source cannot be interpreted as an image.
    """
    from PIL import Image

    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Unsupported array shape for an image: {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as img:
            return np.array(img.convert("RGB"), dtype=np.uint8)

    # PIL image or file-like object (e.g. Streamlit UploadedFile).
    if isinstance(source, Image.Image):
        return np.array(source.convert("RGB"), dtype=np.uint8)
    try:
        with Image.open(source) as img:
            return np.array(img.convert("RGB"), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001 - surface a clear message to callers
        raise ValueError(f"Could not read image from {type(source).__name__}: {exc}") from exc


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert an HxWx3 uint8 RGB array to an HxW float32 luma array (0-255)."""
    if rgb.ndim == 2:
        return rgb.astype(np.float32)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return rgb[:, :, :3].astype(np.float32) @ weights


def optional_import(module: str) -> Optional[Any]:
    """Import ``module``, returning ``None`` instead of raising if unavailable."""
    try:
        return __import__(module, fromlist=["__name__"])
    except ImportError:
        return None
