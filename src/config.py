"""Configuration loading.

Every module in this project takes its paths and hyperparameters from a YAML
config rather than from hardcoded constants. :func:`load_config` is the single
entry point; :class:`Config` is a thin dict wrapper that supports dotted keys
so callers can write ``cfg.get("train.batch_size")``.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Repository root = parent of the directory holding this file (src/).
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = REPO_ROOT / "configs" / "default.yaml"

_MISSING = object()


class Config:
    """Dict-backed configuration with dotted-key access and path resolution."""

    def __init__(self, data: Dict[str, Any], source: Optional[Path] = None) -> None:
        self._data: Dict[str, Any] = data
        self.source: Optional[Path] = source

    # -- access ------------------------------------------------------------
    def get(self, dotted_key: str, default: Any = _MISSING) -> Any:
        """Return the value at ``dotted_key`` (e.g. ``"train.epochs"``).

        Args:
            dotted_key: Dot-separated path into the nested config.
            default: Returned when the key is absent. If omitted, a missing
                key raises :class:`KeyError`.

        Raises:
            KeyError: If the key is missing and no default was supplied.
        """
        node: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                if default is _MISSING:
                    raise KeyError(f"Config key not found: {dotted_key!r}")
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, dotted_key: str) -> bool:
        return self.get(dotted_key, None) is not None

    def as_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the underlying dictionary."""
        return copy.deepcopy(self._data)

    # -- paths -------------------------------------------------------------
    def path(self, dotted_key: str, default: Any = _MISSING) -> Path:
        """Return a config value interpreted as a repo-root-relative path.

        Absolute values are returned unchanged, so a user can point the config
        at a dataset living outside the repository.
        """
        value = self.get(dotted_key, default)
        return self.resolve(value)

    @staticmethod
    def resolve(value: str | os.PathLike[str]) -> Path:
        """Resolve ``value`` against the repository root unless it is absolute."""
        p = Path(value)
        return p if p.is_absolute() else (REPO_ROOT / p)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config(source={self.source}, keys={list(self._data)})"


def load_config(path: Optional[str | os.PathLike[str]] = None,
                overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load the project configuration.

    Args:
        path: YAML file to load. Defaults to ``configs/default.yaml``. The
            ``SCD_CONFIG`` environment variable overrides the default and is
            itself overridden by an explicit ``path`` argument.
        overrides: Optional mapping of dotted keys to values applied after
            loading, e.g. ``{"backbone.backend": "fallback"}``. Useful for
            tests and CLI flags.

    Returns:
        A :class:`Config` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if path is None:
        path = os.environ.get("SCD_CONFIG", DEFAULT_CONFIG_PATH)
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    cfg = Config(data, source=cfg_path)
    for dotted_key, value in (overrides or {}).items():
        _set_dotted(data, dotted_key, value)
    return cfg


def _set_dotted(data: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested value in ``data``, creating intermediate dicts as needed."""
    parts = dotted_key.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise TypeError(f"Cannot descend into non-dict at {part!r} in {dotted_key!r}")
    node[parts[-1]] = value
