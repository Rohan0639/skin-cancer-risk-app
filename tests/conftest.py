"""Shared pytest fixtures.

Two speed rules keep the suite usable:

* Unit tests use ``fast_cfg``, which builds a small **randomly initialised**
  ResNet-18. It needs no network access and constructs in well under a second.
  These tests check shapes, contracts and edge cases -- none of which depend on
  the weights being meaningful.
* The end-to-end tests use ``analyzer``, a session-scoped fixture that builds
  the real configured backbone once and shares it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import Config, load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> Config:
    """The real project configuration."""
    return load_config()


@pytest.fixture(scope="session")
def fast_cfg() -> Config:
    """A config using a tiny, un-pretrained backbone -- fast and offline.

    ``pretrained: false`` matters: it keeps unit tests from hitting the network
    and makes them deterministic.
    """
    return load_config(overrides={
        "backbone.backend": "fallback",
        "backbone.fallback.model_name": "resnet18",
        "backbone.fallback.pretrained": False,
        "backbone.fallback.feature_dim": None,
    })


@pytest.fixture(scope="session")
def make_lesion_image() -> Callable[..., np.ndarray]:
    """Return a factory drawing a synthetic lesion photo.

    The factory signature is ``(size=256, radius=0.2, roughness=0.0,
    tones=1, seed=0)``. Tests use it to build images with known properties --
    a perfectly round single-colour blob, a ragged multicoloured one -- so the
    ABCDE assertions can be about ground truth rather than about whatever the
    dataset happened to contain.
    """
    sys.path.insert(0, str(REPO_ROOT / "data"))
    from make_demo_dataset import _lesion_mask, _paint_lesion, _skin_background

    def factory(size: int = 256, radius: float = 0.20, roughness: float = 0.0,
                tones: int = 1, asymmetry: float = 0.0, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        background = _skin_background(size, rng)
        mask = _lesion_mask(size, (radius, radius + 0.001), roughness, asymmetry, rng)
        return _paint_lesion(background, mask, tones, rng)

    return factory


@pytest.fixture(scope="session")
def round_lesion(make_lesion_image) -> np.ndarray:
    """A smooth, symmetric, single-colour lesion: nothing should be flagged."""
    return make_lesion_image(roughness=0.0, tones=1, asymmetry=0.0, seed=1)


@pytest.fixture(scope="session")
def ragged_lesion(make_lesion_image) -> np.ndarray:
    """A ragged, asymmetric, multicoloured lesion: A, B and C should flag."""
    return make_lesion_image(radius=0.30, roughness=0.30, tones=3,
                             asymmetry=0.28, seed=2)


@pytest.fixture(scope="session")
def analyzer():
    """A session-wide :class:`LesionAnalyzer` using the real configured backbone.

    Built with ``strict=False`` so the suite runs whether or not a trained
    checkpoint exists; the end-to-end tests assert on mechanics (shapes, report
    structure, no exceptions), never on predictive accuracy.
    """
    from src.pipeline import LesionAnalyzer

    return LesionAnalyzer(strict=False)


@pytest.fixture
def tmp_outputs(tmp_path: Path) -> Path:
    """A temporary directory for artefacts a test writes."""
    out = tmp_path / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out
