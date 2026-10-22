"""Image-quality gate.

A screening tool that silently accepts an unusable photo is worse than one
that refuses it: the model will still emit a confident-looking risk score from
a blurred, dark or badly framed image. This module runs first and, on failure,
returns a plain-language reason the UI shows to the user as a retake prompt.

All thresholds live under ``quality:`` in the config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import cv2
import numpy as np

from src.config import Config
from src.segmentation import segment_lesion
from src.utils import ImageLike, get_logger, load_image_rgb, to_grayscale

logger = get_logger(__name__)


@dataclass
class QualityCheck:
    """Outcome of a single quality criterion.

    Attributes:
        name: Machine-readable check id, e.g. ``"blur"``.
        passed: Whether the image satisfied this criterion.
        value: The measured statistic.
        threshold: The threshold (or ``(lo, hi)`` range) it was compared to.
        message: Plain-language explanation shown to the user on failure.
    """

    name: str
    passed: bool
    value: float
    threshold: Any
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of this check."""
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "value": round(float(self.value), 4),
            "threshold": self.threshold,
            "message": self.message,
        }


@dataclass
class QualityReport:
    """Aggregate result of all quality checks on one photo."""

    passed: bool
    checks: List[QualityCheck] = field(default_factory=list)
    width: int = 0
    height: int = 0

    @property
    def failures(self) -> List[QualityCheck]:
        """The checks that did not pass."""
        return [c for c in self.checks if not c.passed]

    @property
    def reason(self) -> str:
        """One human-readable sentence summarising the outcome."""
        failed = self.failures
        if not failed:
            return "Photo quality looks good."
        if len(failed) == 1:
            return failed[0].message
        return " ".join(c.message for c in failed)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of the whole report."""
        return {
            "passed": bool(self.passed),
            "reason": self.reason,
            "width": self.width,
            "height": self.height,
            "checks": [c.to_dict() for c in self.checks],
        }


# ---------------------------------------------------------------------------
# Individual measurements
# ---------------------------------------------------------------------------
def laplacian_variance(rgb: np.ndarray) -> float:
    """Return the variance of the Laplacian -- the standard sharpness proxy.

    A sharp image has strong second derivatives at edges and therefore high
    variance; blur suppresses them. The absolute value depends on image size
    and content, so the threshold is a tunable config value, not a constant of
    nature.
    """
    gray = to_grayscale(rgb).astype(np.uint8)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_stats(rgb: np.ndarray) -> tuple[float, float]:
    """Return ``(mean, std)`` of the luma channel on a 0-255 scale."""
    gray = to_grayscale(rgb)
    return float(gray.mean()), float(gray.std())


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def assess_image_quality(
    image: ImageLike,
    cfg: Config,
    check_framing: bool = True,
) -> QualityReport:
    """Run every configured quality check against one photo.

    Args:
        image: Path, PIL image, NumPy array or file-like object.
        cfg: Loaded project config (reads the ``quality:`` section).
        check_framing: Whether to run lesion segmentation to verify the lesion
            is present and reasonably sized. Disable to skip the extra work
            when only sharpness/exposure matter.

    Returns:
        A :class:`QualityReport`; ``passed`` is ``True`` only if every check
        passed.
    """
    rgb = load_image_rgb(image)
    height, width = rgb.shape[:2]
    checks: List[QualityCheck] = []

    # --- resolution -------------------------------------------------------
    min_w = int(cfg.get("quality.min_width", 224))
    min_h = int(cfg.get("quality.min_height", 224))
    res_ok = width >= min_w and height >= min_h
    checks.append(QualityCheck(
        name="resolution",
        passed=res_ok,
        value=float(min(width, height)),
        threshold={"min_width": min_w, "min_height": min_h},
        message=(
            "Photo quality looks good."
            if res_ok else
            f"The photo is too small ({width}x{height} pixels). "
            f"Please take a photo of at least {min_w}x{min_h} and do not crop it down."
        ),
    ))

    # --- sharpness --------------------------------------------------------
    blur_min = float(cfg.get("quality.blur_laplacian_var_min", 60.0))
    lap_var = laplacian_variance(rgb)
    blur_ok = lap_var >= blur_min
    checks.append(QualityCheck(
        name="blur",
        passed=blur_ok,
        value=lap_var,
        threshold=blur_min,
        message=(
            "Photo quality looks good."
            if blur_ok else
            "The photo looks out of focus. Hold the phone steady, tap the screen "
            "to focus on the lesion, and take it again."
        ),
    ))

    # --- exposure ---------------------------------------------------------
    mean_luma, std_luma = brightness_stats(rgb)
    b_min = float(cfg.get("quality.brightness_min", 40.0))
    b_max = float(cfg.get("quality.brightness_max", 215.0))
    bright_ok = b_min <= mean_luma <= b_max
    if mean_luma < b_min:
        bright_msg = ("The photo is too dark. Move somewhere brighter or turn on "
                      "more light, then take it again.")
    elif mean_luma > b_max:
        bright_msg = ("The photo is over-exposed -- the flash or sunlight has washed "
                      "out the detail. Try again without direct flash.")
    else:
        bright_msg = "Photo quality looks good."
    checks.append(QualityCheck(
        name="brightness",
        passed=bright_ok,
        value=mean_luma,
        threshold={"min": b_min, "max": b_max},
        message=bright_msg,
    ))

    # --- contrast ---------------------------------------------------------
    # This check only catches a genuinely featureless frame. Whether a lesion is
    # actually visible is the framing check's job -- deliberately so, because a
    # pale lesion on even-toned skin is a legitimate low-contrast photo, and
    # such lesions are disproportionately benign.
    contrast_min = float(cfg.get("quality.contrast_std_min", 5.0))
    contrast_ok = std_luma >= contrast_min
    checks.append(QualityCheck(
        name="contrast",
        passed=contrast_ok,
        value=std_luma,
        threshold=contrast_min,
        message=(
            "Photo quality looks good."
            if contrast_ok else
            "The photo is almost completely blank -- we cannot see any detail in "
            "it at all. Check that the camera was pointed at the lesion."
        ),
    ))

    # --- framing ----------------------------------------------------------
    if check_framing:
        min_frac = float(cfg.get("quality.min_lesion_area_fraction", 0.01))
        max_frac = float(cfg.get("quality.max_lesion_area_fraction", 0.90))
        seg = segment_lesion(rgb, min_area_fraction=min_frac * 0.5,
                             max_area_fraction=min(0.99, max_frac + 0.05))
        frac = seg.area_fraction
        framing_ok = seg.found and (min_frac <= frac <= max_frac)
        if not seg.found:
            framing_msg = ("We could not clearly pick out a lesion in this photo. "
                           "Centre the lesion in the frame against plain skin and try again.")
        elif frac < min_frac:
            framing_msg = ("The lesion looks very small in the frame. Move the camera "
                           "closer so the lesion fills more of the photo.")
        elif frac > max_frac:
            framing_msg = ("The camera is too close -- we cannot see the edge of the "
                           "lesion or the surrounding skin. Move back slightly.")
        else:
            framing_msg = "Photo quality looks good."
        checks.append(QualityCheck(
            name="framing",
            passed=framing_ok,
            value=frac,
            threshold={"min": min_frac, "max": max_frac},
            message=framing_msg,
        ))

    report = QualityReport(
        passed=all(c.passed for c in checks),
        checks=checks,
        width=width,
        height=height,
    )
    if not report.passed:
        logger.info("Quality gate FAILED: %s",
                    ", ".join(c.name for c in report.failures))
    return report


def quality_summary_lines(report: QualityReport) -> List[str]:
    """Return one ``"[pass|FAIL] name: value"`` line per check, for logs/reports."""
    lines: List[str] = []
    for check in report.checks:
        status = "pass" if check.passed else "FAIL"
        lines.append(f"[{status}] {check.name}: {check.value:.2f}")
    return lines
