"""Comparing two photos of the same lesion over time.

"Evolving" is the E in ABCDE and arguably the single most useful signal a
patient can provide, because it needs no clinical training -- just an old photo.
This module compares a previous photo with a current one and reports whether
the lesion has meaningfully changed.

**The honest caveat, which the output states plainly:** two phone photos taken
months apart differ in distance, angle, lighting and white balance, and none of
that is the lesion changing. So the comparison is built to be as robust to
those nuisances as possible:

* Both lesions are **cropped to their own bounding box and resized to a common
  size**, which removes camera distance and framing from every shape measure.
* Shape signals (asymmetry, border irregularity) are scale-invariant by
  construction.
* Colour is compared in CIELAB after a grey-world white-balance normalisation,
  so a warmer bulb does not read as "the lesion turned red".
* Absolute physical growth is **not** claimed, because nothing in two unscaled
  photos can establish it. Relative area change is reported only as a weak,
  clearly-labelled signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.config import Config
from src.explain.abcde import asymmetry_score, border_irregularity
from src.segmentation import LesionMask, segment_lesion
from src.utils import get_logger

logger = get_logger(__name__)

#: Size that both lesion crops are resampled to before comparison.
ALIGNED_SIZE = 224


@dataclass
class ChangeSignal:
    """One measured difference between the two photos.

    Attributes:
        name: Signal id, e.g. ``"colour"``.
        value: Measured magnitude of change.
        threshold: Value above which this signal counts as changed.
        changed: Whether ``value`` exceeded ``threshold``.
        weight: Contribution to the overall decision. Signals robust to camera
            nuisances carry more weight than fragile ones.
        description: Plain-language rendering of this signal.
    """

    name: str
    value: float
    threshold: float
    changed: bool
    weight: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "name": self.name,
            "value": round(float(self.value), 4),
            "threshold": round(float(self.threshold), 4),
            "changed": bool(self.changed),
            "weight": round(float(self.weight), 3),
            "description": self.description,
        }


@dataclass
class ChangeResult:
    """Outcome of comparing a previous and a current photo.

    Attributes:
        changed: Whether the lesion changed meaningfully.
        score: Weighted evidence for change, in [0, 1].
        threshold: Score above which ``changed`` is ``True``.
        signals: Every individual measurement.
        reason: One or two sentences explaining the verdict.
        reliable: ``False`` when either photo segmented poorly, meaning the
            comparison should not be trusted.
    """

    changed: bool
    score: float
    threshold: float
    signals: List[ChangeSignal] = field(default_factory=list)
    reason: str = ""
    reliable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view, also consumed by the ABCDE 'E'."""
        return {
            "changed": bool(self.changed),
            "score": round(float(self.score), 4),
            "threshold": round(float(self.threshold), 4),
            "reason": self.reason,
            "reliable": bool(self.reliable),
            "signals": [s.to_dict() for s in self.signals],
        }


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def align_lesion_crop(
    rgb: np.ndarray,
    mask: LesionMask,
    size: int = ALIGNED_SIZE,
    padding: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop to the lesion's bounding box (plus padding) and resize to ``size``.

    This is what makes the two photos comparable: after this, camera distance
    and where the lesion sat in the frame no longer affect any measurement.

    Args:
        rgb: HxWx3 uint8 RGB image.
        mask: The lesion's segmentation.
        size: Output edge length.
        padding: Fraction of the bounding box to add on each side, so the
            lesion border and a little surrounding skin are both included.

    Returns:
        ``(cropped_rgb, cropped_mask)``, both ``size`` x ``size``.
    """
    h, w = rgb.shape[:2]
    y0, x0, y1, x1 = mask.bbox
    pad_y = int((y1 - y0) * padding)
    pad_x = int((x1 - x0) * padding)
    y0 = max(0, y0 - pad_y)
    x0 = max(0, x0 - pad_x)
    y1 = min(h, y1 + pad_y)
    x1 = min(w, x1 + pad_x)

    if y1 <= y0 or x1 <= x0:  # degenerate box: fall back to the whole frame
        y0, x0, y1, x1 = 0, 0, h, w

    crop_rgb = cv2.resize(rgb[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
    crop_mask = cv2.resize(mask.mask[y0:y1, x0:x1], (size, size),
                           interpolation=cv2.INTER_NEAREST)
    return crop_rgb, crop_mask


def grey_world_balance(rgb: np.ndarray) -> np.ndarray:
    """Apply grey-world white balance so lighting colour does not fake a change.

    Grey-world assumes the average scene colour is neutral grey and rescales
    each channel to make it so. For skin photos this largely cancels the
    difference between, say, a warm indoor bulb and daylight.

    Args:
        rgb: HxWx3 uint8 RGB image.

    Returns:
        A white-balanced HxWx3 uint8 RGB image.
    """
    balanced = rgb.astype(np.float32)
    means = balanced.reshape(-1, 3).mean(axis=0)
    grey = float(means.mean())
    for channel in range(3):
        if means[channel] > 1e-6:
            balanced[:, :, channel] *= grey / means[channel]
    return np.clip(balanced, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Individual comparisons
# ---------------------------------------------------------------------------
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine distance (``1 - cosine similarity``) between vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm < 1e-12:
        return 0.0
    return float(np.clip(1.0 - float(np.dot(a, b)) / norm, 0.0, 2.0))


def mean_colour_shift(rgb_a: np.ndarray, mask_a: np.ndarray,
                      rgb_b: np.ndarray, mask_b: np.ndarray) -> float:
    """Return the normalised CIELAB distance between the two lesions' mean colour.

    Both images are white-balanced first. The result is scaled by 100, roughly
    the dynamic range of a LAB colour difference for skin lesions, giving a
    value in [0, 1].
    """
    lab_a = cv2.cvtColor(grey_world_balance(rgb_a), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(grey_world_balance(rgb_b), cv2.COLOR_RGB2LAB).astype(np.float32)

    sel_a, sel_b = mask_a > 0, mask_b > 0
    if sel_a.sum() < 10 or sel_b.sum() < 10:
        return 0.0

    delta = lab_a[sel_a].mean(axis=0) - lab_b[sel_b].mean(axis=0)
    return float(np.clip(np.linalg.norm(delta) / 100.0, 0.0, 1.0))


def relative_area_change(mask_a: LesionMask, mask_b: LesionMask) -> float:
    """Return the relative change in lesion area fraction between two photos.

    This is the weakest signal in the module: it is only meaningful if both
    photos were taken from a similar distance, which nothing enforces. It is
    weighted accordingly and always described with that caveat.
    """
    before = max(mask_a.area_fraction, 1e-6)
    after = mask_b.area_fraction
    return float(abs(after - before) / before)


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------
def detect_change(
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
    cfg: Config,
    backbone: Optional[Any] = None,
    previous_mask: Optional[LesionMask] = None,
    current_mask: Optional[LesionMask] = None,
) -> ChangeResult:
    """Compare two photos of the same lesion and decide whether it has changed.

    Args:
        previous_rgb: The earlier photo, HxWx3 uint8 RGB.
        current_rgb: The recent photo, HxWx3 uint8 RGB.
        cfg: Loaded project config (reads the ``tracking:`` section).
        backbone: An optional :class:`FeatureBackbone`. When supplied, the
            cosine distance between the two lesion embeddings is used as the
            strongest single signal, since a foundation-model embedding
            captures appearance changes the hand-written descriptors miss.
        previous_mask: Precomputed mask for ``previous_rgb``.
        current_mask: Precomputed mask for ``current_rgb``.

    Returns:
        A :class:`ChangeResult`.

    Raises:
        ValueError: If either image is not an HxWx3 array.
    """
    for name, image in (("previous", previous_rgb), ("current", current_rgb)):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{name} image must be HxWx3 RGB, got {image.shape}")

    mask_prev = previous_mask or segment_lesion(previous_rgb)
    mask_curr = current_mask or segment_lesion(current_rgb)
    reliable = bool(mask_prev.found and mask_curr.found)

    crop_prev, cmask_prev = align_lesion_crop(previous_rgb, mask_prev)
    crop_curr, cmask_curr = align_lesion_crop(current_rgb, mask_curr)

    signals: List[ChangeSignal] = []

    # --- appearance embedding (strongest, when available) -----------------
    if backbone is not None:
        try:
            emb_prev = backbone.embed_image(crop_prev).numpy()
            emb_curr = backbone.embed_image(crop_curr).numpy()
            distance = cosine_distance(emb_prev, emb_curr)
            threshold = float(cfg.get("tracking.embedding_distance_threshold", 0.15))
            signals.append(ChangeSignal(
                name="appearance",
                value=distance, threshold=threshold,
                changed=distance > threshold, weight=0.40,
                description=(
                    "Its overall appearance has shifted noticeably."
                    if distance > threshold else
                    "Its overall appearance is essentially unchanged."
                ),
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding comparison unavailable (%s); relying on "
                           "shape and colour signals only.", exc)

    # --- colour -----------------------------------------------------------
    colour_delta = mean_colour_shift(crop_prev, cmask_prev, crop_curr, cmask_curr)
    colour_threshold = float(cfg.get("tracking.color_change_threshold", 0.20))
    signals.append(ChangeSignal(
        name="colour",
        value=colour_delta, threshold=colour_threshold,
        changed=colour_delta > colour_threshold, weight=0.25,
        description=(
            "Its colour has changed."
            if colour_delta > colour_threshold else
            "Its colour looks about the same."
        ),
    ))

    # --- asymmetry --------------------------------------------------------
    asym_delta = abs(asymmetry_score(cmask_curr) - asymmetry_score(cmask_prev))
    asym_threshold = float(cfg.get("tracking.asymmetry_change_threshold", 0.10))
    signals.append(ChangeSignal(
        name="asymmetry",
        value=asym_delta, threshold=asym_threshold,
        changed=asym_delta > asym_threshold, weight=0.20,
        description=(
            "It has become more lopsided than it was."
            if asym_delta > asym_threshold else
            "Its symmetry is about the same."
        ),
    ))

    # --- border -----------------------------------------------------------
    border_delta = abs(border_irregularity(cmask_curr) - border_irregularity(cmask_prev))
    border_threshold = 0.35
    signals.append(ChangeSignal(
        name="border",
        value=border_delta, threshold=border_threshold,
        changed=border_delta > border_threshold, weight=0.20,
        description=(
            "Its edge has become more ragged."
            if border_delta > border_threshold else
            "Its edge looks much as it did."
        ),
    ))

    # --- relative size (weak: no scale reference between the two photos) ---
    area_delta = relative_area_change(mask_prev, mask_curr)
    area_threshold = float(cfg.get("tracking.area_change_threshold", 0.25))
    signals.append(ChangeSignal(
        name="relative_size",
        value=area_delta, threshold=area_threshold,
        changed=area_delta > area_threshold, weight=0.10,
        description=(
            "It takes up more of the frame than before -- though that may simply "
            "mean the camera was closer this time."
            if area_delta > area_threshold else
            "It takes up a similar share of the frame."
        ),
    ))

    total_weight = sum(s.weight for s in signals)
    score = (sum(s.weight for s in signals if s.changed) / total_weight
             if total_weight > 0 else 0.0)
    decision_threshold = 0.35
    changed = score >= decision_threshold

    if not reliable:
        reason = ("We could not outline the lesion clearly in one of the two "
                  "photos, so this comparison is unreliable. Retake both photos "
                  "with the lesion centred against plain skin.")
    elif changed:
        drivers = [s.description for s in signals if s.changed]
        reason = " ".join(drivers[:3])
    else:
        reason = ("Nothing we measure has shifted meaningfully between the two "
                  "photos.")

    result = ChangeResult(
        changed=changed and reliable,
        score=score,
        threshold=decision_threshold,
        signals=signals,
        reason=reason,
        reliable=reliable,
    )
    logger.info("Change detection: changed=%s score=%.3f (%s)",
                result.changed, result.score,
                ", ".join(f"{s.name}={s.value:.3f}" for s in signals))
    return result
