"""ABCDE heuristics turned into plain-language explanations.

The ABCDE rule is the mnemonic dermatologists teach patients for spotting
melanoma:

* **A**symmetry -- one half does not match the other.
* **B**order -- edges are ragged, notched or blurred.
* **C**olour -- more than one shade, especially black, red or blue.
* **D**iameter -- larger than about 6 mm (a pencil eraser).
* **E**volving -- changing in size, shape, colour, or newly itching/bleeding.

These are computed from the segmented lesion with classical image measurements,
**independently of the neural network**. That independence is the point: if the
model says High and the ABCDE measurements agree, the user gets a reason they
can check with their own eyes; if they disagree, the report says so rather than
inventing a justification for the model's output.

Two honest limitations, surfaced in the output rather than hidden:

* Diameter in millimetres is an *estimate*. A phone photo has no scale
  reference, so it is derived from an assumed field of view
  (``explain.abcde.assumed_field_of_view_mm``). Treat it as a rough guide.
* "Evolving" cannot be measured from one photo. It is only reported when a
  prior photo is supplied, via :mod:`src.tracking.change_detection`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.config import Config
from src.segmentation import LesionMask, largest_contour, segment_lesion
from src.utils import get_logger

logger = get_logger(__name__)

#: Order used when presenting the criteria.
ABCDE_ORDER = ["asymmetry", "border", "colour", "diameter", "evolving"]


@dataclass
class ABCDEFeature:
    """One ABCDE criterion's measurement and its plain-language rendering.

    Attributes:
        key: Criterion id, one of :data:`ABCDE_ORDER`.
        letter: The mnemonic letter, e.g. ``"A"``.
        title: Display name, e.g. ``"Asymmetry"``.
        score: The measured statistic (units differ per criterion).
        threshold: The value ``score`` was compared against.
        flagged: ``True`` when the criterion is a concern.
        sentence: Patient-facing explanation.
        measurable: ``False`` when the criterion could not be assessed (e.g.
            "evolving" with only one photo). ``flagged`` is then meaningless.
    """

    key: str
    letter: str
    title: str
    score: Optional[float]
    threshold: Optional[float]
    flagged: bool
    sentence: str
    measurable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "key": self.key,
            "letter": self.letter,
            "title": self.title,
            "score": (round(float(self.score), 4) if self.score is not None else None),
            "threshold": self.threshold,
            "flagged": bool(self.flagged),
            "measurable": bool(self.measurable),
            "sentence": self.sentence,
        }


@dataclass
class ABCDEReport:
    """All five criteria for one lesion.

    Attributes:
        features: The criteria, in :data:`ABCDE_ORDER`.
        segmentation_confident: ``False`` when the lesion mask fell back to a
            centred ellipse, meaning every measurement here is unreliable.
    """

    features: List[ABCDEFeature] = field(default_factory=list)
    segmentation_confident: bool = True

    @property
    def flagged(self) -> List[ABCDEFeature]:
        """Criteria that were both measurable and concerning."""
        return [f for f in self.features if f.flagged and f.measurable]

    @property
    def flag_count(self) -> int:
        """How many criteria were flagged."""
        return len(self.flagged)

    def get(self, key: str) -> Optional[ABCDEFeature]:
        """Return the feature with the given key, or ``None``."""
        return next((f for f in self.features if f.key == key), None)

    def sentences(self) -> List[str]:
        """Return every criterion's sentence, in presentation order."""
        return [f.sentence for f in self.features]

    def summary(self) -> str:
        """Return a one-line summary of how many criteria were flagged."""
        measurable = [f for f in self.features if f.measurable]
        count = self.flag_count
        if not self.segmentation_confident:
            return ("The lesion outline could not be found reliably, so these "
                    "ABCDE measurements are unreliable. Retake the photo with the "
                    "lesion centred against plain skin.")
        if count == 0:
            return (f"None of the {len(measurable)} ABCDE features we could measure "
                    "looked concerning.")
        letters = ", ".join(f.letter for f in self.flagged)
        return (f"{count} of the {len(measurable)} ABCDE features we could measure "
                f"looked concerning ({letters}).")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "summary": self.summary(),
            "flag_count": self.flag_count,
            "segmentation_confident": bool(self.segmentation_confident),
            "features": [f.to_dict() for f in self.features],
        }


# ---------------------------------------------------------------------------
# Individual measurements
# ---------------------------------------------------------------------------
def asymmetry_score(mask: np.ndarray) -> float:
    """Measure how poorly the lesion matches itself when folded in half.

    The mask is rotated onto its principal axes (via PCA of the lesion pixel
    coordinates) so the measurement does not depend on how the photo was
    oriented, then mirrored about each axis. The score is the fraction of
    lesion area that fails to overlap its mirror image, averaged over the two
    axes.

    Args:
        mask: HxW mask, non-zero inside the lesion.

    Returns:
        A value in [0, 1]; 0 is perfectly symmetric. Returns 0 for an empty
        mask.
    """
    binary = (mask > 0).astype(np.uint8)
    coords = np.column_stack(np.nonzero(binary))  # (y, x)
    if coords.shape[0] < 10:
        return 0.0

    centre = coords.mean(axis=0)
    centred = coords - centre
    # Principal axis = eigenvector of the covariance with the largest eigenvalue.
    cov = np.cov(centred, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = float(np.degrees(np.arctan2(major[0], major[1])))

    h, w = binary.shape
    rotation = cv2.getRotationMatrix2D((float(centre[1]), float(centre[0])), angle, 1.0)
    aligned = cv2.warpAffine(binary, rotation, (w, h), flags=cv2.INTER_NEAREST)

    ys, xs = np.nonzero(aligned)
    if ys.size == 0:
        return 0.0
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = aligned[y0:y1, x0:x1]

    area = float(crop.sum())
    if area <= 0:
        return 0.0

    horizontal_mismatch = float(np.logical_xor(crop, np.flip(crop, axis=1)).sum())
    vertical_mismatch = float(np.logical_xor(crop, np.flip(crop, axis=0)).sum())
    # XOR double-counts (it covers both the shape and its mirror), hence /2.
    return float(np.clip(
        ((horizontal_mismatch / 2) + (vertical_mismatch / 2)) / (2 * area), 0.0, 1.0
    ))


def border_irregularity(mask: np.ndarray) -> float:
    """Measure border raggedness as ``perimeter^2 / (4*pi*area)``.

    This is the reciprocal of circularity: a perfect circle scores 1.0 and any
    departure -- notches, scalloped edges, spiky projections -- pushes it
    higher. The mask is smoothed first, because a pixelated staircase edge
    inflates the measured perimeter regardless of the real shape.

    Args:
        mask: HxW mask, non-zero inside the lesion.

    Returns:
        A value >= 1.0, or 1.0 if no contour could be found.
    """
    binary = (mask > 0).astype(np.uint8) * 255
    ksize = max(3, (min(binary.shape) // 100) * 2 + 1)
    smoothed = cv2.GaussianBlur(binary, (ksize, ksize), 0)
    _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)

    contour = largest_contour(smoothed)
    if contour is None or len(contour) < 5:
        return 1.0

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, closed=True))
    if area <= 1.0 or perimeter <= 0.0:
        return 1.0
    return float(max(1.0, (perimeter ** 2) / (4.0 * np.pi * area)))


def colour_variation(rgb: np.ndarray, mask: np.ndarray,
                     max_clusters: int = 6,
                     min_cluster_fraction: float = 0.08) -> tuple[float, int]:
    """Measure colour spread and count distinct colour regions in the lesion.

    Colours are clustered in CIELAB, where Euclidean distance approximates
    perceived colour difference far better than it does in RGB, so "how many
    distinct shades would a person see?" maps onto "how many well-populated
    clusters are there?".

    Args:
        rgb: HxWx3 uint8 RGB image.
        mask: HxW mask, non-zero inside the lesion.
        max_clusters: Upper bound on clusters to fit.
        min_cluster_fraction: A cluster must hold at least this share of lesion
            pixels to count as a distinct shade rather than an edge artefact.

    Returns:
        ``(variation, n_shades)`` where ``variation`` is the normalised
        standard deviation of lesion colour in [0, 1] and ``n_shades`` is the
        number of well-populated clusters.
    """
    binary = mask > 0
    if binary.sum() < 20:
        return 0.0, 1

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pixels = lab[binary]

    # Normalised spread: LAB channels each span roughly 0-255 here.
    variation = float(np.clip(pixels.std(axis=0).mean() / 64.0, 0.0, 1.0))

    # Subsample for speed -- k-means on 100k pixels is pointless.
    rng = np.random.default_rng(0)
    if pixels.shape[0] > 4000:
        pixels = pixels[rng.choice(pixels.shape[0], 4000, replace=False)]

    n_shades = 1
    try:
        from sklearn.cluster import KMeans

        # Never ask for more clusters than there are distinct colours: KMeans
        # cannot satisfy that and warns on every call. Quantising to 4-unit
        # bins first also stops sensor noise from inflating the count.
        distinct = len(np.unique((pixels / 4.0).astype(np.int16), axis=0))
        max_clusters = min(max_clusters, max(1, distinct))

        best_k = 1
        for k in range(2, max_clusters + 1):
            if pixels.shape[0] < k * 10:
                break
            kmeans = KMeans(n_clusters=k, n_init=4, random_state=0).fit(pixels)
            counts = np.bincount(kmeans.labels_, minlength=k) / pixels.shape[0]
            populated = int((counts >= min_cluster_fraction).sum())
            # Clusters must also be perceptually apart, not just numerous.
            centres = kmeans.cluster_centers_[counts >= min_cluster_fraction]
            if len(centres) >= 2:
                distances = [
                    float(np.linalg.norm(a - b))
                    for i, a in enumerate(centres) for b in centres[i + 1:]
                ]
                if max(distances) < 18.0:  # ~ a just-noticeable difference in LAB
                    populated = 1
            best_k = max(best_k, populated)
        n_shades = best_k
    except ImportError:  # pragma: no cover - sklearn is a hard dependency
        logger.warning("scikit-learn unavailable; colour clustering skipped.")

    return variation, int(n_shades)


def estimate_diameter_mm(mask: np.ndarray, image_width: int,
                         field_of_view_mm: float) -> tuple[float, float]:
    """Estimate the lesion's widest diameter in pixels and millimetres.

    The millimetre figure assumes the photo spans ``field_of_view_mm`` across
    its width. Without a ruler or coin in frame there is no way to do better,
    so this is explicitly an estimate and the report says so.

    Args:
        mask: HxW mask, non-zero inside the lesion.
        image_width: Image width in pixels.
        field_of_view_mm: Assumed real-world width of the frame.

    Returns:
        ``(diameter_px, diameter_mm)``.
    """
    contour = largest_contour(mask)
    if contour is None or len(contour) < 5:
        return 0.0, 0.0
    _, radius = cv2.minEnclosingCircle(contour)
    diameter_px = float(radius) * 2.0
    mm_per_px = float(field_of_view_mm) / max(int(image_width), 1)
    return diameter_px, diameter_px * mm_per_px


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def compute_abcde(
    rgb: np.ndarray,
    cfg: Config,
    mask: Optional[LesionMask] = None,
    change_summary: Optional[Dict[str, Any]] = None,
    frame_width_mm: Optional[float] = None,
) -> ABCDEReport:
    """Measure all five ABCDE criteria and phrase each in plain language.

    Args:
        rgb: HxWx3 uint8 RGB image.
        cfg: Loaded project config (reads ``explain.abcde``).
        mask: A precomputed lesion mask. Segmented here if omitted.
        change_summary: Output of :func:`src.tracking.change_detection.detect_change`
            as a dict. Supplies the "Evolving" criterion; when ``None``, that
            criterion is reported as not measurable.
        frame_width_mm: Real-world width of the photo in millimetres, if the
            user can supply it (e.g. by photographing a ruler alongside the
            lesion). Without it the Diameter criterion reports a relative size
            and is marked not measurable, because a photo carries no scale.

    Returns:
        A populated :class:`ABCDEReport`.

    Raises:
        ValueError: If ``rgb`` is not an HxWx3 array.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got shape {rgb.shape}")

    if mask is None:
        mask = segment_lesion(rgb)
    binary = mask.mask
    features: List[ABCDEFeature] = []

    # --- A: Asymmetry -----------------------------------------------------
    a_threshold = float(cfg.get("explain.abcde.asymmetry_threshold", 0.18))
    a_score = asymmetry_score(binary)
    a_flag = a_score > a_threshold
    features.append(ABCDEFeature(
        key="asymmetry", letter="A", title="Asymmetry",
        score=a_score, threshold=a_threshold, flagged=a_flag,
        sentence=(
            "Asymmetry: the two halves of this mark do not match each other well. "
            "Harmless moles are usually close to symmetrical."
            if a_flag else
            "Asymmetry: the mark looks fairly symmetrical -- one half is a good "
            "match for the other, which is reassuring."
        ),
    ))

    # --- B: Border --------------------------------------------------------
    b_threshold = float(cfg.get("explain.abcde.border_irregularity_threshold", 1.55))
    b_score = border_irregularity(binary)
    b_flag = b_score > b_threshold
    features.append(ABCDEFeature(
        key="border", letter="B", title="Border",
        score=b_score, threshold=b_threshold, flagged=b_flag,
        sentence=(
            "Border: the edge is ragged and uneven rather than a smooth curve. "
            "Notched or scalloped edges are one of the things doctors look for."
            if b_flag else
            "Border: the edge is smooth and well defined, the way an ordinary "
            "mole's edge usually looks."
        ),
    ))

    # --- C: Colour --------------------------------------------------------
    c_var_threshold = float(cfg.get("explain.abcde.color_variance_threshold", 0.16))
    c_cluster_threshold = int(cfg.get("explain.abcde.color_cluster_threshold", 3))
    c_variation, n_shades = colour_variation(rgb, binary)
    c_flag = (c_variation > c_var_threshold) or (n_shades >= c_cluster_threshold)
    if c_flag:
        c_sentence = (
            f"Colour: we count about {n_shades} clearly different shades in this "
            "mark. Several colours mixed together in one lesion is a feature "
            "worth having checked."
        )
    elif n_shades > 1:
        c_sentence = (
            f"Colour: there are around {n_shades} shades here, but they are close "
            "to one another. Even colouring is typical of a harmless mole."
        )
    else:
        c_sentence = ("Colour: the mark is a single even colour throughout, which "
                      "is typical of a harmless mole.")
    features.append(ABCDEFeature(
        key="colour", letter="C", title="Colour",
        score=c_variation, threshold=c_var_threshold, flagged=c_flag,
        sentence=c_sentence,
    ))

    # --- D: Diameter ------------------------------------------------------
    # A photo carries no scale. Without knowing how wide the frame is in real
    # life, "how many millimetres across is this?" is unanswerable: the same
    # lesion fills the frame from 5 cm away and a corner of it from 30 cm.
    # So the criterion only *counts* when the user supplies a scale; otherwise
    # we still show the estimate but mark it unmeasurable, rather than flagging
    # every single lesion and making the letter meaningless.
    d_threshold = float(cfg.get("explain.abcde.diameter_mm_threshold", 6.0))
    fov = (float(frame_width_mm) if frame_width_mm else
           float(cfg.get("explain.abcde.assumed_field_of_view_mm", 40.0)))
    diameter_px, diameter_mm = estimate_diameter_mm(binary, rgb.shape[1], fov)
    has_scale = frame_width_mm is not None and frame_width_mm > 0
    lesion_share = diameter_px / max(rgb.shape[1], 1)

    if has_scale:
        d_flag = diameter_mm > d_threshold
        d_sentence = (
            f"Diameter: using the {fov:.0f} mm frame width you gave us, this mark "
            f"is about {diameter_mm:.1f} mm across -- wider than the "
            f"{d_threshold:.0f} mm guideline (roughly a pencil eraser)."
            if d_flag else
            f"Diameter: using the {fov:.0f} mm frame width you gave us, this mark "
            f"is about {diameter_mm:.1f} mm across, under the {d_threshold:.0f} mm "
            "guideline."
        )
    else:
        d_flag = False
        d_sentence = (
            f"Diameter: the mark spans about {lesion_share:.0%} of the photo's "
            "width, but we cannot convert that to millimetres without knowing how "
            "close the camera was. Measure it with a ruler: anything wider than "
            f"{d_threshold:.0f} mm (about a pencil eraser) is worth showing to a "
            "doctor."
        )
    features.append(ABCDEFeature(
        key="diameter", letter="D", title="Diameter",
        score=diameter_mm if has_scale else lesion_share,
        threshold=d_threshold if has_scale else None,
        flagged=d_flag, measurable=has_scale,
        sentence=d_sentence,
    ))

    # --- E: Evolving ------------------------------------------------------
    if change_summary is None:
        features.append(ABCDEFeature(
            key="evolving", letter="E", title="Evolving",
            score=None, threshold=None, flagged=False, measurable=False,
            sentence=(
                "Evolving: we cannot tell whether this mark is changing from a "
                "single photo. Upload an earlier photo of the same lesion to have "
                "this checked -- change over time is one of the most important "
                "warning signs."
            ),
        ))
    else:
        changed = bool(change_summary.get("changed", False))
        features.append(ABCDEFeature(
            key="evolving", letter="E", title="Evolving",
            score=float(change_summary.get("score", 0.0)),
            threshold=float(change_summary.get("threshold", 0.0)),
            flagged=changed, measurable=True,
            sentence=(
                "Evolving: compared with the earlier photo you gave us, this mark "
                f"has changed noticeably. {change_summary.get('reason', '')}".strip()
                if changed else
                "Evolving: compared with the earlier photo you gave us, this mark "
                "looks much the same. No significant change detected."
            ),
        ))

    return ABCDEReport(features=features, segmentation_confident=mask.found)
