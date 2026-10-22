"""Classical lesion segmentation.

Both the image-quality gate (is a lesion actually framed?) and the ABCDE
heuristics (asymmetry, border, colour, diameter) need a lesion mask. Rather
than train a segmentation network -- which would need pixel-level annotations
this project does not have -- we use a deliberately simple, inspectable
classical pipeline: blur, Otsu threshold on a lesion-contrast channel,
morphological cleanup, then keep the most lesion-like connected component.

This is approximate. It works well on the centred, reasonably lit photos the
app asks for and degrades on cluttered ones, which is exactly why the quality
gate runs first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class LesionMask:
    """Result of segmenting a single lesion.

    Attributes:
        mask: HxW uint8 array, 255 inside the lesion and 0 outside.
        area_fraction: Lesion area as a fraction of total image area.
        centroid: ``(y, x)`` centroid in pixel coordinates.
        bbox: ``(y0, x0, y1, x1)`` bounding box, half-open on the high side.
        found: ``False`` when no plausible lesion region was detected, in which
            case ``mask`` is a centred elliptical fallback so downstream code
            always has something to work with.
    """

    mask: np.ndarray
    area_fraction: float
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    found: bool


def _flat_field(channel: np.ndarray) -> np.ndarray:
    """Divide out low-frequency illumination from a float channel.

    Phone photos of skin always carry a lighting gradient -- flash falloff,
    shadow from the hand holding the camera. A single global Otsu threshold
    will happily cut along that gradient instead of along the lesion, which
    produces a mask covering half the frame. Dividing by a heavily blurred copy
    of the image flattens the illumination while preserving the lesion, whose
    spatial scale is much smaller than the blur kernel.
    """
    h, w = channel.shape
    # The kernel must be much larger than any lesion, otherwise the lesion
    # contributes to its own background estimate and gets suppressed from the
    # inside out. sigma is set explicitly rather than derived from ksize.
    ksize = int(max(h, w) * 0.75) | 1
    sigma = max(h, w) * 0.25
    background = cv2.GaussianBlur(channel, (ksize, ksize), sigma)
    corrected = channel / np.maximum(background, 1e-3)
    return corrected * float(np.mean(background))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes by refilling the mask's largest external contour.

    Lesions are simply connected in practice, so any hole is an artefact --
    a specular highlight from the flash, or a patch of the lesion centre that
    the threshold clipped. Leaving holes in would corrupt every downstream
    ABCDE measurement (area, asymmetry and especially perimeter).
    """
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return mask
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 1, thickness=-1)
    return (filled * 255).astype(np.uint8)


def _lesion_contrast_channel(rgb: np.ndarray) -> np.ndarray:
    """Return a uint8 channel in which the lesion is bright and skin is dark.

    Skin is reddish-orange and lesions are typically darker and/or bluer. We
    combine inverted luminance with the blue-yellow opponent channel of LAB,
    which separates pigmented lesions from surrounding skin better than
    grayscale alone, then flat-field the result so uneven lighting cannot
    dominate the threshold.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0].astype(np.float32)
    b_channel = lab[:, :, 2].astype(np.float32)  # blue(-) <-> yellow(+)

    darkness = _flat_field(255.0 - lightness)
    blueness = _flat_field(255.0 - b_channel)

    combined = 0.65 * darkness + 0.35 * blueness
    combined -= combined.min()
    peak = combined.max()
    if peak > 1e-6:
        combined *= 255.0 / peak
    return combined.astype(np.uint8)


def _border_touch_fraction(mask: np.ndarray) -> float:
    """Return the fraction of the image border occupied by ``mask``.

    A lesion the user was asked to centre should not run off the edge of the
    frame. A component hugging the border is almost always mis-thresholded
    background, so this is used to penalise such components.
    """
    binary = mask > 0
    border = np.concatenate([
        binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1],
    ])
    return float(border.mean()) if border.size else 0.0


def _refine_with_grabcut(rgb: np.ndarray, seed: np.ndarray,
                         iterations: int = 3) -> Optional[np.ndarray]:
    """Refine a coarse mask with GrabCut, seeded by ``seed``.

    Otsu works on one scalar channel, so it can split a genuinely two-tone
    lesion down the middle -- keeping the dark half and discarding the tan
    half. GrabCut models foreground and background as full colour distributions
    in RGB, so once the dark half seeds the foreground model, the adjoining
    lighter lesion tissue is pulled in too because it still differs from skin.

    Args:
        rgb: HxWx3 uint8 RGB image.
        seed: Coarse HxW mask to seed from.
        iterations: GrabCut iterations.

    Returns:
        The refined mask, or ``None`` if GrabCut failed or produced something
        implausible (in which case the caller keeps the original).
    """
    h, w = seed.shape
    binary = (seed > 0).astype(np.uint8)
    if binary.sum() < 50:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, min(h, w) // 25) | 1,) * 2)
    sure_fg = cv2.erode(binary, kernel, iterations=2)
    probable_fg = cv2.dilate(binary, kernel, iterations=2)

    grabcut_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    grabcut_mask[probable_fg > 0] = cv2.GC_PR_BGD
    grabcut_mask[binary > 0] = cv2.GC_PR_FGD
    if sure_fg.sum() > 20:
        grabcut_mask[sure_fg > 0] = cv2.GC_FGD

    # Need both foreground and background seeds for GrabCut to be solvable.
    if not (grabcut_mask == cv2.GC_BGD).any() or not (
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD)
    ).any():
        return None

    try:
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(np.ascontiguousarray(rgb), grabcut_mask, None,
                    bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error as exc:
        logger.debug("GrabCut refinement failed: %s", exc)
        return None

    refined = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # Keep only the component overlapping the original seed.
    n_labels, labels = cv2.connectedComponents((refined > 0).astype(np.uint8))
    if n_labels <= 1:
        return None
    overlaps = [(int(((labels == i) & (binary > 0)).sum()), i) for i in range(1, n_labels)]
    best_overlap, best_label = max(overlaps)
    if best_overlap == 0:
        return None
    refined = np.where(labels == best_label, 255, 0).astype(np.uint8)

    # Reject a refinement that ballooned into the background or collapsed.
    original_area = float(binary.sum())
    refined_area = float((refined > 0).sum())
    if refined_area < 0.4 * original_area or refined_area > 4.0 * original_area:
        logger.debug("Discarding GrabCut refinement (area %0.f -> %0.f).",
                     original_area, refined_area)
        return None
    if _border_touch_fraction(refined) > 0.30:
        return None
    return refined


def _fallback_mask(shape: Tuple[int, int]) -> np.ndarray:
    """A centred ellipse covering ~30% of the frame, used when detection fails."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.30), int(h * 0.30)), 0, 0, 360, 255, -1)
    return mask


def _describe(mask: np.ndarray, found: bool) -> LesionMask:
    """Package a binary mask into a :class:`LesionMask` with its statistics."""
    h, w = mask.shape
    binary = (mask > 0)
    total = float(h * w)
    ys, xs = np.nonzero(binary)
    if ys.size == 0:
        return LesionMask(mask=mask, area_fraction=0.0, centroid=(h / 2.0, w / 2.0),
                          bbox=(0, 0, h, w), found=False)
    return LesionMask(
        mask=(binary * 255).astype(np.uint8),
        area_fraction=float(binary.sum()) / total,
        centroid=(float(ys.mean()), float(xs.mean())),
        bbox=(int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1),
        found=found,
    )


def segment_lesion(
    rgb: np.ndarray,
    min_area_fraction: float = 0.005,
    max_area_fraction: float = 0.95,
    blur_ksize: int = 7,
    refine: bool = True,
) -> LesionMask:
    """Segment the dominant lesion in an RGB photo.

    Pipeline: flat-field the lesion-contrast channel, Otsu threshold,
    morphological cleanup, pick the most lesion-like connected component, then
    optionally refine its boundary with GrabCut.

    Args:
        rgb: HxWx3 uint8 RGB image.
        min_area_fraction: Components smaller than this fraction of the frame
            are rejected as noise.
        max_area_fraction: Components larger than this are rejected as
            background (usually a mis-thresholded skin field).
        blur_ksize: Gaussian kernel size used to suppress hair and texture.
        refine: Whether to run the GrabCut refinement pass. Disable for speed
            when only a rough area estimate is needed.

    Returns:
        A :class:`LesionMask`. ``found`` is ``False`` when nothing plausible was
        detected and the elliptical fallback is returned instead.

    Raises:
        ValueError: If ``rgb`` is not an HxWx3 array.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got shape {rgb.shape}")

    h, w = rgb.shape[:2]
    total = float(h * w)

    channel = _lesion_contrast_channel(rgb)
    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    channel = cv2.GaussianBlur(channel, (ksize, ksize), 0)

    _, binary = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Close holes (specular highlights) then open away hair and speckle.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(3, int(min(h, w) * 0.02)) | 1,) * 2
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)

    best_label: Optional[int] = None
    best_score = -np.inf
    img_centre = np.array([w / 2.0, h / 2.0])
    max_dist = float(np.hypot(w / 2.0, h / 2.0))

    for label in range(1, n_labels):  # 0 is background
        area = float(stats[label, cv2.CC_STAT_AREA])
        frac = area / total
        if frac < min_area_fraction or frac > max_area_fraction:
            continue

        component = (labels == label).astype(np.uint8)
        border_touch = _border_touch_fraction(component)
        # A component running along the frame edge is mis-thresholded
        # background, not a lesion the user centred.
        if border_touch > 0.35:
            continue

        cx, cy = centroids[label]
        centredness = 1.0 - float(np.linalg.norm(np.array([cx, cy]) - img_centre)) / max_dist
        # Prefer large, centred components that keep clear of the frame edge.
        score = frac + 1.2 * centredness - 1.5 * border_touch
        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        logger.debug("No plausible lesion component found; using elliptical fallback.")
        mask = _fallback_mask((h, w))
        return LesionMask(
            mask=mask,
            area_fraction=float((mask > 0).sum()) / total,
            centroid=(h / 2.0, w / 2.0),
            bbox=(0, 0, h, w),
            found=False,
        )

    mask = _fill_holes(np.where(labels == best_label, 255, 0).astype(np.uint8))

    if refine:
        refined = _refine_with_grabcut(rgb, mask)
        if refined is not None:
            mask = _fill_holes(cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel))

    return _describe(mask, found=True)


def largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Return the largest external contour of ``mask``, or ``None`` if empty."""
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)
