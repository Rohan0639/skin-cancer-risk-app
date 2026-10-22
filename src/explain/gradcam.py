"""Grad-CAM heatmaps over the lesion.

Grad-CAM (Selvaraju et al., ICCV 2017) weights a convolutional feature map by
the gradient of the predicted class score with respect to that map, producing
a coarse "where did the model look?" image. Here it answers the question a
patient actually has: *is the model reacting to my lesion, or to a freckle, a
ruler, or the shadow at the edge of the photo?*

``pytorch-grad-cam`` is used when installed; otherwise the equivalent
:func:`_gradcam_manual` implementation below runs, so the explanation never
silently disappears. Both hook the layer the backbone nominates via
:meth:`FeatureBackbone.gradcam_target_layers`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch

from src.config import Config
from src.dataset import encode_metadata
from src.model.risk_classifier import GradCAMTarget, RiskClassifier
from src.utils import get_logger

logger = get_logger(__name__)

#: Matplotlib-style colormap names mapped to OpenCV constants.
_COLORMAPS: Dict[str, int] = {
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "hot": cv2.COLORMAP_HOT,
}


@dataclass
class GradCAMResult:
    """A Grad-CAM explanation for one image.

    Attributes:
        heatmap: HxW float32 array in [0, 1], at the original image resolution.
        overlay: HxWx3 uint8 RGB image with the heatmap blended over the photo.
        target_class: Index of the class the saliency was computed for.
        target_class_name: Human-readable name of that class.
        peak_inside_lesion: Fraction of the heatmap's mass falling inside the
            segmented lesion, or ``None`` if no mask was supplied. Interpret it
            via :attr:`focus_ratio`, not on its own.
        lesion_area_fraction: Share of the frame the lesion occupies. This is
            what a *uniform* heatmap would score, and therefore the baseline
            ``peak_inside_lesion`` must be judged against.
        method: ``"pytorch-grad-cam"`` or ``"builtin"``.
    """

    heatmap: np.ndarray
    overlay: np.ndarray
    target_class: int
    target_class_name: str
    peak_inside_lesion: Optional[float] = None
    lesion_area_fraction: Optional[float] = None
    method: str = "builtin"

    @property
    def focus_ratio(self) -> Optional[float]:
        """How much more attention the lesion got than pure chance.

        ``peak_inside_lesion`` alone is misleading: a lesion filling 10% of the
        frame cannot attract 35% of a diffuse heatmap however well the model is
        behaving, so a fixed fraction would flag every small lesion. Dividing
        by the lesion's area fraction gives a scale-free number where 1.0 means
        "no better than uniform" and higher means genuinely concentrated.

        Returns:
            The ratio, or ``None`` if either input is unavailable.
        """
        if self.peak_inside_lesion is None or not self.lesion_area_fraction:
            return None
        return float(self.peak_inside_lesion / max(self.lesion_area_fraction, 1e-6))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary (without the image arrays)."""
        ratio = self.focus_ratio
        return {
            "target_class": int(self.target_class),
            "target_class_name": self.target_class_name,
            "peak_inside_lesion": (round(float(self.peak_inside_lesion), 4)
                                   if self.peak_inside_lesion is not None else None),
            "lesion_area_fraction": (round(float(self.lesion_area_fraction), 4)
                                     if self.lesion_area_fraction is not None else None),
            "focus_ratio": round(float(ratio), 3) if ratio is not None else None,
            "method": self.method,
        }


def overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.45, colormap: str = "jet") -> np.ndarray:
    """Blend a [0, 1] heatmap over an RGB image.

    Args:
        rgb: HxWx3 uint8 RGB image.
        heatmap: HxW float array in [0, 1]; resized to match ``rgb`` if needed.
        alpha: Heatmap opacity, 0 (invisible) to 1 (opaque).
        colormap: Name from :data:`_COLORMAPS`.

    Returns:
        HxWx3 uint8 RGB overlay.
    """
    height, width = rgb.shape[:2]
    if heatmap.shape[:2] != (height, width):
        heatmap = cv2.resize(heatmap.astype(np.float32), (width, height),
                             interpolation=cv2.INTER_LINEAR)

    scaled = np.clip(heatmap, 0.0, 1.0)
    coloured = cv2.applyColorMap((scaled * 255).astype(np.uint8),
                                 _COLORMAPS.get(colormap, cv2.COLORMAP_JET))
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = rgb.astype(np.float32) * (1 - alpha) + coloured.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _normalise(array: np.ndarray) -> np.ndarray:
    """Rescale an array to [0, 1], returning zeros if it is constant."""
    array = array.astype(np.float32)
    low, high = float(array.min()), float(array.max())
    if high - low < 1e-8:
        return np.zeros_like(array)
    return (array - low) / (high - low)


def _gradcam_manual(
    wrapped: GradCAMTarget,
    input_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    class_index: int,
    reshape_transform=None,
) -> np.ndarray:
    """Compute Grad-CAM directly with forward/backward hooks.

    Used when ``pytorch-grad-cam`` is not installed. Registers hooks on
    ``target_layer``, backpropagates the target class score, then weights each
    activation channel by its globally averaged gradient.

    Returns:
        An HxW float32 saliency map in [0, 1] at the *activation* resolution.
    """
    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handles = [
        target_layer.register_forward_hook(forward_hook),
        target_layer.register_full_backward_hook(backward_hook),
    ]

    try:
        wrapped.zero_grad(set_to_none=True)
        logits = wrapped(input_tensor)
        score = logits[0, class_index]
        score.backward()

        if "value" not in activations or "value" not in gradients:
            raise RuntimeError("Grad-CAM hooks captured nothing for the target layer.")

        acts = activations["value"].detach()
        grads = gradients["value"].detach()
        if reshape_transform is not None:
            acts = reshape_transform(acts)
            grads = reshape_transform(grads)

        # Channel weights = spatially averaged gradients.
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = torch.relu(cam)  # only evidence *for* the class
        return _normalise(cam[0].cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()


def compute_gradcam(
    model: RiskClassifier,
    rgb: np.ndarray,
    cfg: Config,
    age: Any = None,
    sex: Any = None,
    site: Any = None,
    class_index: Optional[int] = None,
    lesion_mask: Optional[np.ndarray] = None,
) -> GradCAMResult:
    """Produce a Grad-CAM explanation for one photo.

    Args:
        model: A trained :class:`RiskClassifier`.
        rgb: HxWx3 uint8 RGB image.
        cfg: Loaded project config (reads ``explain.gradcam``).
        age: Patient age, used to bind the metadata branch.
        sex: Patient sex.
        site: Anatomical site.
        class_index: Class to explain. Defaults to the predicted class, which
            is what a user wants: "why did you say *this*?"
        lesion_mask: Optional HxW mask used to compute
            :attr:`GradCAMResult.peak_inside_lesion`.

    Returns:
        A :class:`GradCAMResult` with the heatmap and a blended overlay.

    Raises:
        ValueError: If ``rgb`` is not an HxWx3 array.
    """
    from PIL import Image

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got shape {rgb.shape}")

    device = next(model.parameters()).device
    model.eval()

    transform = model.backbone.build_transform(train=False)
    input_tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    meta = torch.from_numpy(
        encode_metadata(age, sex, site, cfg)
    ).unsqueeze(0).to(device)
    wrapped = GradCAMTarget(model, meta).to(device).eval()

    if class_index is None:
        with torch.no_grad():
            class_index = int(wrapped(input_tensor).argmax(dim=1).item())

    target_layers = model.backbone.gradcam_target_layers()
    reshape_transform = model.backbone.gradcam_reshape_transform()
    method = "builtin"
    heatmap: Optional[np.ndarray] = None

    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        # The backbone is frozen (requires_grad=False), but Grad-CAM needs
        # gradients to flow back to the target layer's activations. Re-enable
        # them for the duration of the call, then restore.
        previously = [(p, p.requires_grad) for p in model.backbone.parameters()]
        for param, _ in previously:
            param.requires_grad_(True)
        try:
            with GradCAM(model=wrapped, target_layers=target_layers,
                         reshape_transform=reshape_transform) as cam:
                grayscale = cam(
                    input_tensor=input_tensor,
                    targets=[ClassifierOutputTarget(class_index)],
                )
            heatmap = _normalise(grayscale[0])
            method = "pytorch-grad-cam"
        finally:
            for param, was in previously:
                param.requires_grad_(was)
    except ImportError:
        logger.info("pytorch-grad-cam is not installed; using the built-in "
                    "Grad-CAM implementation.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("pytorch-grad-cam failed (%s); using the built-in "
                       "implementation instead.", exc)

    if heatmap is None:
        previously = [(p, p.requires_grad) for p in model.backbone.parameters()]
        for param, _ in previously:
            param.requires_grad_(True)
        try:
            heatmap = _gradcam_manual(wrapped, input_tensor, target_layers[0],
                                      class_index, reshape_transform)
        finally:
            for param, was in previously:
                param.requires_grad_(was)

    height, width = rgb.shape[:2]
    heatmap_full = cv2.resize(heatmap.astype(np.float32), (width, height),
                              interpolation=cv2.INTER_LINEAR)
    heatmap_full = _normalise(heatmap_full)

    overlay = overlay_heatmap(
        rgb, heatmap_full,
        alpha=float(cfg.get("explain.gradcam.alpha", 0.45)),
        colormap=str(cfg.get("explain.gradcam.colormap", "jet")),
    )

    degenerate = float(heatmap_full.max()) <= 1e-6
    if degenerate:
        # Every activation's gradient was non-positive for this class, so
        # Grad-CAM's ReLU zeroed the whole map. Say so rather than reporting a
        # meaningless attention fraction from an empty heatmap.
        logger.warning("Grad-CAM produced an empty heatmap for class %d; the "
                       "attention overlay is not informative for this image.",
                       class_index)

    inside: Optional[float] = None
    area_fraction: Optional[float] = None
    if lesion_mask is not None and not degenerate:
        mask = (cv2.resize(lesion_mask.astype(np.uint8), (width, height),
                           interpolation=cv2.INTER_NEAREST) > 0)
        area_fraction = float(mask.mean())
        total = float(heatmap_full.sum())
        if total > 1e-8:
            inside = float(heatmap_full[mask].sum() / total)

    classes = list(model.classes)
    return GradCAMResult(
        heatmap=heatmap_full,
        overlay=overlay,
        target_class=int(class_index),
        target_class_name=classes[int(class_index)],
        peak_inside_lesion=inside,
        lesion_area_fraction=area_fraction,
        method=method,
    )


def attention_sanity_note(result: GradCAMResult,
                          min_focus_ratio: float = 1.2) -> Optional[str]:
    """Return a caution sentence when the heatmap is not concentrated on the lesion.

    The test is on :attr:`GradCAMResult.focus_ratio`, not on the raw in-lesion
    fraction: a ratio at or below 1.0 means the model spread its attention no
    better than if it had ignored the lesion entirely, and that is the case
    worth warning about regardless of how big the lesion is in frame.

    Args:
        result: A computed Grad-CAM result carrying a lesion mask.
        min_focus_ratio: Concentration below which the note is emitted. The
            default of 1.2 asks for only modestly better than chance.

    Returns:
        A warning sentence, or ``None`` if attention looks reasonable or could
        not be measured.
    """
    ratio = result.focus_ratio
    if ratio is None or ratio >= min_focus_ratio:
        return None
    return (
        f"Caution: the model's attention was no more concentrated on the lesion "
        f"than on the surrounding skin (only {result.peak_inside_lesion:.0%} of it "
        f"fell on a lesion covering {result.lesion_area_fraction:.0%} of the "
        "photo). It may be reacting to the background rather than the mark. "
        "Retake the photo with the lesion centred against plain skin."
    )
