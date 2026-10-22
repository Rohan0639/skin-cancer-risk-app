"""ImageNet-pretrained fallback backbone.

Used whenever PanDerm's source or weights are unavailable. It keeps the whole
pipeline runnable end-to-end on a laptop with nothing but ``pip install``.

This is a **general-purpose** ImageNet model, not a dermatology model. Its
features were never trained on skin lesions, so results obtained through it
are strictly weaker than PanDerm's. Every code path that uses it logs a loud
warning and sets :attr:`FeatureBackbone.is_fallback`, which the reports and the
Streamlit UI surface to the user.
"""

from __future__ import annotations

from typing import Any, Callable, List

import torch
import torch.nn as nn

from src.backbone.base import BackboneUnavailable, FeatureBackbone, default_transform
from src.config import Config
from src.utils import get_logger

logger = get_logger(__name__)


class FallbackBackbone(FeatureBackbone):
    """Wraps any ``timm`` classification model as a pooled feature extractor."""

    is_fallback = True

    def __init__(self, cfg: Config) -> None:
        """Build the timm model named by ``backbone.fallback.model_name``.

        Raises:
            BackboneUnavailable: If timm is missing or the model cannot be
                created (e.g. no network access for pretrained weights and no
                cached copy).
        """
        super().__init__()
        self.cfg = cfg
        model_name = str(cfg.get("backbone.fallback.model_name", "convnext_base"))
        pretrained = bool(cfg.get("backbone.fallback.pretrained", True))

        try:
            import timm
        except ImportError as exc:  # pragma: no cover - timm is in requirements
            raise BackboneUnavailable(f"timm is not installed: {exc}") from exc

        try:
            # num_classes=0 strips the classifier and returns pooled features.
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        except Exception as exc:  # noqa: BLE001
            if pretrained:
                logger.warning(
                    "Could not download pretrained weights for '%s' (%s). "
                    "Falling back to RANDOM initialisation -- features will be "
                    "close to meaningless. Connect to the internet once to cache "
                    "the weights.", model_name, exc,
                )
                try:
                    self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
                except Exception as inner:  # noqa: BLE001
                    raise BackboneUnavailable(
                        f"Could not create timm model '{model_name}': {inner}"
                    ) from inner
            else:
                raise BackboneUnavailable(
                    f"Could not create timm model '{model_name}': {exc}"
                ) from exc

        configured_dim = cfg.get("backbone.fallback.feature_dim", None)
        self.feature_dim = int(configured_dim or getattr(self.model, "num_features", 0))
        if self.feature_dim <= 0:
            raise BackboneUnavailable(
                f"Could not determine the feature dimension of '{model_name}'."
            )

        self.name = f"timm:{model_name}"
        # Reproducibility matters at checkpoint-load time: a checkpoint stores
        # only the head, and rebuilds the backbone from config. That is sound
        # for pretrained weights, which are identical every time, but a
        # randomly initialised backbone differs on every construction, so the
        # restored head would sit on top of features it never saw.
        self.is_reproducible = bool(pretrained)
        self._pretrained = pretrained
        logger.warning(
            "USING FALLBACK BACKBONE: %s. This is a general ImageNet model, "
            "NOT the PanDerm dermatology foundation model. See README > "
            "'Using PanDerm' to enable PanDerm.", self.describe(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Embed a ``(B, 3, H, W)`` batch into ``(B, feature_dim)``."""
        return self.model(images)

    def build_transform(self, train: bool = False) -> Callable[[Any], torch.Tensor]:
        """Return the resize/normalise (and optionally augment) transform."""
        return default_transform(self.cfg, train=train)

    def gradcam_target_layers(self) -> List[nn.Module]:
        """Return the last full feature *stage* for Grad-CAM to hook.

        The right hook point is the output of the final stage, not the final
        ``Conv2d``. In ConvNeXt the deepest conv is a depthwise 7x7 inside the
        last block: its channels are spatially independent, so gradient-weighted
        channel summation over it produces sparse and frequently all-zero maps.
        Hooking the stage output -- after the block's pointwise mixing -- gives
        the dense semantic feature map Grad-CAM assumes.

        Known container attributes are tried in order (covering ConvNeXt,
        ResNet, EfficientNet, VGG and timm ViTs) before falling back to the
        last convolution.
        """
        for attribute in ("stages", "layer4", "blocks", "features"):
            container = getattr(self.model, attribute, None)
            if container is None:
                continue
            try:
                last = container[-1]
            except (TypeError, IndexError, KeyError):
                continue
            if isinstance(last, nn.Module):
                logger.debug("Grad-CAM target for %s: %s[-1] (%s)",
                             self.name, attribute, type(last).__name__)
                return [last]

        convs = [m for m in self.model.modules() if isinstance(m, nn.Conv2d)]
        if convs:
            logger.debug("Grad-CAM target for %s: last Conv2d (generic fallback).",
                         self.name)
            return [convs[-1]]
        raise BackboneUnavailable(
            f"No Grad-CAM target layer found in {self.name}."
        )
