"""Backbone factory.

``build_backbone(cfg)`` is the only entry point the rest of the project uses.
Which model it returns is decided entirely by ``backbone.backend`` in the
config: ``panderm``, ``fallback``, or ``auto`` (try PanDerm, fall back).
"""

from __future__ import annotations

from src.backbone.base import BackboneUnavailable, FeatureBackbone, default_transform
from src.config import Config
from src.utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "BackboneUnavailable",
    "FeatureBackbone",
    "build_backbone",
    "default_transform",
]


def build_backbone(cfg: Config) -> FeatureBackbone:
    """Construct the configured feature-extraction backbone.

    Args:
        cfg: Loaded project config. Reads ``backbone.backend``:

            * ``"panderm"``  -- require PanDerm; raise if it is unavailable.
            * ``"fallback"`` -- always use the ImageNet backbone.
            * ``"auto"``     -- try PanDerm, fall back with a loud warning.

    Returns:
        A ready-to-use :class:`FeatureBackbone` in eval mode.

    Raises:
        BackboneUnavailable: If ``backend`` is ``"panderm"`` and PanDerm cannot
            be loaded, or if even the fallback fails to build.
        ValueError: If ``backend`` is not one of the three accepted values.
    """
    backend = str(cfg.get("backbone.backend", "auto")).lower().strip()

    if backend not in {"panderm", "fallback", "auto"}:
        raise ValueError(
            f"Unknown backbone.backend {backend!r}; expected "
            "'panderm', 'fallback' or 'auto'."
        )

    if backend in {"panderm", "auto"}:
        from src.backbone.panderm_wrapper import PanDermBackbone

        try:
            backbone = PanDermBackbone(cfg)
            backbone.eval()
            return backbone
        except BackboneUnavailable as exc:
            if backend == "panderm":
                raise BackboneUnavailable(
                    f"backbone.backend is 'panderm' but PanDerm could not be "
                    f"loaded: {exc}\nSet backbone.backend to 'auto' in the config "
                    "to fall back automatically."
                ) from exc
            # backend == "auto": this is the expected path when the weights
            # have not been downloaded, so explain it clearly rather than
            # dumping a stack trace.
            logger.warning("=" * 74)
            logger.warning("PanDerm is not available -- FALLING BACK to an ImageNet backbone.")
            logger.warning("Reason: %s", exc)
            logger.warning("=" * 74)

    from src.backbone.fallback_backbone import FallbackBackbone

    backbone = FallbackBackbone(cfg)
    backbone.eval()
    return backbone
