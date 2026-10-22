"""Abstract interface every feature-extraction backbone implements.

The whole point of this abstraction is that PanDerm and the ImageNet fallback
are interchangeable: the fusion head, the training loop, Grad-CAM and the app
only ever talk to :class:`FeatureBackbone`. Swapping backbones is a one-line
config change (``backbone.backend``), not a code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, List

import numpy as np
import torch
import torch.nn as nn

from src.config import Config


class BackboneUnavailable(RuntimeError):
    """Raised when a backbone's weights or source code cannot be loaded.

    The factory catches this to trigger the fallback path, so it must be raised
    (rather than the process exiting) whenever PanDerm is not usable.
    """


class FeatureBackbone(nn.Module, ABC):
    """Maps a batch of preprocessed images to a batch of feature vectors.

    Attributes:
        name: Short identifier used in logs, checkpoints and reports.
        feature_dim: Length of the embedding returned by :meth:`forward`.
        is_fallback: ``True`` for the non-dermatology fallback backbone. The
            UI and reports surface this so a user is never misled about which
            model produced a result.
        is_reproducible: ``True`` when rebuilding this backbone from config
            yields identical weights every time (pretrained or checkpointed).
            ``False`` for a randomly initialised backbone, whose features
            change on every construction -- which would invalidate a saved
            head. Checkpoint loading warns when this is ``False``.
    """

    name: str = "abstract"
    feature_dim: int = 0
    is_fallback: bool = False
    is_reproducible: bool = True

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Embed a batch of images.

        Args:
            images: Float tensor of shape ``(B, 3, H, W)``, already normalised
                by this backbone's own transform.

        Returns:
            Float tensor of shape ``(B, feature_dim)``.
        """

    @abstractmethod
    def build_transform(self, train: bool = False) -> Callable[[Any], torch.Tensor]:
        """Return the torchvision transform mapping a PIL image to a tensor.

        Args:
            train: When ``True``, include data augmentation.
        """

    @abstractmethod
    def gradcam_target_layers(self) -> List[nn.Module]:
        """Return the layer(s) Grad-CAM should hook.

        Convention: the last spatial feature map before global pooling.
        """

    def gradcam_reshape_transform(self) -> Callable[[torch.Tensor], torch.Tensor] | None:
        """Return a reshape function for transformer activations, else ``None``.

        Grad-CAM expects ``(B, C, H, W)`` activations. CNNs already provide
        that; ViTs produce ``(B, tokens, C)`` and must be reshaped back onto a
        grid. CNN backbones can keep this default.
        """
        return None

    # -- convenience -------------------------------------------------------
    @torch.no_grad()
    def embed_image(self, rgb: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
        """Embed a single HxWx3 uint8 RGB array.

        Returns:
            A 1-D tensor of length :attr:`feature_dim` on the CPU.
        """
        from PIL import Image

        device = device or next(self.parameters()).device
        transform = self.build_transform(train=False)
        tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
        was_training = self.training
        self.eval()
        try:
            features = self.forward(tensor)
        finally:
            self.train(was_training)
        return features.squeeze(0).detach().cpu()

    def describe(self) -> str:
        """One-line human-readable description for logs and reports."""
        kind = "ImageNet fallback" if self.is_fallback else "dermatology foundation model"
        return f"{self.name} ({kind}, {self.feature_dim}-dim embedding)"


def default_transform(cfg: Config, train: bool = False,
                      mean: List[float] | None = None,
                      std: List[float] | None = None) -> Callable[[Any], torch.Tensor]:
    """Build the standard resize/normalise transform used by both backbones.

    Args:
        cfg: Loaded project config (reads ``backbone.image_size`` and the
            normalisation constants).
        train: When ``True``, adds flips, rotation and mild colour jitter.
            Lesion photos have no canonical orientation, so flips and rotations
            are label-preserving; colour jitter is kept mild because colour is
            a genuine diagnostic signal we do not want to destroy.
        mean: Override for the normalisation mean.
        std: Override for the normalisation std.
    """
    from torchvision import transforms

    size = int(cfg.get("backbone.image_size", 224))
    mean = mean if mean is not None else list(cfg.get("backbone.normalize_mean"))
    std = std if std is not None else list(cfg.get("backbone.normalize_std"))

    if train:
        steps = [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        ]
    else:
        steps = [transforms.Resize((size, size))]

    steps += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(steps)
