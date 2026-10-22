"""Fusing image embeddings with patient metadata.

Age, sex and anatomical site carry real diagnostic signal that a photo alone
does not: a pigmented lesion on the back of a 70-year-old is a different
proposition from the same lesion on a 20-year-old's arm. This module encodes
that metadata and concatenates it with the backbone embedding before the
classification head.

The two streams are projected to comparable widths before concatenation. Left
raw, the ~1000-dim image vector would numerically swamp the ~12-dim metadata
vector and the head would learn to ignore the metadata entirely.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class MetadataEncoder(nn.Module):
    """Encodes the one-hot/scaled metadata vector into a dense embedding.

    Attributes:
        input_dim: Length of the incoming metadata vector.
        output_dim: Length of the produced embedding.
    """

    def __init__(self, input_dim: int, output_dim: int = 32, dropout: float = 0.1) -> None:
        """
        Args:
            input_dim: Length of vectors from ``dataset.encode_metadata``.
            output_dim: Width of the embedding to produce.
            dropout: Dropout applied after the hidden activation.
        """
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, input_dim)`` metadata into ``(B, output_dim)``."""
        if metadata.ndim != 2 or metadata.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected metadata of shape (B, {self.input_dim}), "
                f"got {tuple(metadata.shape)}"
            )
        return self.net(metadata)


class FusionHead(nn.Module):
    """Projects the image embedding, fuses metadata, and classifies risk.

    Architecture::

        image  (B, F) -> Linear -> ReLU -> LayerNorm -> Dropout -> (B, P)
        meta   (B, M) -> MetadataEncoder                         -> (B, E)
                            concat -> MLP(hidden_dims) -> (B, n_classes)

    Attributes:
        feature_dim: Backbone embedding width ``F``.
        metadata_dim: Metadata vector width ``M`` (0 when metadata is disabled).
        num_classes: Number of risk classes.
        use_metadata: Whether the metadata stream is active.
    """

    def __init__(
        self,
        feature_dim: int,
        metadata_dim: int,
        num_classes: int,
        image_proj_dim: int = 256,
        metadata_embed_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        use_metadata: bool = True,
    ) -> None:
        """
        Args:
            feature_dim: Width of the backbone embedding.
            metadata_dim: Width of the encoded metadata vector.
            num_classes: Number of output risk classes.
            image_proj_dim: Width the image embedding is projected to.
            metadata_embed_dim: Width of the metadata embedding.
            hidden_dims: Widths of the MLP hidden layers after fusion.
            dropout: Dropout probability throughout.
            use_metadata: Set ``False`` to train an image-only baseline. Useful
                for measuring how much the metadata actually contributes.

        Raises:
            ValueError: If any dimension is non-positive.
        """
        super().__init__()
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        self.feature_dim = feature_dim
        self.metadata_dim = metadata_dim if use_metadata else 0
        self.num_classes = num_classes
        self.use_metadata = bool(use_metadata and metadata_dim > 0)

        self.image_proj = nn.Sequential(
            nn.Linear(feature_dim, image_proj_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(image_proj_dim),
            nn.Dropout(dropout),
        )

        fused_dim = image_proj_dim
        if self.use_metadata:
            self.metadata_encoder = MetadataEncoder(
                metadata_dim, metadata_embed_dim, dropout=min(dropout, 0.2)
            )
            fused_dim += metadata_embed_dim
        else:
            self.metadata_encoder = None

        layers: List[nn.Module] = []
        prev = fused_dim
        for width in (hidden_dims or [256, 128]):
            layers += [
                nn.Linear(prev, width),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(width),
                nn.Dropout(dropout),
            ]
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor,
                metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Produce class logits.

        Args:
            features: ``(B, feature_dim)`` backbone embeddings.
            metadata: ``(B, metadata_dim)`` encoded metadata. Required when
                ``use_metadata`` is ``True``.

        Returns:
            ``(B, num_classes)`` logits.

        Raises:
            ValueError: If shapes are wrong or metadata is missing when needed.
        """
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected features of shape (B, {self.feature_dim}), "
                f"got {tuple(features.shape)}"
            )

        fused = self.image_proj(features)
        if self.use_metadata:
            if metadata is None:
                raise ValueError("This head was built with use_metadata=True, "
                                 "so metadata must be provided.")
            fused = torch.cat([fused, self.metadata_encoder(metadata)], dim=1)

        # BatchNorm1d needs >1 sample in training mode; a single-sample forward
        # during inference is fine because eval() uses running statistics.
        if self.training and fused.shape[0] == 1:
            self.eval()
            try:
                return self.mlp(fused)
            finally:
                self.train()
        return self.mlp(fused)

    def output_dim(self) -> int:
        """Return the number of risk classes this head predicts."""
        return self.num_classes
