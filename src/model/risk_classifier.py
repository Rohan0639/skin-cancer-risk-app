"""End-to-end risk classifier: backbone + fusion head, plus checkpoint I/O.

The backbone is kept **frozen** and only the fusion head is trained. This is
the linear-probe setup PanDerm itself is evaluated with upstream, and it is
what makes the project trainable on a laptop CPU: the expensive embeddings can
be computed once and cached, after which training a small MLP takes seconds.

Set :attr:`RiskClassifier.freeze_backbone` to ``False`` to fine-tune end to
end -- worthwhile on a GPU with the full datasets, painfully slow on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.backbone import FeatureBackbone, build_backbone
from src.config import Config, load_config
from src.dataset import (
    diagnosis_classes,
    diagnosis_description,
    diagnosis_name,
    encode_metadata,
    metadata_dim,
    risk_for,
)
from src.model.fusion import FusionHead
from src.utils import get_logger, resolve_device

logger = get_logger(__name__)


@dataclass
class RiskPrediction:
    """What the model thinks the lesion is, and what that implies.

    The network predicts a diagnosis; the risk tier is looked up from it, so
    the two are always consistent.

    Attributes:
        diagnosis: Predicted lesion-type code, e.g. ``"BCC"``.
        diagnosis_name: Patient-facing name, e.g. ``"Basal cell carcinoma"``.
        diagnosis_description: One-line plain-language explanation.
        diagnosis_confidence: Probability assigned to ``diagnosis``.
        diagnosis_probabilities: Probability for every lesion type.
        label: The risk tier implied by ``diagnosis``.
        confidence: Total probability of all diagnoses in that tier. This is
            the number that matters for the advice: three malignant diagnoses
            at 30% each mean high risk, even though no single one is likely.
        probabilities: Probability per risk tier.
        features: The backbone embedding, kept for change detection.
        low_confidence: ``True`` when ``confidence`` is below the configured
            threshold, meaning the advice should be escalated, not trusted.
    """

    diagnosis: str
    diagnosis_name: str
    diagnosis_confidence: float
    diagnosis_probabilities: Dict[str, float]
    label: str
    confidence: float
    probabilities: Dict[str, float]
    diagnosis_description: str = ""
    features: Optional[np.ndarray] = field(default=None, repr=False)
    low_confidence: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view (excluding the raw features)."""
        return {
            "diagnosis": self.diagnosis,
            "diagnosis_name": self.diagnosis_name,
            "diagnosis_description": self.diagnosis_description,
            "diagnosis_confidence": round(float(self.diagnosis_confidence), 4),
            "diagnosis_probabilities": {
                k: round(float(v), 4) for k, v in self.diagnosis_probabilities.items()},
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "probabilities": {k: round(float(v), 4) for k, v in self.probabilities.items()},
            "low_confidence": bool(self.low_confidence),
        }


class RiskClassifier(nn.Module):
    """Composes a :class:`FeatureBackbone` with a :class:`FusionHead`.

    Attributes:
        backbone: The (frozen) feature extractor.
        head: The trainable fusion + classification head.
        classes: Ordered risk class names.
    """

    def __init__(
        self,
        backbone: FeatureBackbone,
        cfg: Config,
        freeze_backbone: bool = True,
    ) -> None:
        """
        Args:
            backbone: A constructed feature backbone.
            cfg: Loaded project config.
            freeze_backbone: When ``True`` (default) backbone parameters get
                ``requires_grad=False`` and are excluded from the optimiser.
        """
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        # The head predicts lesion types; risk is derived from them.
        self.classes: List[str] = diagnosis_classes(cfg)
        self.freeze_backbone = freeze_backbone

        self.head = FusionHead(
            feature_dim=backbone.feature_dim,
            metadata_dim=metadata_dim(cfg),
            num_classes=len(self.classes),
            image_proj_dim=int(cfg.get("model.image_proj_dim", 256)),
            metadata_embed_dim=int(cfg.get("model.metadata_embed_dim", 32)),
            hidden_dims=list(cfg.get("model.hidden_dims", [256, 128])),
            dropout=float(cfg.get("model.dropout", 0.3)),
            use_metadata=bool(cfg.get("model.use_metadata", True)),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    # -- forward paths -----------------------------------------------------
    def forward(self, images: torch.Tensor,
                metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Classify a batch of images end to end.

        Args:
            images: ``(B, 3, H, W)`` normalised image batch.
            metadata: ``(B, M)`` encoded metadata.

        Returns:
            ``(B, num_classes)`` logits.
        """
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(images)
            features = features.detach()
        else:
            features = self.backbone(images)
        return self.head(features, metadata)

    def forward_from_features(self, features: torch.Tensor,
                              metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Classify from precomputed embeddings, skipping the backbone."""
        return self.head(features, metadata)

    def trainable_parameters(self):
        """Yield only the parameters the optimiser should update."""
        return (p for p in self.parameters() if p.requires_grad)

    def train(self, mode: bool = True) -> "RiskClassifier":
        """Override so a frozen backbone stays in eval mode (fixed BN stats)."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    # -- inference ---------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        rgb: np.ndarray,
        age: Any = None,
        sex: Any = None,
        site: Any = None,
    ) -> RiskPrediction:
        """Predict the risk class for one photo plus its metadata.

        Args:
            rgb: HxWx3 uint8 RGB image.
            age: Patient age in years, or ``None``.
            sex: Patient sex string, or ``None``.
            site: Anatomical site string, or ``None``.

        Returns:
            A :class:`RiskPrediction` including the backbone embedding, which
            the change-detection module reuses.
        """
        from PIL import Image

        self.eval()
        device = next(self.parameters()).device
        transform = self.backbone.build_transform(train=False)
        image_t = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)

        features = self.backbone(image_t)
        meta_np = encode_metadata(age, sex, site, self.cfg)
        meta_t = torch.from_numpy(meta_np).unsqueeze(0).to(device)

        logits = self.head(features, meta_t)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        best = int(np.argmax(probs))
        code = self.classes[best]
        dx_probs = {c: float(p) for c, p in zip(self.classes, probs)}

        # Risk probability is the total mass of every diagnosis in that tier,
        # not the top diagnosis alone.
        risk_probs: Dict[str, float] = {
            r: 0.0 for r in self.cfg.get("labels.risk_classes")}
        for c, p in dx_probs.items():
            risk_probs[risk_for(self.cfg, c)] = risk_probs.get(
                risk_for(self.cfg, c), 0.0) + p

        risk = risk_for(self.cfg, code)
        threshold = float(self.cfg.get("report.low_confidence_threshold", 0.5))

        return RiskPrediction(
            diagnosis=code,
            diagnosis_name=diagnosis_name(self.cfg, code),
            diagnosis_description=diagnosis_description(self.cfg, code),
            diagnosis_confidence=float(probs[best]),
            diagnosis_probabilities=dx_probs,
            label=risk,
            confidence=float(risk_probs.get(risk, 0.0)),
            probabilities=risk_probs,
            features=features.squeeze(0).cpu().numpy(),
            low_confidence=float(risk_probs.get(risk, 0.0)) < threshold,
        )

    # -- checkpoint I/O ----------------------------------------------------
    def save(self, path: Path, extra: Optional[Dict[str, Any]] = None) -> Path:
        """Save the trained head plus everything needed to rebuild the model.

        Only the head's weights are stored -- the backbone is reconstructed
        from the config on load, which keeps checkpoints at a few megabytes
        instead of hundreds.

        Args:
            path: Destination ``.pt`` file.
            extra: Extra metadata to embed (training history, metrics, ...).

        Returns:
            The path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "head_state_dict": self.head.state_dict(),
            "classes": self.classes,
            "feature_dim": self.backbone.feature_dim,
            "backbone_name": self.backbone.name,
            "backbone_is_fallback": self.backbone.is_fallback,
            "backbone_is_reproducible": getattr(self.backbone, "is_reproducible", True),
            "metadata_dim": metadata_dim(self.cfg),
            "config": self.cfg.as_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)
        if not getattr(self.backbone, "is_reproducible", True):
            logger.warning(
                "Saving a head trained on a NON-REPRODUCIBLE backbone (%s). Only "
                "the head is stored; the backbone is rebuilt from config, and a "
                "randomly initialised one differs every time, so this checkpoint "
                "will not reproduce its predictions.", self.backbone.name,
            )
        logger.info("Saved checkpoint -> %s", path)
        return path


def load_risk_classifier(
    checkpoint_path: Optional[Path] = None,
    cfg: Optional[Config] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> RiskClassifier:
    """Rebuild a :class:`RiskClassifier` and load trained head weights.

    Args:
        checkpoint_path: ``.pt`` file written by :meth:`RiskClassifier.save`.
            Defaults to ``<checkpoint_dir>/<train.checkpoint_name>``.
        cfg: Config to use. Defaults to the project config.
        device: Device to place the model on. Defaults to the configured one.
        strict: When ``True``, a missing checkpoint raises. When ``False``, an
            untrained model is returned instead -- used by the app so it can
            show a clear "train the model first" message rather than crashing.

    Returns:
        The classifier in eval mode.

    Raises:
        FileNotFoundError: If ``strict`` and the checkpoint does not exist.
        ValueError: If the checkpoint's backbone dimensions disagree with the
            backbone the current config builds.
    """
    cfg = cfg or load_config()
    device = device or resolve_device(cfg.get("project.device", "auto"))

    if checkpoint_path is None:
        checkpoint_path = (cfg.path("paths.checkpoint_dir")
                           / str(cfg.get("train.checkpoint_name", "risk_head.pt")))
    checkpoint_path = Path(checkpoint_path)

    backbone = build_backbone(cfg)
    model = RiskClassifier(backbone, cfg)

    if not checkpoint_path.exists():
        message = (f"No trained checkpoint at {checkpoint_path}. "
                   "Run: python src/model/train.py")
        if strict:
            raise FileNotFoundError(message)
        logger.warning("%s Returning an UNTRAINED model -- its predictions are "
                       "meaningless.", message)
        return model.to(device).eval()

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    saved_dim = int(payload.get("feature_dim", backbone.feature_dim))
    if saved_dim != backbone.feature_dim:
        raise ValueError(
            f"Checkpoint was trained on a {saved_dim}-dim backbone "
            f"({payload.get('backbone_name')}) but the current config builds a "
            f"{backbone.feature_dim}-dim one ({backbone.name}). Retrain, or "
            "point backbone.backend at the backbone used for training."
        )

    if not payload.get("backbone_is_reproducible", True):
        logger.warning(
            "This checkpoint was trained on a randomly initialised backbone, "
            "which cannot be reconstructed. Its predictions will NOT match the "
            "trained model. Set backbone.fallback.pretrained: true and retrain."
        )

    model.head.load_state_dict(payload["head_state_dict"])
    model.classes = list(payload.get("classes", model.classes))
    logger.info("Loaded checkpoint %s (backbone=%s)",
                checkpoint_path.name, payload.get("backbone_name"))
    return model.to(device).eval()


class GradCAMTarget(nn.Module):
    """Adapts :class:`RiskClassifier` to the image-only interface Grad-CAM needs.

    ``pytorch_grad_cam`` calls ``model(input_tensor)`` with a single argument,
    so the patient metadata is bound at construction time.
    """

    def __init__(self, model: RiskClassifier, metadata: torch.Tensor) -> None:
        """
        Args:
            model: The trained classifier.
            metadata: ``(1, M)`` encoded metadata to hold fixed.
        """
        super().__init__()
        self.model = model
        self.register_buffer("metadata", metadata)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits for ``images``, repeating the bound metadata."""
        features = self.model.backbone(images)
        meta = self.metadata.expand(images.shape[0], -1)
        return self.model.head(features, meta)
