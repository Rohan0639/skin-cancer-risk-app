"""End-to-end analysis pipeline.

:class:`LesionAnalyzer` is the single entry point used by the Streamlit app,
the CLI demo and the end-to-end tests. It runs the components in the order that
matters:

1. **Quality gate** -- refuse an unusable photo before anything else, because a
   confident-looking number computed from a blurred image is the worst possible
   output for this kind of tool.
2. **Segmentation** -- one lesion mask, shared by everything downstream.
3. **Risk prediction** -- backbone embedding + metadata -> risk class.
4. **Change detection** -- only if a prior photo was supplied.
5. **ABCDE** -- classical measurements, independent of the network.
6. **Grad-CAM** -- where the network actually looked.
7. **Report** -- one document, one recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.config import Config, load_config
from src.explain.abcde import ABCDEReport, compute_abcde
from src.explain.gradcam import GradCAMResult, compute_gradcam
from src.model.risk_classifier import RiskClassifier, RiskPrediction, load_risk_classifier
from src.quality.image_quality import QualityReport, assess_image_quality
from src.report.report_generator import RiskReport, build_report
from src.segmentation import LesionMask, segment_lesion
from src.tracking.change_detection import ChangeResult, detect_change
from src.utils import ImageLike, get_logger, load_image_rgb, resolve_device

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    """Everything produced by one run of the pipeline.

    Attributes:
        quality: The image-quality report (always present).
        quality_passed: Convenience mirror of ``quality.passed``.
        report: The assembled risk report, or ``None`` if analysis stopped at
            the quality gate.
        prediction: The raw model prediction, if it ran.
        abcde: The ABCDE report, if computed.
        gradcam: The Grad-CAM result, if computed.
        change: The change-detection result, if a prior photo was supplied.
        mask: The lesion segmentation, if computed.
        image: The loaded RGB image that was analysed.
    """

    quality: QualityReport
    quality_passed: bool
    report: Optional[RiskReport] = None
    prediction: Optional[RiskPrediction] = None
    abcde: Optional[ABCDEReport] = None
    gradcam: Optional[GradCAMResult] = None
    change: Optional[ChangeResult] = None
    mask: Optional[LesionMask] = None
    image: Optional[np.ndarray] = None

    @property
    def stopped_at_quality_gate(self) -> bool:
        """``True`` when the photo was rejected and no analysis was performed."""
        return self.report is None


class LesionAnalyzer:
    """Runs the full screening pipeline on lesion photos.

    Loading the backbone takes several seconds, so construct one analyzer and
    reuse it across photos rather than building one per request.
    """

    def __init__(
        self,
        cfg: Optional[Config] = None,
        checkpoint: Optional[Path] = None,
        strict: bool = False,
    ) -> None:
        """
        Args:
            cfg: Project config. Loaded from the default path if omitted.
            checkpoint: Trained head checkpoint. Defaults to the configured one.
            strict: When ``True``, a missing checkpoint raises. When ``False``
                (default) an untrained model is loaded so the UI can render a
                clear "train the model first" message instead of crashing.

        Raises:
            FileNotFoundError: If ``strict`` and no checkpoint exists.
        """
        self.cfg = cfg or load_config()
        self.device = resolve_device(self.cfg.get("project.device", "auto"))
        self.model: RiskClassifier = load_risk_classifier(
            checkpoint, cfg=self.cfg, device=self.device, strict=strict
        )
        self.is_trained = checkpoint is not None or self._checkpoint_exists()
        logger.info("Analyzer ready on %s using %s",
                    self.device, self.model.backbone.describe())

    def _checkpoint_exists(self) -> bool:
        """Return whether the configured default checkpoint is on disk."""
        path = (self.cfg.path("paths.checkpoint_dir")
                / str(self.cfg.get("train.checkpoint_name", "risk_head.pt")))
        return path.exists()

    def analyse(
        self,
        image: ImageLike,
        age: Any = None,
        sex: Any = None,
        site: Any = None,
        previous_image: Optional[ImageLike] = None,
        frame_width_mm: Optional[float] = None,
        require_quality: bool = True,
        explain: bool = True,
    ) -> AnalysisResult:
        """Analyse one lesion photo end to end.

        Args:
            image: The photo to analyse (path, array, PIL image or upload).
            age: Patient age in years.
            sex: Patient sex.
            site: Anatomical site of the lesion.
            previous_image: An earlier photo of the same lesion, enabling the
                "Evolving" criterion and change detection.
            frame_width_mm: Real-world width of the photo in millimetres, if
                known. Without it the Diameter criterion cannot be measured.
            require_quality: When ``True`` (default), a photo that fails the
                quality gate stops the pipeline and the caller should prompt
                for a retake. Set ``False`` to analyse anyway, in which case
                the report carries a prominent caution.
            explain: Whether to compute Grad-CAM. Disable for speed when only
                the risk level is needed.

        Returns:
            An :class:`AnalysisResult`. Check
            :attr:`AnalysisResult.stopped_at_quality_gate` before using the
            report.
        """
        rgb = load_image_rgb(image)

        quality = assess_image_quality(rgb, self.cfg)
        if require_quality and not quality.passed:
            logger.info("Stopping at the quality gate: %s", quality.reason)
            return AnalysisResult(quality=quality, quality_passed=False, image=rgb)

        mask = segment_lesion(rgb)
        prediction = self.model.predict(rgb, age=age, sex=sex, site=site)

        change: Optional[ChangeResult] = None
        if previous_image is not None:
            previous_rgb = load_image_rgb(previous_image)
            change = detect_change(
                previous_rgb, rgb, self.cfg,
                backbone=self.model.backbone, current_mask=mask,
            )

        abcde = compute_abcde(
            rgb, self.cfg, mask=mask,
            change_summary=change.to_dict() if change is not None else None,
            frame_width_mm=frame_width_mm,
        )

        gradcam: Optional[GradCAMResult] = None
        if explain and bool(self.cfg.get("explain.gradcam.enabled", True)):
            try:
                gradcam = compute_gradcam(
                    self.model, rgb, self.cfg,
                    age=age, sex=sex, site=site, lesion_mask=mask.mask,
                )
            except Exception as exc:  # noqa: BLE001 - never lose the whole report
                logger.warning("Grad-CAM failed (%s); continuing without a heatmap.", exc)

        report = build_report(
            prediction=prediction,
            cfg=self.cfg,
            quality=quality,
            abcde=abcde,
            gradcam=gradcam,
            change=change,
            backbone_description=self.model.backbone.describe(),
            backbone_is_fallback=self.model.backbone.is_fallback,
            metadata={
                "age": age,
                "sex": sex,
                "site": site,
                "compared with earlier photo": "yes" if previous_image is not None else "no",
            },
        )

        if not self.is_trained:
            report.notes.insert(0, (
                "No trained model was found, so this risk level came from an "
                "UNTRAINED network and is meaningless. Run "
                "`python src/model/train.py` first."
            ))

        return AnalysisResult(
            quality=quality,
            quality_passed=quality.passed,
            report=report,
            prediction=prediction,
            abcde=abcde,
            gradcam=gradcam,
            change=change,
            mask=mask,
            image=rgb,
        )
