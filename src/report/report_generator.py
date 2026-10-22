"""Assembling one patient-facing risk report.

This module turns the outputs of every other component -- quality gate, risk
model, Grad-CAM, ABCDE heuristics, change detection -- into a single document
with exactly one recommendation at the top.

Two design rules run through it:

1. **Escalate, never de-escalate.** Corroborating evidence (a changing lesion,
   several ABCDE flags) can only push the advice *towards* seeing a doctor.
   Nothing here can talk a worried user out of a consultation, and a
   low-confidence prediction is treated as a reason for more caution, not less.
2. **Say what could not be measured.** A criterion that was not assessable is
   reported as such rather than silently dropped, so absence of a warning is
   never mistaken for a reassurance.
"""

from __future__ import annotations

import base64
import html
import io
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import Config
from src.explain.abcde import ABCDEReport
from src.explain.gradcam import GradCAMResult, attention_sanity_note
from src.model.risk_classifier import RiskPrediction
from src.quality.image_quality import QualityReport
from src.tracking.change_detection import ChangeResult
from src.utils import get_logger

logger = get_logger(__name__)

#: Ordered from least to most urgent; used by the escalation logic.
RISK_ORDER = ["Low", "Moderate", "High"]


@dataclass
class RiskReport:
    """A complete screening report for one lesion.

    Attributes:
        risk_level: The reported risk level *after* any escalation.
        model_risk_level: What the model alone predicted, before escalation.
        confidence: The model's confidence in ``model_risk_level``.
        probabilities: Full probability distribution over risk classes.
        recommendation: The single action the user should take.
        escalation_reasons: Why the reported level differs from the model's, if
            it does.
        quality: The image-quality report.
        abcde: The ABCDE heuristic report.
        gradcam: The Grad-CAM explanation, if computed.
        change: The change-detection result, if a prior photo was given.
        backbone_description: Which feature extractor produced the embedding.
        backbone_is_fallback: Whether that was the non-dermatology fallback.
        metadata: The patient details the report was generated from.
        created_at: ISO-8601 timestamp.
        notes: Extra cautions surfaced to the user.
    """

    risk_level: str
    model_risk_level: str
    confidence: float
    probabilities: Dict[str, float]
    recommendation: str
    disclaimer: str
    escalation_reasons: List[str] = field(default_factory=list)
    quality: Optional[QualityReport] = None
    abcde: Optional[ABCDEReport] = None
    gradcam: Optional[GradCAMResult] = None
    change: Optional[ChangeResult] = None
    backbone_description: str = ""
    backbone_is_fallback: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of the entire report."""
        return {
            "created_at": self.created_at,
            "risk_level": self.risk_level,
            "model_risk_level": self.model_risk_level,
            "confidence": round(float(self.confidence), 4),
            "probabilities": {k: round(float(v), 4) for k, v in self.probabilities.items()},
            "recommendation": self.recommendation,
            "escalation_reasons": self.escalation_reasons,
            "notes": self.notes,
            "metadata": self.metadata,
            "backbone": {
                "description": self.backbone_description,
                "is_fallback": self.backbone_is_fallback,
            },
            "quality": self.quality.to_dict() if self.quality else None,
            "abcde": self.abcde.to_dict() if self.abcde else None,
            "gradcam": self.gradcam.to_dict() if self.gradcam else None,
            "change": self.change.to_dict() if self.change else None,
            "disclaimer": self.disclaimer,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return the report as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    # -- rendering ---------------------------------------------------------
    def to_markdown(self, include_images: bool = False) -> str:
        """Render the report as Markdown.

        Args:
            include_images: Embed the Grad-CAM overlay as a base64 data URI.
                Off by default to keep the text readable in a terminal.
        """
        lines: List[str] = []
        add = lines.append

        add("# Skin lesion screening report")
        add("")
        add(f"*Generated {self.created_at}*")
        add("")
        add("> **This is not a diagnosis.** " + self.disclaimer.strip())
        add("")

        add("## Result")
        add("")
        add(f"**Risk level: {self.risk_level}**")
        add("")
        add(f"**What to do: {self.recommendation}**")
        add("")
        add(f"- Model confidence: {self.confidence:.0%} "
            f"(predicted {self.model_risk_level})")
        breakdown = ", ".join(f"{k} {v:.0%}" for k, v in self.probabilities.items())
        add(f"- Full breakdown: {breakdown}")
        if self.escalation_reasons:
            add(f"- Advice raised above the model's own prediction because: "
                f"{'; '.join(self.escalation_reasons)}")
        add("")

        if self.notes:
            add("## Please note")
            add("")
            for note in self.notes:
                add(f"- {note}")
            add("")

        if self.abcde is not None:
            add("## What we looked at (the ABCDE rule)")
            add("")
            add(f"{self.abcde.summary()}")
            add("")
            for feature in self.abcde.features:
                if not feature.measurable:
                    marker = "?"
                elif feature.flagged:
                    marker = "!"
                else:
                    marker = "-"
                # The sentence starts with "Asymmetry: ..."; the letter and
                # title already say that, so keep only the explanation.
                detail = feature.sentence.split(":", 1)[-1].strip()
                detail = detail[:1].upper() + detail[1:]
                add(f"- **[{marker}] {feature.letter} - {feature.title}.** {detail}")
            add("")
            add("Key: `!` = worth attention, `-` = looks unremarkable, "
                "`?` = could not be measured.")
            add("")

        if self.change is not None:
            add("## Compared with your earlier photo")
            add("")
            verdict = "Meaningful change detected" if self.change.changed else "No significant change"
            add(f"**{verdict}.** {self.change.reason}")
            add("")

        if self.gradcam is not None:
            add("## Where the model looked")
            add("")
            ratio = self.gradcam.focus_ratio
            if ratio is not None:
                add(f"{self.gradcam.peak_inside_lesion:.0%} of the model's attention fell "
                    f"on the lesion, which covers {self.gradcam.lesion_area_fraction:.0%} "
                    f"of the photo -- {ratio:.1f}x more concentrated than chance.")
            else:
                add("The heatmap below shows which parts of the photo drove the result.")
            add("")
            if include_images:
                add(f"![Grad-CAM overlay]({encode_image_data_uri(self.gradcam.overlay)})")
                add("")

        if self.quality is not None:
            add("## Photo quality")
            add("")
            add(f"{self.quality.reason}")
            add("")

        add("## About this tool")
        add("")
        add(f"- Image features: {self.backbone_description}")
        if self.backbone_is_fallback:
            add("- **Note:** a general-purpose ImageNet model was used, not the "
                "PanDerm dermatology model. Results are correspondingly weaker.")
        if self.metadata:
            details = ", ".join(f"{k}: {v}" for k, v in self.metadata.items() if v not in (None, ""))
            if details:
                add(f"- Details you gave us: {details}")
        add("")
        add("---")
        add("")
        add(self.disclaimer.strip())
        add("")
        return "\n".join(lines)

    def to_html(self, include_images: bool = True) -> str:
        """Render the report as a self-contained HTML fragment.

        Args:
            include_images: Embed the Grad-CAM overlay as a base64 data URI.
        """
        colour = {"Low": "#2e7d32", "Moderate": "#ef6c00", "High": "#c62828"}.get(
            self.risk_level, "#455a64")

        parts: List[str] = []
        add = parts.append

        add('<div class="scd-report" style="font-family:system-ui,-apple-system,'
            'Segoe UI,Roboto,sans-serif;line-height:1.55;max-width:52rem;">')
        add('<div style="border:2px solid #c62828;background:#fff3f3;color:#7f1d1d;'
            'padding:0.75rem 1rem;border-radius:8px;margin-bottom:1rem;">'
            f'<strong>This is not a diagnosis.</strong> {html.escape(self.disclaimer.strip())}'
            '</div>')

        add(f'<div style="border-left:6px solid {colour};background:rgba(0,0,0,0.03);'
            'padding:1rem 1.25rem;border-radius:6px;margin-bottom:1rem;">')
        add(f'<div style="font-size:0.85rem;letter-spacing:0.08em;text-transform:uppercase;'
            f'opacity:0.7;">Risk level</div>')
        add(f'<div style="font-size:2rem;font-weight:700;color:{colour};">'
            f'{html.escape(self.risk_level)}</div>')
        add(f'<p style="margin:0.5rem 0 0;font-size:1.05rem;"><strong>What to do:</strong> '
            f'{html.escape(self.recommendation)}</p>')
        add('</div>')

        breakdown = ", ".join(f"{html.escape(k)} {v:.0%}" for k, v in self.probabilities.items())
        add(f'<p><strong>Model confidence:</strong> {self.confidence:.0%} '
            f'(predicted {html.escape(self.model_risk_level)})<br>'
            f'<strong>Full breakdown:</strong> {breakdown}</p>')

        if self.escalation_reasons:
            add('<p style="background:#fff8e1;border:1px solid #ffe082;padding:0.6rem 0.9rem;'
                'border-radius:6px;"><strong>Advice raised above the model\'s own '
                'prediction because:</strong> '
                + html.escape("; ".join(self.escalation_reasons)) + '</p>')

        if self.notes:
            add('<h2>Please note</h2><ul>')
            for note in self.notes:
                add(f'<li>{html.escape(note)}</li>')
            add('</ul>')

        if self.abcde is not None:
            add('<h2>What we looked at (the ABCDE rule)</h2>')
            add(f'<p>{html.escape(self.abcde.summary())}</p><ul>')
            for feature in self.abcde.features:
                if not feature.measurable:
                    badge, tint = "not measured", "#607d8b"
                elif feature.flagged:
                    badge, tint = "worth attention", "#c62828"
                else:
                    badge, tint = "looks fine", "#2e7d32"
                add(f'<li style="margin-bottom:0.4rem;">'
                    f'<span style="display:inline-block;min-width:1.4rem;font-weight:700;">'
                    f'{html.escape(feature.letter)}</span> '
                    f'<strong>{html.escape(feature.title)}</strong> '
                    f'<span style="color:{tint};font-size:0.85rem;">({badge})</span><br>'
                    f'{html.escape(feature.sentence)}</li>')
            add('</ul>')

        if self.change is not None:
            verdict = ("Meaningful change detected" if self.change.changed
                       else "No significant change")
            add('<h2>Compared with your earlier photo</h2>')
            add(f'<p><strong>{html.escape(verdict)}.</strong> '
                f'{html.escape(self.change.reason)}</p>')

        if self.gradcam is not None:
            add('<h2>Where the model looked</h2>')
            if include_images:
                add(f'<img src="{encode_image_data_uri(self.gradcam.overlay)}" '
                    'alt="Grad-CAM heatmap over the lesion" '
                    'style="max-width:100%;border-radius:8px;">')
            if self.gradcam.peak_inside_lesion is not None:
                add(f'<p>{self.gradcam.peak_inside_lesion:.0%} of the model\'s attention '
                    'fell on the lesion itself.</p>')

        if self.quality is not None:
            add('<h2>Photo quality</h2>')
            add(f'<p>{html.escape(self.quality.reason)}</p>')

        add('<h2>About this tool</h2><ul>')
        add(f'<li>Image features: {html.escape(self.backbone_description)}</li>')
        if self.backbone_is_fallback:
            add('<li><strong>Note:</strong> a general-purpose ImageNet model was used, '
                'not the PanDerm dermatology model. Results are correspondingly weaker.</li>')
        if self.metadata:
            details = ", ".join(f"{k}: {v}" for k, v in self.metadata.items()
                                if v not in (None, ""))
            if details:
                add(f'<li>Details you gave us: {html.escape(details)}</li>')
        add('</ul>')

        add(f'<hr><p style="font-size:0.9rem;opacity:0.8;">'
            f'{html.escape(self.disclaimer.strip())}</p>')
        add(f'<p style="font-size:0.8rem;opacity:0.6;">Generated {self.created_at}</p>')
        add('</div>')
        return "\n".join(parts)

    def save(self, directory: Path, stem: str = "report") -> Dict[str, Path]:
        """Write the report as Markdown, HTML and JSON.

        Args:
            directory: Destination directory, created if needed.
            stem: Base filename without extension.

        Returns:
            A mapping of format name to the path written.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written: Dict[str, Path] = {}

        for suffix, content in (
            ("md", self.to_markdown(include_images=True)),
            ("html", self.to_html(include_images=True)),
            ("json", self.to_json()),
        ):
            path = directory / f"{stem}.{suffix}"
            path.write_text(content, encoding="utf-8")
            written[suffix] = path

        logger.info("Wrote report -> %s", ", ".join(str(p) for p in written.values()))
        return written


def encode_image_data_uri(rgb: np.ndarray, fmt: str = "PNG") -> str:
    """Encode an RGB array as a base64 ``data:`` URI for embedding in HTML/MD."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def _escalate(level: str, steps: int = 1) -> str:
    """Return the risk level ``steps`` positions more urgent than ``level``."""
    try:
        index = RISK_ORDER.index(level)
    except ValueError:
        return level
    return RISK_ORDER[min(index + steps, len(RISK_ORDER) - 1)]


def build_report(
    prediction: RiskPrediction,
    cfg: Config,
    quality: Optional[QualityReport] = None,
    abcde: Optional[ABCDEReport] = None,
    gradcam: Optional[GradCAMResult] = None,
    change: Optional[ChangeResult] = None,
    backbone_description: str = "",
    backbone_is_fallback: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> RiskReport:
    """Combine every component's output into one report with one recommendation.

    The reported risk level starts at the model's prediction and is raised (never
    lowered) when independent evidence supports more caution:

    * the model was not confident in its own answer;
    * the lesion has measurably changed since an earlier photo;
    * three or more ABCDE criteria were flagged;
    * the model's attention mostly missed the lesion.

    Args:
        prediction: The model's risk prediction.
        cfg: Loaded project config (reads the ``report:`` section).
        quality: Image-quality report, if run.
        abcde: ABCDE report, if computed.
        gradcam: Grad-CAM result, if computed.
        change: Change-detection result, if a prior photo was supplied.
        backbone_description: Human-readable backbone description.
        backbone_is_fallback: Whether the fallback backbone was used.
        metadata: Patient details to echo back in the report.

    Returns:
        A fully populated :class:`RiskReport`.
    """
    recommendations = dict(cfg.get("report.recommendations"))
    disclaimer = str(cfg.get("report.disclaimer"))

    level = prediction.label
    reasons: List[str] = []
    notes: List[str] = []

    if prediction.low_confidence:
        threshold = float(cfg.get("report.low_confidence_threshold", 0.5))
        level = _escalate(level)
        reasons.append(
            f"the model was only {prediction.confidence:.0%} confident, below the "
            f"{threshold:.0%} mark we treat as reliable"
        )

    if change is not None and change.changed:
        level = _escalate(level)
        reasons.append("the lesion has visibly changed since your earlier photo")

    if abcde is not None and abcde.flag_count >= 3:
        level = _escalate(level)
        letters = ", ".join(f.letter for f in abcde.flagged)
        reasons.append(f"{abcde.flag_count} ABCDE features were flagged ({letters})")

    if gradcam is not None:
        note_threshold = float(cfg.get("explain.gradcam.min_focus_ratio_note", 1.2))
        escalate_threshold = float(
            cfg.get("explain.gradcam.min_focus_ratio_escalate", 1.0))
        note = attention_sanity_note(gradcam, min_focus_ratio=note_threshold)
        if note:
            notes.append(note)
            # Telling the user the heatmap looks unfocused is cheap and always
            # worth doing. Raising their risk level is not: reserve that for
            # attention that is genuinely *worse than chance*, otherwise a
            # merely-mediocre heatmap escalates a large share of all reports
            # and the escalation stops carrying information.
            ratio = gradcam.focus_ratio
            if ratio is not None and ratio < escalate_threshold:
                level = _escalate(level)
                reasons.append(
                    "the model's attention fell on the lesion less than chance "
                    "alone would predict"
                )

    if abcde is not None and not abcde.segmentation_confident:
        notes.append(
            "We could not trace the lesion's outline confidently, so the ABCDE "
            "measurements above are rough. A clearer, better-lit photo would help."
        )

    if quality is not None and not quality.passed:
        notes.append(
            "This photo did not pass our quality checks, so the result is less "
            f"reliable than usual. {quality.reason}"
        )

    if backbone_is_fallback:
        notes.append(
            "This result came from a general-purpose image model rather than the "
            "PanDerm dermatology model, because PanDerm's weights were not "
            "available. Treat the risk level as a rough indication only."
        )

    if change is None:
        notes.append(
            "You did not upload an earlier photo, so we could not check whether "
            "this lesion is changing -- one of the strongest warning signs there "
            "is. Take a photo now and compare again in about 3 months."
        )

    return RiskReport(
        risk_level=level,
        model_risk_level=prediction.label,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        recommendation=recommendations.get(level, recommendations.get("High", "")),
        disclaimer=disclaimer,
        escalation_reasons=reasons,
        quality=quality,
        abcde=abcde,
        gradcam=gradcam,
        change=change,
        backbone_description=backbone_description,
        backbone_is_fallback=backbone_is_fallback,
        metadata=metadata or {},
        notes=notes,
    )
