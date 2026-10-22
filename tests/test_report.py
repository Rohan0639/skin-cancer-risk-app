"""Tests for report assembly and rendering.

The escalation logic is the safety-critical part of this module: evidence may
push the advice towards seeing a doctor, and must never push it away.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.explain.abcde import ABCDEFeature, ABCDEReport
from src.explain.gradcam import GradCAMResult
from src.model.risk_classifier import RiskPrediction
from src.quality.image_quality import QualityCheck, QualityReport
from src.report.report_generator import (
    RISK_ORDER,
    build_report,
    encode_image_data_uri,
)
from src.tracking.change_detection import ChangeResult


def make_prediction(label: str = "Low", confidence: float = 0.9,
                    low_confidence: bool = False) -> RiskPrediction:
    """Build a RiskPrediction with a valid probability distribution."""
    remainder = (1.0 - confidence) / 2
    probabilities = {name: remainder for name in RISK_ORDER}
    probabilities[label] = confidence
    return RiskPrediction(label=label, confidence=confidence,
                          probabilities=probabilities, low_confidence=low_confidence)


def make_abcde(n_flagged: int = 0, confident: bool = True) -> ABCDEReport:
    """Build an ABCDE report with a chosen number of flagged criteria."""
    letters = ["A", "B", "C", "D", "E"]
    features = [
        ABCDEFeature(key=letter.lower(), letter=letter, title=f"Title {letter}",
                     score=0.5, threshold=0.3, flagged=(i < n_flagged),
                     sentence=f"Sentence for {letter} with enough length to read.")
        for i, letter in enumerate(letters)
    ]
    return ABCDEReport(features=features, segmentation_confident=confident)


def make_change(changed: bool) -> ChangeResult:
    """Build a ChangeResult with a chosen verdict."""
    return ChangeResult(changed=changed, score=0.8 if changed else 0.0,
                        threshold=0.35, reason="Because of reasons.", reliable=True)


def make_gradcam(inside: float = 0.7, area: float = 0.2) -> GradCAMResult:
    """Build a GradCAMResult with a chosen attention concentration."""
    return GradCAMResult(
        heatmap=np.zeros((8, 8), dtype=np.float32),
        overlay=np.zeros((8, 8, 3), dtype=np.uint8),
        target_class=0, target_class_name="Low",
        peak_inside_lesion=inside, lesion_area_fraction=area,
    )


class TestEscalation:
    def test_confident_prediction_is_not_escalated(self, cfg):
        report = build_report(make_prediction("Low", 0.95), cfg,
                              abcde=make_abcde(0), gradcam=make_gradcam())
        assert report.risk_level == "Low"
        assert report.model_risk_level == "Low"
        assert report.escalation_reasons == []

    def test_low_confidence_escalates(self, cfg):
        report = build_report(make_prediction("Low", 0.35, low_confidence=True), cfg,
                              abcde=make_abcde(0), gradcam=make_gradcam())
        assert report.risk_level == "Moderate"
        assert any("confident" in reason for reason in report.escalation_reasons)

    def test_detected_change_escalates(self, cfg):
        report = build_report(make_prediction("Low", 0.95), cfg,
                              abcde=make_abcde(0), gradcam=make_gradcam(),
                              change=make_change(True))
        assert report.risk_level == "Moderate"
        assert any("changed" in reason for reason in report.escalation_reasons)

    def test_many_abcde_flags_escalate(self, cfg):
        report = build_report(make_prediction("Low", 0.95), cfg,
                              abcde=make_abcde(3), gradcam=make_gradcam())
        assert report.risk_level == "Moderate"
        assert any("ABCDE" in reason for reason in report.escalation_reasons)

    def test_two_abcde_flags_do_not_escalate(self, cfg):
        report = build_report(make_prediction("Low", 0.95), cfg,
                              abcde=make_abcde(2), gradcam=make_gradcam())
        assert report.risk_level == "Low"

    def test_worse_than_chance_attention_escalates(self, cfg):
        """focus ratio 0.25 -- the model looked at the lesion less than a
        uniform heatmap would have."""
        report = build_report(make_prediction("Low", 0.95), cfg,
                              abcde=make_abcde(0),
                              gradcam=make_gradcam(inside=0.05, area=0.20))
        assert report.risk_level == "Moderate"
        assert any("chance" in reason for reason in report.escalation_reasons)

    def test_mediocre_attention_warns_without_escalating(self, cfg):
        """focus ratio 1.1 -- unimpressive but still better than chance. The
        user is told, but their advice is not changed: escalating on merely
        mediocre heatmaps would fire on a third of all reports."""
        report = build_report(make_prediction("Low", 0.95), cfg,
                              abcde=make_abcde(0),
                              gradcam=make_gradcam(inside=0.22, area=0.20))
        assert report.risk_level == "Low"
        assert report.escalation_reasons == []
        assert any("attention" in note for note in report.notes)

    def test_escalation_never_exceeds_the_highest_level(self, cfg):
        """Several escalations stacked must saturate at High, not overflow."""
        report = build_report(
            make_prediction("High", 0.30, low_confidence=True), cfg,
            abcde=make_abcde(4), change=make_change(True),
            gradcam=make_gradcam(inside=0.01, area=0.30),
        )
        assert report.risk_level == "High"

    def test_evidence_never_de_escalates(self, cfg):
        """The core safety property: nothing may talk the advice down."""
        for label in RISK_ORDER:
            report = build_report(make_prediction(label, 0.99), cfg,
                                  abcde=make_abcde(0), change=make_change(False),
                                  gradcam=make_gradcam())
            assert RISK_ORDER.index(report.risk_level) >= RISK_ORDER.index(label)

    def test_recommendation_matches_the_escalated_level(self, cfg):
        recommendations = cfg.get("report.recommendations")
        report = build_report(make_prediction("Low", 0.30, low_confidence=True), cfg,
                              abcde=make_abcde(0), gradcam=make_gradcam())
        assert report.recommendation == recommendations[report.risk_level]
        assert report.recommendation != recommendations["Low"]


class TestNotes:
    def test_missing_prior_photo_is_noted(self, cfg):
        report = build_report(make_prediction(), cfg, abcde=make_abcde(0))
        assert any("earlier photo" in note for note in report.notes)

    def test_prior_photo_removes_that_note(self, cfg):
        report = build_report(make_prediction(), cfg, abcde=make_abcde(0),
                              change=make_change(False))
        assert not any("did not upload an earlier photo" in note
                       for note in report.notes)

    def test_fallback_backbone_is_disclosed(self, cfg):
        report = build_report(make_prediction(), cfg, backbone_is_fallback=True)
        assert any("PanDerm" in note for note in report.notes)

    def test_failed_quality_is_disclosed(self, cfg):
        quality = QualityReport(passed=False, checks=[
            QualityCheck("blur", False, 5.0, 60.0, "It is blurry.")])
        report = build_report(make_prediction(), cfg, quality=quality)
        assert any("quality checks" in note for note in report.notes)

    def test_unconfident_segmentation_is_disclosed(self, cfg):
        report = build_report(make_prediction(), cfg,
                              abcde=make_abcde(0, confident=False))
        assert any("outline" in note for note in report.notes)


class TestRendering:
    @pytest.fixture
    def full_report(self, cfg):
        return build_report(
            make_prediction("Moderate", 0.72), cfg,
            quality=QualityReport(passed=True, checks=[], width=256, height=256),
            abcde=make_abcde(2), gradcam=make_gradcam(), change=make_change(False),
            backbone_description="test-backbone (256-dim)",
            metadata={"age": 55, "sex": "male", "site": "trunk"},
        )

    def test_markdown_contains_the_essentials(self, full_report):
        markdown = full_report.to_markdown()
        assert "# Skin lesion screening report" in markdown
        assert "Risk level: Moderate" in markdown
        assert "not a diagnosis" in markdown.lower()
        assert full_report.recommendation in markdown
        assert "ABCDE" in markdown

    def test_markdown_marks_unmeasurable_criteria(self, cfg):
        abcde = make_abcde(0)
        abcde.features[4].measurable = False
        markdown = build_report(make_prediction(), cfg, abcde=abcde).to_markdown()
        assert "[?]" in markdown, "Unmeasured criteria must be visibly distinct"

    def test_markdown_can_embed_the_heatmap(self, full_report):
        assert "data:image/png;base64," in full_report.to_markdown(include_images=True)
        assert "data:image/png;base64," not in full_report.to_markdown(include_images=False)

    def test_html_is_well_formed_enough_to_parse(self, full_report):
        from html.parser import HTMLParser

        class Parser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.tags = 0

            def handle_starttag(self, tag, attrs):
                self.tags += 1

        parser = Parser()
        parser.feed(full_report.to_html())
        assert parser.tags > 10

    def test_html_escapes_injected_content(self, cfg):
        report = build_report(make_prediction(), cfg,
                              backbone_description="<script>alert(1)</script>")
        html_output = report.to_html()
        assert "<script>alert(1)</script>" not in html_output
        assert "&lt;script&gt;" in html_output

    def test_json_round_trips(self, full_report):
        payload = json.loads(full_report.to_json())
        assert payload["risk_level"] == "Moderate"
        assert payload["disclaimer"]
        assert set(payload["probabilities"]) == set(RISK_ORDER)

    def test_disclaimer_appears_in_every_format(self, full_report):
        assert "NOT a medical device" in full_report.to_markdown()
        assert "NOT a medical device" in full_report.to_html()
        assert "NOT a medical device" in full_report.to_json()

    def test_save_writes_all_three_formats(self, full_report, tmp_outputs):
        written = full_report.save(tmp_outputs, stem="report")
        assert set(written) == {"md", "html", "json"}
        for path in written.values():
            assert path.exists() and path.stat().st_size > 0

    def test_data_uri_encoding(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        uri = encode_image_data_uri(image)
        assert uri.startswith("data:image/png;base64,")
        assert len(uri) > 40
