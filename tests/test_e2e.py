"""End-to-end smoke tests.

These run the real pipeline -- real backbone, real segmentation, real Grad-CAM
-- on sample images, and assert on *mechanics*: that the whole chain executes,
that the report is complete and internally consistent, and that the quality
gate short-circuits correctly.

They deliberately assert nothing about predictive accuracy. Accuracy is
measured by ``src/evaluate.py`` on a held-out split, not asserted in a test.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.pipeline import AnalysisResult, LesionAnalyzer
from src.report.report_generator import RISK_ORDER


@pytest.mark.slow
class TestPipelineEndToEnd:
    def test_analyse_returns_a_complete_report(self, analyzer, round_lesion):
        result = analyzer.analyse(round_lesion, age=45, sex="female", site="trunk")

        assert isinstance(result, AnalysisResult)
        assert not result.stopped_at_quality_gate
        assert result.report is not None
        assert result.prediction is not None
        assert result.abcde is not None
        assert result.mask is not None

        report = result.report
        assert report.risk_level in RISK_ORDER
        assert 0.0 <= report.confidence <= 1.0
        assert report.recommendation
        assert sum(report.probabilities.values()) == pytest.approx(1.0, abs=1e-4)

    def test_gradcam_is_produced_and_aligned(self, analyzer, ragged_lesion):
        result = analyzer.analyse(ragged_lesion, age=70, sex="male", site="head_neck")
        assert result.gradcam is not None
        assert result.gradcam.heatmap.shape == ragged_lesion.shape[:2]
        assert result.gradcam.overlay.shape == ragged_lesion.shape
        assert 0.0 <= result.gradcam.heatmap.min()
        assert result.gradcam.heatmap.max() <= 1.0

    def test_gradcam_target_matches_the_prediction(self, analyzer, round_lesion):
        """The heatmap must explain the class that was actually reported."""
        result = analyzer.analyse(round_lesion, age=45, sex="female", site="trunk")
        assert result.gradcam.target_class_name == result.prediction.diagnosis

    def test_explain_can_be_disabled(self, analyzer, round_lesion):
        result = analyzer.analyse(round_lesion, age=45, sex="female", site="trunk",
                                  explain=False)
        assert result.gradcam is None
        assert result.report is not None

    def test_all_five_abcde_criteria_are_reported(self, analyzer, round_lesion):
        result = analyzer.analyse(round_lesion, age=45, sex="female", site="trunk")
        assert [f.letter for f in result.abcde.features] == ["A", "B", "C", "D", "E"]

    def test_prior_photo_enables_change_detection(self, analyzer, round_lesion):
        result = analyzer.analyse(round_lesion, age=45, sex="female", site="trunk",
                                  previous_image=round_lesion)
        assert result.change is not None
        assert result.change.changed is False, "The same photo cannot have changed"
        evolving = result.abcde.get("evolving")
        assert evolving.measurable is True

    def test_change_detection_flags_a_different_lesion(self, analyzer, round_lesion,
                                                       ragged_lesion):
        result = analyzer.analyse(ragged_lesion, age=45, sex="female", site="trunk",
                                  previous_image=round_lesion)
        assert result.change is not None
        assert result.change.changed is True

    def test_scale_makes_diameter_measurable(self, analyzer, round_lesion):
        without = analyzer.analyse(round_lesion, 45, "female", "trunk", explain=False)
        with_scale = analyzer.analyse(round_lesion, 45, "female", "trunk",
                                      frame_width_mm=40.0, explain=False)
        assert without.abcde.get("diameter").measurable is False
        assert with_scale.abcde.get("diameter").measurable is True

    def test_report_renders_in_every_format(self, analyzer, round_lesion, tmp_outputs):
        result = analyzer.analyse(round_lesion, age=45, sex="female", site="trunk")
        written = result.report.save(tmp_outputs, stem="e2e")

        markdown = written["md"].read_text(encoding="utf-8")
        assert "Risk level" in markdown
        assert "not a diagnosis" in markdown.lower()
        assert written["html"].read_text(encoding="utf-8").startswith("<div")

        import json

        payload = json.loads(written["json"].read_text(encoding="utf-8"))
        assert payload["risk_level"] == result.report.risk_level

    def test_metadata_is_echoed_back(self, analyzer, round_lesion):
        result = analyzer.analyse(round_lesion, age=61, sex="male", site="lower_extremity")
        assert result.report.metadata["age"] == 61
        assert result.report.metadata["sex"] == "male"
        assert result.report.metadata["site"] == "lower_extremity"

    def test_metadata_changes_the_prediction(self, analyzer, round_lesion):
        """If patient details never moved the output, fusing them would be
        decorative rather than functional."""
        young = analyzer.analyse(round_lesion, age=18, sex="female", site="trunk",
                                 explain=False)
        old = analyzer.analyse(round_lesion, age=85, sex="male", site="head_neck",
                               explain=False)
        assert young.prediction.probabilities != old.prediction.probabilities


@pytest.mark.slow
class TestQualityGateIntegration:
    def test_blurry_photo_stops_before_prediction(self, analyzer, round_lesion):
        blurred = cv2.GaussianBlur(round_lesion, (41, 41), 0)
        result = analyzer.analyse(blurred, age=45, sex="female", site="trunk")

        assert result.stopped_at_quality_gate
        assert result.report is None
        assert result.prediction is None, \
            "No risk score should be produced from a rejected photo"
        assert "focus" in result.quality.reason.lower()

    def test_override_analyses_anyway_with_a_caution(self, analyzer, round_lesion):
        blurred = cv2.GaussianBlur(round_lesion, (41, 41), 0)
        result = analyzer.analyse(blurred, age=45, sex="female", site="trunk",
                                  require_quality=False)

        assert not result.stopped_at_quality_gate
        assert result.report is not None
        assert any("quality checks" in note for note in result.report.notes)

    def test_dark_photo_is_rejected(self, analyzer, round_lesion):
        dark = (round_lesion.astype(np.float32) * 0.06).astype(np.uint8)
        result = analyzer.analyse(dark, age=45, sex="female", site="trunk")
        assert result.stopped_at_quality_gate


@pytest.mark.slow
class TestInputHandling:
    def test_accepts_a_file_path(self, analyzer, round_lesion, tmp_path):
        from PIL import Image

        path = tmp_path / "lesion.png"
        Image.fromarray(round_lesion).save(path)
        result = analyzer.analyse(str(path), age=45, sex="female", site="trunk",
                                  explain=False)
        assert result.report is not None

    def test_accepts_a_file_object(self, analyzer, round_lesion, tmp_path):
        from PIL import Image

        path = tmp_path / "lesion.png"
        Image.fromarray(round_lesion).save(path)
        with path.open("rb") as handle:
            result = analyzer.analyse(handle, age=45, sex="female", site="trunk",
                                      explain=False)
        assert result.report is not None

    def test_tolerates_missing_metadata(self, analyzer, round_lesion):
        result = analyzer.analyse(round_lesion, explain=False)
        assert result.report is not None
        assert result.report.risk_level in RISK_ORDER

    def test_handles_rgba_input(self, analyzer, round_lesion):
        rgba = np.dstack([round_lesion,
                          np.full(round_lesion.shape[:2], 255, dtype=np.uint8)])
        result = analyzer.analyse(rgba, age=45, sex="female", site="trunk",
                                  explain=False)
        assert result.report is not None
