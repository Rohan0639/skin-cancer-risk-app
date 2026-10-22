"""Tests for the image-quality gate.

The gate's job is to reject unusable photos, so each test degrades a known-good
image in one specific way and asserts that the matching check -- and ideally
only that check -- fails.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.quality.image_quality import (
    QualityCheck,
    QualityReport,
    assess_image_quality,
    brightness_stats,
    laplacian_variance,
    quality_summary_lines,
)


def failed_names(report: QualityReport) -> set[str]:
    """Return the set of check names that failed."""
    return {check.name for check in report.failures}


class TestMeasurements:
    def test_laplacian_variance_drops_when_blurred(self, round_lesion):
        sharp = laplacian_variance(round_lesion)
        blurred = laplacian_variance(cv2.GaussianBlur(round_lesion, (31, 31), 0))
        assert sharp > blurred, "Blurring must reduce the sharpness measure"

    def test_brightness_stats_on_flat_image(self):
        flat = np.full((64, 64, 3), 128, dtype=np.uint8)
        mean, std = brightness_stats(flat)
        assert mean == pytest.approx(128, abs=1)
        assert std == pytest.approx(0, abs=1e-3)

    def test_brightness_tracks_scaling(self, round_lesion):
        bright_mean, _ = brightness_stats(round_lesion)
        dark_mean, _ = brightness_stats((round_lesion * 0.3).astype(np.uint8))
        assert dark_mean < bright_mean


class TestQualityGate:
    def test_good_image_passes(self, round_lesion, cfg):
        report = assess_image_quality(round_lesion, cfg)
        assert report.passed, f"Expected a pass, got: {report.reason}"
        assert report.reason == "Photo quality looks good."
        assert report.failures == []

    def test_blurred_image_fails_blur_check(self, round_lesion, cfg):
        blurred = cv2.GaussianBlur(round_lesion, (31, 31), 0)
        report = assess_image_quality(blurred, cfg)
        assert not report.passed
        assert "blur" in failed_names(report)
        assert "focus" in report.reason.lower()

    def test_dark_image_fails_brightness_check(self, round_lesion, cfg):
        dark = (round_lesion.astype(np.float32) * 0.08).astype(np.uint8)
        report = assess_image_quality(dark, cfg)
        assert not report.passed
        assert "brightness" in failed_names(report)
        assert "too dark" in report.reason.lower()

    def test_overexposed_image_fails_brightness_check(self, round_lesion, cfg):
        bright = np.clip(round_lesion.astype(np.float32) * 1.9 + 80, 0, 255).astype(np.uint8)
        report = assess_image_quality(bright, cfg)
        assert not report.passed
        assert "brightness" in failed_names(report)
        assert "exposed" in report.reason.lower()

    def test_small_image_fails_resolution_check(self, round_lesion, cfg):
        tiny = cv2.resize(round_lesion, (64, 64))
        report = assess_image_quality(tiny, cfg)
        assert not report.passed
        assert "resolution" in failed_names(report)
        assert "64x64" in report.reason

    def test_flat_image_fails_contrast_check(self, cfg):
        flat = np.full((256, 256, 3), 128, dtype=np.uint8)
        report = assess_image_quality(flat, cfg)
        assert not report.passed
        assert "contrast" in failed_names(report)

    def test_framing_can_be_skipped(self, round_lesion, cfg):
        report = assess_image_quality(round_lesion, cfg, check_framing=False)
        assert "framing" not in {c.name for c in report.checks}

    def test_report_records_dimensions(self, round_lesion, cfg):
        report = assess_image_quality(round_lesion, cfg)
        assert (report.height, report.width) == round_lesion.shape[:2]

    def test_accepts_a_path(self, tmp_path, round_lesion, cfg):
        from PIL import Image

        path = tmp_path / "lesion.png"
        Image.fromarray(round_lesion).save(path)
        report = assess_image_quality(path, cfg)
        assert report.passed

    def test_missing_file_raises(self, tmp_path, cfg):
        with pytest.raises(FileNotFoundError):
            assess_image_quality(tmp_path / "nope.png", cfg)


class TestSerialisation:
    def test_report_to_dict_is_json_serialisable(self, round_lesion, cfg):
        import json

        report = assess_image_quality(round_lesion, cfg)
        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["passed"] is True
        assert len(payload["checks"]) == len(report.checks)
        assert all({"name", "passed", "value"} <= set(c) for c in payload["checks"])

    def test_summary_lines_cover_every_check(self, round_lesion, cfg):
        report = assess_image_quality(round_lesion, cfg)
        lines = quality_summary_lines(report)
        assert len(lines) == len(report.checks)
        assert all(line.startswith(("[pass]", "[FAIL]")) for line in lines)

    def test_reason_concatenates_multiple_failures(self):
        report = QualityReport(passed=False, checks=[
            QualityCheck("a", False, 1.0, 2.0, "First problem."),
            QualityCheck("b", False, 1.0, 2.0, "Second problem."),
        ])
        assert "First problem." in report.reason
        assert "Second problem." in report.reason
