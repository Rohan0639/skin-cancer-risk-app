"""Tests for segmentation, the ABCDE heuristics and Grad-CAM.

The ABCDE tests are written against images with *known* geometry -- a perfect
circle, a deliberately ragged blob -- so the assertions are about ground truth
rather than about whatever the demo dataset happened to produce.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.explain.abcde import (
    ABCDEReport,
    asymmetry_score,
    border_irregularity,
    colour_variation,
    compute_abcde,
    estimate_diameter_mm,
)
from src.explain.gradcam import (
    GradCAMResult,
    attention_sanity_note,
    overlay_heatmap,
)
from src.segmentation import largest_contour, segment_lesion


# ---------------------------------------------------------------------------
# Synthetic masks with exactly known properties
# ---------------------------------------------------------------------------
def circle_mask(size: int = 200, radius: int = 60) -> np.ndarray:
    """A perfect circle: maximally symmetric, minimally irregular."""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), radius, 255, -1)
    return mask


def ellipse_mask(size: int = 200) -> np.ndarray:
    """An ellipse: still symmetric about both principal axes."""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, (size // 2, size // 2), (70, 35), 0, 0, 360, 255, -1)
    return mask


def half_moon_mask(size: int = 200) -> np.ndarray:
    """A circle with a bite taken out of one side: clearly asymmetric."""
    mask = circle_mask(size)
    cv2.circle(mask, (size // 2 + 45, size // 2), 45, 0, -1)
    return mask


def star_mask(size: int = 200, points: int = 9) -> np.ndarray:
    """A spiky star: a long perimeter for its area, so highly irregular."""
    mask = np.zeros((size, size), dtype=np.uint8)
    centre = size // 2
    vertices = []
    for i in range(points * 2):
        radius = 80 if i % 2 == 0 else 30
        angle = np.pi * i / points
        vertices.append([int(centre + radius * np.cos(angle)),
                         int(centre + radius * np.sin(angle))])
    cv2.fillPoly(mask, [np.array(vertices, dtype=np.int32)], 255)
    return mask


class TestAsymmetry:
    def test_circle_is_symmetric(self):
        assert asymmetry_score(circle_mask()) < 0.05

    def test_ellipse_is_symmetric_after_axis_alignment(self):
        """PCA alignment means orientation must not affect the score."""
        assert asymmetry_score(ellipse_mask()) < 0.08

    def test_rotation_does_not_change_the_score(self):
        mask = half_moon_mask()
        rotated = cv2.warpAffine(
            mask, cv2.getRotationMatrix2D((100, 100), 37, 1.0), (200, 200))
        assert asymmetry_score(mask) == pytest.approx(
            asymmetry_score(rotated), abs=0.06)

    def test_asymmetric_shape_scores_higher(self):
        assert asymmetry_score(half_moon_mask()) > asymmetry_score(circle_mask())

    def test_empty_mask_is_zero(self):
        assert asymmetry_score(np.zeros((50, 50), dtype=np.uint8)) == 0.0

    def test_score_stays_in_range(self):
        for mask in (circle_mask(), half_moon_mask(), star_mask()):
            assert 0.0 <= asymmetry_score(mask) <= 1.0


class TestBorderIrregularity:
    def test_circle_is_near_one(self):
        """perimeter^2/(4*pi*area) == 1 exactly for a circle."""
        assert border_irregularity(circle_mask()) == pytest.approx(1.0, abs=0.15)

    def test_star_is_far_above_one(self):
        assert border_irregularity(star_mask()) > 1.6

    def test_ordering_matches_intuition(self):
        assert border_irregularity(star_mask()) > border_irregularity(circle_mask())

    def test_never_below_one(self):
        for mask in (circle_mask(), ellipse_mask(), star_mask(),
                     np.zeros((50, 50), dtype=np.uint8)):
            assert border_irregularity(mask) >= 1.0

    def test_empty_mask_does_not_crash(self):
        assert border_irregularity(np.zeros((50, 50), dtype=np.uint8)) == 1.0

    def test_scale_invariance(self):
        small = circle_mask(size=100, radius=30)
        large = circle_mask(size=400, radius=120)
        assert border_irregularity(small) == pytest.approx(
            border_irregularity(large), abs=0.2)


class TestColourVariation:
    def test_uniform_colour_gives_one_shade(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[:, :] = (140, 100, 70)
        variation, shades = colour_variation(image, circle_mask())
        assert shades == 1
        assert variation < 0.05

    def test_two_tone_lesion_detects_more_shades(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[:100, :] = (200, 170, 140)
        image[100:, :] = (40, 25, 20)
        _, shades = colour_variation(image, circle_mask())
        assert shades >= 2

    def test_variation_rises_with_contrast(self):
        mask = circle_mask()
        flat = np.full((200, 200, 3), 120, dtype=np.uint8)
        varied = flat.copy()
        varied[:100, :] = (230, 210, 190)
        assert colour_variation(varied, mask)[0] > colour_variation(flat, mask)[0]

    def test_tiny_mask_is_handled(self):
        image = np.full((200, 200, 3), 120, dtype=np.uint8)
        tiny = np.zeros((200, 200), dtype=np.uint8)
        tiny[100:102, 100:102] = 255
        variation, shades = colour_variation(image, tiny)
        assert variation == 0.0 and shades == 1


class TestDiameter:
    def test_pixel_diameter_matches_the_circle(self):
        diameter_px, _ = estimate_diameter_mm(circle_mask(radius=60), 200, 40.0)
        assert diameter_px == pytest.approx(120, rel=0.1)

    def test_millimetres_scale_with_the_frame_width(self):
        mask = circle_mask(radius=50)
        _, small = estimate_diameter_mm(mask, 200, 20.0)
        _, large = estimate_diameter_mm(mask, 200, 40.0)
        assert large == pytest.approx(2 * small, rel=0.01)

    def test_empty_mask_gives_zero(self):
        assert estimate_diameter_mm(np.zeros((50, 50), dtype=np.uint8), 50, 40.0) == (0.0, 0.0)


class TestSegmentation:
    def test_finds_a_plain_lesion(self, round_lesion):
        result = segment_lesion(round_lesion)
        assert result.found
        assert 0.01 < result.area_fraction < 0.6

    def test_mask_matches_the_image_size(self, round_lesion):
        result = segment_lesion(round_lesion)
        assert result.mask.shape == round_lesion.shape[:2]

    def test_mask_has_no_interior_holes(self, ragged_lesion):
        """Holes would corrupt area, asymmetry and especially perimeter."""
        result = segment_lesion(ragged_lesion)
        filled = np.zeros_like(result.mask)
        contour = largest_contour(result.mask)
        cv2.drawContours(filled, [contour], -1, 255, thickness=-1)
        assert np.array_equal(filled > 0, result.mask > 0)

    def test_blank_image_falls_back_gracefully(self):
        blank = np.full((200, 200, 3), 180, dtype=np.uint8)
        result = segment_lesion(blank)
        assert result.found is False
        assert result.mask.sum() > 0, "The fallback ellipse must still be usable"

    def test_centroid_and_bbox_are_consistent(self, round_lesion):
        result = segment_lesion(round_lesion)
        y0, x0, y1, x1 = result.bbox
        cy, cx = result.centroid
        assert y0 <= cy <= y1 and x0 <= cx <= x1

    def test_rejects_non_rgb_input(self):
        with pytest.raises(ValueError, match="HxWx3"):
            segment_lesion(np.zeros((50, 50), dtype=np.uint8))

    def test_refinement_can_be_disabled(self, round_lesion):
        assert segment_lesion(round_lesion, refine=False).found


class TestABCDEReport:
    def test_round_plain_lesion_flags_nothing(self, round_lesion, cfg):
        report = compute_abcde(round_lesion, cfg)
        assert report.segmentation_confident
        assert report.flag_count == 0, (
            "A smooth, symmetric, single-colour lesion should raise no flags; "
            f"got {[f.letter for f in report.flagged]}"
        )

    def test_ragged_lesion_flags_several(self, ragged_lesion, cfg):
        report = compute_abcde(ragged_lesion, cfg)
        flagged = {f.letter for f in report.flagged}
        assert len(flagged) >= 2, f"Expected multiple flags, got {flagged}"

    def test_all_five_criteria_are_always_present(self, round_lesion, cfg):
        report = compute_abcde(round_lesion, cfg)
        assert [f.letter for f in report.features] == ["A", "B", "C", "D", "E"]

    def test_evolving_is_unmeasurable_without_a_prior_photo(self, round_lesion, cfg):
        report = compute_abcde(round_lesion, cfg)
        evolving = report.get("evolving")
        assert evolving.measurable is False
        assert evolving.flagged is False
        assert "earlier photo" in evolving.sentence

    def test_diameter_is_unmeasurable_without_a_scale(self, round_lesion, cfg):
        """A photo carries no scale, so D must not silently flag everything."""
        feature = compute_abcde(round_lesion, cfg).get("diameter")
        assert feature.measurable is False
        assert feature.flagged is False
        assert "ruler" in feature.sentence

    def test_diameter_becomes_measurable_with_a_scale(self, round_lesion, cfg):
        feature = compute_abcde(round_lesion, cfg, frame_width_mm=40.0).get("diameter")
        assert feature.measurable is True
        assert feature.score > 0

    def test_evolving_uses_the_change_summary(self, round_lesion, cfg):
        summary = {"changed": True, "score": 0.8, "threshold": 0.35,
                   "reason": "Its colour has changed."}
        feature = compute_abcde(round_lesion, cfg, change_summary=summary).get("evolving")
        assert feature.measurable and feature.flagged
        assert "changed noticeably" in feature.sentence

    def test_unconfident_segmentation_is_reported(self, cfg):
        blank = np.full((300, 300, 3), 175, dtype=np.uint8)
        report = compute_abcde(blank, cfg)
        assert report.segmentation_confident is False
        assert "unreliable" in report.summary()

    def test_every_feature_has_a_plain_language_sentence(self, ragged_lesion, cfg):
        for feature in compute_abcde(ragged_lesion, cfg).features:
            assert len(feature.sentence) > 30
            assert feature.sentence[0].isupper()

    def test_report_is_json_serialisable(self, round_lesion, cfg):
        import json

        payload = json.loads(json.dumps(compute_abcde(round_lesion, cfg).to_dict()))
        assert len(payload["features"]) == 5
        assert "summary" in payload

    def test_rejects_non_rgb_input(self, cfg):
        with pytest.raises(ValueError, match="HxWx3"):
            compute_abcde(np.zeros((50, 50), dtype=np.uint8), cfg)

    def test_empty_report_summary_does_not_crash(self):
        assert isinstance(ABCDEReport().summary(), str)


class TestGradCAMHelpers:
    def test_overlay_preserves_shape_and_dtype(self, round_lesion):
        heatmap = np.random.default_rng(0).random(round_lesion.shape[:2])
        overlay = overlay_heatmap(round_lesion, heatmap)
        assert overlay.shape == round_lesion.shape
        assert overlay.dtype == np.uint8

    def test_overlay_resizes_a_mismatched_heatmap(self, round_lesion):
        small = np.random.default_rng(0).random((7, 7))
        assert overlay_heatmap(round_lesion, small).shape == round_lesion.shape

    def test_zero_alpha_returns_the_original(self, round_lesion):
        heatmap = np.ones(round_lesion.shape[:2])
        np.testing.assert_array_equal(
            overlay_heatmap(round_lesion, heatmap, alpha=0.0), round_lesion)

    def test_focus_ratio_is_relative_to_lesion_size(self):
        """A small lesion cannot capture a large share of a diffuse heatmap,
        so the ratio -- not the raw fraction -- is what must be judged."""
        result = GradCAMResult(
            heatmap=np.zeros((4, 4)), overlay=np.zeros((4, 4, 3), np.uint8),
            target_class=0, target_class_name="Low",
            peak_inside_lesion=0.30, lesion_area_fraction=0.10,
        )
        assert result.focus_ratio == pytest.approx(3.0)
        assert attention_sanity_note(result) is None

    def test_note_fires_when_attention_is_no_better_than_chance(self):
        result = GradCAMResult(
            heatmap=np.zeros((4, 4)), overlay=np.zeros((4, 4, 3), np.uint8),
            target_class=0, target_class_name="Low",
            peak_inside_lesion=0.10, lesion_area_fraction=0.20,
        )
        assert result.focus_ratio == pytest.approx(0.5)
        note = attention_sanity_note(result)
        assert note is not None and "background" in note

    def test_no_note_without_a_mask(self):
        result = GradCAMResult(
            heatmap=np.zeros((4, 4)), overlay=np.zeros((4, 4, 3), np.uint8),
            target_class=0, target_class_name="Low",
        )
        assert result.focus_ratio is None
        assert attention_sanity_note(result) is None
