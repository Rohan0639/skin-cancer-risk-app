"""Tests for change detection between two photos of the same lesion.

The hard requirement is discrimination: nuisance variation (lighting, camera
distance, angle) must NOT read as change, while a genuinely different lesion
must. Both directions are tested, because a detector that fires on everything
is as useless as one that never fires.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.tracking.change_detection import (
    ChangeResult,
    align_lesion_crop,
    cosine_distance,
    detect_change,
    grey_world_balance,
    mean_colour_shift,
    relative_area_change,
)
from src.segmentation import segment_lesion


def rephotograph(image: np.ndarray, angle: float = 12.0, scale: float = 1.1,
                 warmth: float = 1.15, exposure: float = 1.08) -> np.ndarray:
    """Simulate the same lesion shot again: different angle, distance, bulb."""
    size = image.shape[0]
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
    out = cv2.warpAffine(image, matrix, (size, size), borderMode=cv2.BORDER_REFLECT)
    out = out.astype(np.float32) * np.array([warmth, 1.0, 2.0 - warmth]) * exposure
    return np.clip(out, 0, 255).astype(np.uint8)


class TestHelpers:
    def test_cosine_distance_of_identical_vectors_is_zero(self):
        vector = np.array([1.0, 2.0, 3.0])
        assert cosine_distance(vector, vector) == pytest.approx(0.0, abs=1e-9)

    def test_cosine_distance_of_opposites_is_two(self):
        vector = np.array([1.0, 0.0])
        assert cosine_distance(vector, -vector) == pytest.approx(2.0)

    def test_cosine_distance_ignores_magnitude(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_distance(a, a * 7.5) == pytest.approx(0.0, abs=1e-9)

    def test_cosine_distance_of_zero_vector_is_safe(self):
        assert cosine_distance(np.zeros(4), np.ones(4)) == 0.0

    def test_grey_world_neutralises_a_colour_cast(self):
        rng = np.random.default_rng(0)
        image = rng.integers(60, 200, (64, 64, 3), dtype=np.uint8)
        warm = np.clip(image.astype(np.float32) * [1.35, 1.0, 0.7],
                       0, 255).astype(np.uint8)
        before = np.abs(image.reshape(-1, 3).mean(0) - warm.reshape(-1, 3).mean(0)).mean()
        after = np.abs(grey_world_balance(image).reshape(-1, 3).mean(0)
                       - grey_world_balance(warm).reshape(-1, 3).mean(0)).mean()
        assert after < before / 3, "White balancing must cancel most of the cast"

    def test_align_crop_normalises_size(self, round_lesion):
        mask = segment_lesion(round_lesion)
        crop, crop_mask = align_lesion_crop(round_lesion, mask, size=224)
        assert crop.shape == (224, 224, 3)
        assert crop_mask.shape == (224, 224)

    def test_align_crop_removes_camera_distance(self, round_lesion):
        """The whole point of cropping: a closer photo must align to the same
        lesion size."""
        zoomed = cv2.resize(round_lesion[40:216, 40:216], (256, 256))
        a, _ = align_lesion_crop(round_lesion, segment_lesion(round_lesion))
        b, _ = align_lesion_crop(zoomed, segment_lesion(zoomed))
        frac_a = (segment_lesion(a).area_fraction)
        frac_b = (segment_lesion(b).area_fraction)
        assert abs(frac_a - frac_b) < 0.25

    def test_relative_area_change_is_zero_for_identical_masks(self, round_lesion):
        mask = segment_lesion(round_lesion)
        assert relative_area_change(mask, mask) == pytest.approx(0.0)

    def test_colour_shift_is_zero_for_identical_images(self, round_lesion):
        mask = segment_lesion(round_lesion).mask
        assert mean_colour_shift(round_lesion, mask, round_lesion, mask) == \
            pytest.approx(0.0, abs=1e-6)


class TestDetectChange:
    def test_identical_photos_report_no_change(self, round_lesion, cfg):
        result = detect_change(round_lesion, round_lesion, cfg)
        assert result.changed is False
        assert result.score == pytest.approx(0.0)
        assert "Nothing we measure" in result.reason

    def test_nuisance_variation_does_not_count_as_change(self, round_lesion, cfg):
        """Different angle, distance and lighting is not the lesion evolving."""
        result = detect_change(round_lesion, rephotograph(round_lesion), cfg)
        assert result.changed is False, (
            f"Camera/lighting variation was misread as change "
            f"(score {result.score:.2f}: "
            f"{[(s.name, round(s.value, 3)) for s in result.signals if s.changed]})"
        )

    def test_a_genuinely_different_lesion_is_flagged(self, round_lesion,
                                                     ragged_lesion, cfg):
        result = detect_change(round_lesion, ragged_lesion, cfg)
        assert result.changed is True
        assert result.score > result.threshold

    def test_score_ordering_is_sensible(self, round_lesion, ragged_lesion, cfg):
        same = detect_change(round_lesion, round_lesion, cfg).score
        nuisance = detect_change(round_lesion, rephotograph(round_lesion), cfg).score
        different = detect_change(round_lesion, ragged_lesion, cfg).score
        assert same <= nuisance < different

    def test_all_signals_are_reported(self, round_lesion, cfg):
        result = detect_change(round_lesion, round_lesion, cfg)
        names = {s.name for s in result.signals}
        assert {"colour", "asymmetry", "border", "relative_size"} <= names

    def test_appearance_signal_appears_with_a_backbone(self, round_lesion,
                                                       ragged_lesion, fast_cfg):
        from src.backbone import build_backbone

        result = detect_change(round_lesion, ragged_lesion, fast_cfg,
                               backbone=build_backbone(fast_cfg))
        assert any(s.name == "appearance" for s in result.signals)

    def test_unreliable_when_segmentation_fails(self, round_lesion, cfg):
        blank = np.full(round_lesion.shape, 178, dtype=np.uint8)
        result = detect_change(blank, round_lesion, cfg)
        assert result.reliable is False
        assert result.changed is False, \
            "An unreliable comparison must not assert that a lesion changed"
        assert "unreliable" in result.reason

    def test_rejects_non_rgb_input(self, round_lesion, cfg):
        with pytest.raises(ValueError, match="HxWx3"):
            detect_change(np.zeros((50, 50), dtype=np.uint8), round_lesion, cfg)

    def test_result_is_json_serialisable(self, round_lesion, cfg):
        import json

        payload = json.loads(json.dumps(detect_change(round_lesion, round_lesion, cfg).to_dict()))
        assert set(payload) >= {"changed", "score", "reason", "signals", "reliable"}

    def test_output_feeds_the_abcde_evolving_criterion(self, round_lesion,
                                                       ragged_lesion, cfg):
        """The dict returned here is consumed directly by compute_abcde."""
        from src.explain.abcde import compute_abcde

        change = detect_change(round_lesion, ragged_lesion, cfg)
        feature = compute_abcde(ragged_lesion, cfg,
                                change_summary=change.to_dict()).get("evolving")
        assert feature.measurable is True
        assert feature.flagged == change.changed

    def test_weights_sum_into_a_bounded_score(self, round_lesion, ragged_lesion, cfg):
        for result in (detect_change(round_lesion, round_lesion, cfg),
                       detect_change(round_lesion, ragged_lesion, cfg)):
            assert 0.0 <= result.score <= 1.0

    def test_empty_result_renders(self):
        assert ChangeResult(changed=False, score=0.0, threshold=0.35).to_dict()
