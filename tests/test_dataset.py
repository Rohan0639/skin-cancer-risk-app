"""Tests for config loading, metadata encoding and dataset splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config, load_config
from src.dataset import (
    UNIFIED_COLUMNS,
    build_metadata_matrix,
    class_distribution,
    encode_metadata,
    group_aware_split,
    metadata_dim,
    normalise_age,
    normalise_sex,
    normalise_site,
)


class TestConfig:
    def test_dotted_access(self, cfg):
        assert cfg.get("labels.risk_classes") == ["Low", "Moderate", "High"]
        assert isinstance(cfg.get("train.epochs"), int)

    def test_missing_key_raises_without_default(self, cfg):
        with pytest.raises(KeyError):
            cfg.get("nope.not.here")

    def test_missing_key_returns_default(self, cfg):
        assert cfg.get("nope.not.here", "fallback") == "fallback"

    def test_overrides_are_applied(self):
        overridden = load_config(overrides={"train.epochs": 999})
        assert overridden.get("train.epochs") == 999

    def test_relative_paths_resolve_against_repo_root(self, cfg):
        resolved = cfg.path("paths.unified_csv")
        assert resolved.is_absolute()
        assert resolved.name == "unified.csv"

    def test_absolute_paths_are_left_alone(self):
        absolute = Config.resolve("C:/somewhere/else.csv"
                                  if __import__("os").name == "nt" else "/tmp/else.csv")
        assert absolute.is_absolute()

    def test_missing_config_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("configs/does_not_exist.yaml")


class TestNormalisation:
    @pytest.mark.parametrize("raw,expected", [
        ("male", "male"), ("M", "male"), ("m", "male"),
        ("female", "female"), ("F", "female"),
        ("", "unknown"), (None, "unknown"), (np.nan, "unknown"),
        ("other", "unknown"),
    ])
    def test_sex_normalisation(self, raw, expected, cfg):
        assert normalise_sex(raw, cfg.get("metadata.sex_vocab")) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("back", "trunk"), ("BACK", "trunk"),        # HAM lower, PAD upper
        ("upper extremity", "upper_extremity"),
        ("FOREARM", "upper_extremity"),
        ("face", "head_neck"), ("NOSE", "head_neck"),
        ("acral", "palms_soles"),
        ("somewhere odd", "unknown"), (None, "unknown"),
    ])
    def test_site_normalisation_handles_both_datasets(self, raw, expected, cfg):
        site_map = dict(cfg.get("metadata.ham10000_site_map"))
        site_map.update(dict(cfg.get("metadata.pad_ufes_site_map")))
        vocab = cfg.get("metadata.site_vocab")
        assert normalise_site(raw, site_map, vocab) == expected

    @pytest.mark.parametrize("raw", [None, np.nan, "", "not a number", -5])
    def test_age_falls_back_to_impute_value(self, raw, cfg):
        assert normalise_age(raw, cfg) == float(cfg.get("metadata.age.impute"))

    def test_valid_age_passes_through(self, cfg):
        assert normalise_age(42, cfg) == 42.0
        assert normalise_age("37.5", cfg) == 37.5


class TestMetadataEncoding:
    def test_vector_has_the_declared_length(self, cfg):
        vector = encode_metadata(50, "male", "trunk", cfg)
        assert vector.shape == (metadata_dim(cfg),)
        assert vector.dtype == np.float32

    def test_one_hots_are_valid(self, cfg):
        n_sex = len(cfg.get("metadata.sex_vocab"))
        n_site = len(cfg.get("metadata.site_vocab"))
        vector = encode_metadata(50, "female", "head_neck", cfg)
        sex_block = vector[2:2 + n_sex]
        site_block = vector[2 + n_sex:2 + n_sex + n_site]
        assert sex_block.sum() == 1.0
        assert site_block.sum() == 1.0

    def test_age_is_scaled_into_unit_range(self, cfg):
        assert encode_metadata(0, "male", "trunk", cfg)[0] == pytest.approx(0.0)
        assert encode_metadata(100, "male", "trunk", cfg)[0] == pytest.approx(1.0)
        assert encode_metadata(50, "male", "trunk", cfg)[0] == pytest.approx(0.5)

    def test_out_of_range_age_is_clipped(self, cfg):
        assert 0.0 <= encode_metadata(150, "male", "trunk", cfg)[0] <= 1.0

    def test_missing_age_sets_the_missing_flag(self, cfg):
        present = encode_metadata(40, "male", "trunk", cfg)
        missing = encode_metadata(None, "male", "trunk", cfg)
        assert present[1] == 0.0
        assert missing[1] == 1.0, "A missing age must be distinguishable from a real one"

    def test_unknown_categories_do_not_raise(self, cfg):
        vector = encode_metadata(None, "wibble", "left elbow", cfg)
        assert vector.shape == (metadata_dim(cfg),)
        assert np.isfinite(vector).all()

    def test_matrix_matches_row_by_row_encoding(self, cfg):
        frame = pd.DataFrame({
            "age": [30, None, 70],
            "sex": ["male", "F", None],
            "site": ["trunk", "FACE", "nowhere"],
        })
        matrix = build_metadata_matrix(frame, cfg)
        assert matrix.shape == (3, metadata_dim(cfg))
        for i, row in enumerate(frame.to_dict("records")):
            expected = encode_metadata(row["age"], row["sex"], row["site"], cfg)
            np.testing.assert_allclose(matrix[i], expected)

    def test_empty_frame_gives_empty_matrix(self, cfg):
        matrix = build_metadata_matrix(pd.DataFrame(columns=["age", "sex", "site"]), cfg)
        assert matrix.shape == (0, metadata_dim(cfg))


def _fake_dataset(n_lesions: int = 30, shots: int = 2) -> pd.DataFrame:
    """Build a unified-schema frame with several images per lesion."""
    rows = []
    classes = ["Low", "Moderate", "High"]
    diagnoses = ["NEV", "ACK", "MEL"]          # one per tier, in the same order
    for i in range(n_lesions):
        risk = classes[i % len(classes)]
        dx = diagnoses[i % len(diagnoses)]
        for shot in range(shots):
            rows.append({
                "image_path": f"img_{i}_{shot}.png",
                "age": 40 + i, "sex": "male", "site": "trunk",
                "diagnosis": dx, "risk_label": risk,
                "source": "test", "lesion_id": f"lesion_{i}",
            })
    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS)


class TestSplitting:
    def test_splits_are_disjoint(self, cfg):
        train, val, test = group_aware_split(_fake_dataset(), cfg)
        paths = [set(part["image_path"]) for part in (train, val, test)]
        assert not paths[0] & paths[1]
        assert not paths[0] & paths[2]
        assert not paths[1] & paths[2]

    def test_no_lesion_spans_two_splits(self, cfg):
        """The key anti-leakage property: HAM10000 repeats lesions."""
        train, val, test = group_aware_split(_fake_dataset(), cfg)
        groups = [set(part["lesion_id"]) for part in (train, val, test)]
        assert not groups[0] & groups[1]
        assert not groups[0] & groups[2]
        assert not groups[1] & groups[2]

    def test_every_image_lands_somewhere(self, cfg):
        frame = _fake_dataset()
        train, val, test = group_aware_split(frame, cfg)
        assert len(train) + len(val) + len(test) == len(frame)

    def test_every_class_survives_in_training(self, cfg):
        train, _, _ = group_aware_split(_fake_dataset(), cfg)
        counts = class_distribution(train, ["NEV", "ACK", "MEL"])
        assert all(count > 0 for count in counts.values())

    def test_split_is_reproducible(self, cfg):
        frame = _fake_dataset()
        first = group_aware_split(frame, cfg)[2]["image_path"].tolist()
        second = group_aware_split(frame, cfg)[2]["image_path"].tolist()
        assert first == second

    def test_tiny_dataset_does_not_crash(self, cfg):
        train, val, test = group_aware_split(_fake_dataset(n_lesions=3, shots=1), cfg)
        assert len(train) >= 1

    def test_class_distribution_includes_absent_classes(self):
        frame = _fake_dataset()
        frame = frame[frame["risk_label"] != "High"]
        counts = class_distribution(frame, ["Low", "Moderate", "High"])
        assert counts["High"] == 0
        assert set(counts) == {"Low", "Moderate", "High"}
