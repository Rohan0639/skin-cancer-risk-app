"""Tests for the backbone interface, fusion head and risk classifier.

These use the ``fast_cfg`` fixture (small, un-pretrained ResNet-18) because the
properties under test -- tensor shapes, the metadata contract, checkpoint
round-tripping -- do not depend on the weights being meaningful.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.backbone import BackboneUnavailable, build_backbone
from src.dataset import metadata_dim
from src.model.fusion import FusionHead, MetadataEncoder
from src.model.risk_classifier import (
    GradCAMTarget,
    RiskClassifier,
    RiskPrediction,
    load_risk_classifier,
)
from src.model.train import compute_class_weights, macro_f1


class TestBackboneFactory:
    def test_fallback_backend_builds(self, fast_cfg):
        backbone = build_backbone(fast_cfg)
        assert backbone.is_fallback is True
        assert backbone.feature_dim > 0
        assert "resnet18" in backbone.name

    def test_unknown_backend_raises(self, fast_cfg):
        bad = type(fast_cfg)(fast_cfg.as_dict())
        bad._data["backbone"]["backend"] = "wibble"
        with pytest.raises(ValueError, match="Unknown backbone.backend"):
            build_backbone(bad)

    def test_strict_panderm_raises_when_unavailable(self, fast_cfg):
        """With backend='panderm' the caller asked for PanDerm specifically, so
        a silent fallback would be wrong -- it must raise."""
        strict = type(fast_cfg)(fast_cfg.as_dict())
        strict._data["backbone"]["backend"] = "panderm"
        strict._data["backbone"]["panderm"]["repo_dir"] = "third_party/definitely_absent"
        with pytest.raises(BackboneUnavailable, match="PanDerm"):
            build_backbone(strict)

    def test_auto_backend_falls_back_silently(self, fast_cfg):
        auto = type(fast_cfg)(fast_cfg.as_dict())
        auto._data["backbone"]["backend"] = "auto"
        auto._data["backbone"]["panderm"]["repo_dir"] = "third_party/definitely_absent"
        backbone = build_backbone(auto)
        assert backbone.is_fallback is True

    def test_embeddings_have_a_fixed_size(self, fast_cfg):
        backbone = build_backbone(fast_cfg)
        transform = backbone.build_transform(train=False)
        from PIL import Image

        rng = np.random.default_rng(0)
        images = [Image.fromarray(rng.integers(0, 255, (h, h, 3), dtype=np.uint8))
                  for h in (128, 256, 400)]
        batch = torch.stack([transform(img) for img in images])
        with torch.no_grad():
            features = backbone(batch)
        assert features.shape == (3, backbone.feature_dim), \
            "Different input sizes must still give one fixed-size vector each"
        assert torch.isfinite(features).all()

    def test_embed_image_helper(self, fast_cfg, round_lesion):
        backbone = build_backbone(fast_cfg)
        vector = backbone.embed_image(round_lesion)
        assert vector.shape == (backbone.feature_dim,)

    def test_train_transform_augments(self, fast_cfg):
        backbone = build_backbone(fast_cfg)
        assert len(backbone.build_transform(train=True).transforms) > \
               len(backbone.build_transform(train=False).transforms)

    def test_gradcam_target_layers_exist(self, fast_cfg):
        backbone = build_backbone(fast_cfg)
        layers = backbone.gradcam_target_layers()
        assert len(layers) >= 1
        assert all(isinstance(layer, torch.nn.Module) for layer in layers)

    def test_describe_mentions_fallback(self, fast_cfg):
        assert "fallback" in build_backbone(fast_cfg).describe().lower()


class TestMetadataEncoder:
    def test_output_shape(self):
        encoder = MetadataEncoder(input_dim=12, output_dim=32)
        assert encoder(torch.randn(5, 12)).shape == (5, 32)

    def test_wrong_input_width_raises(self):
        encoder = MetadataEncoder(input_dim=12, output_dim=32)
        with pytest.raises(ValueError, match="Expected metadata"):
            encoder(torch.randn(5, 7))

    def test_non_positive_dim_raises(self):
        with pytest.raises(ValueError):
            MetadataEncoder(input_dim=0)


class TestFusionHead:
    def test_output_shape(self):
        head = FusionHead(feature_dim=512, metadata_dim=12, num_classes=3)
        logits = head(torch.randn(4, 512), torch.randn(4, 12))
        assert logits.shape == (4, 3)

    def test_single_sample_works_in_train_mode(self):
        """BatchNorm needs >1 sample; the head must handle a batch of one."""
        head = FusionHead(feature_dim=512, metadata_dim=12, num_classes=3).train()
        logits = head(torch.randn(1, 512), torch.randn(1, 12))
        assert logits.shape == (1, 3)

    def test_metadata_can_be_disabled(self):
        head = FusionHead(feature_dim=512, metadata_dim=12, num_classes=3,
                          use_metadata=False)
        assert head(torch.randn(4, 512)).shape == (4, 3)
        assert head.metadata_encoder is None

    def test_missing_metadata_raises_when_required(self):
        head = FusionHead(feature_dim=512, metadata_dim=12, num_classes=3)
        with pytest.raises(ValueError, match="metadata must be provided"):
            head(torch.randn(4, 512), None)

    def test_wrong_feature_width_raises(self):
        head = FusionHead(feature_dim=512, metadata_dim=12, num_classes=3)
        with pytest.raises(ValueError, match="Expected features"):
            head(torch.randn(4, 256), torch.randn(4, 12))

    def test_metadata_changes_the_output(self):
        """If metadata never moved the logits, fusing it would be pointless."""
        torch.manual_seed(0)
        head = FusionHead(feature_dim=64, metadata_dim=12, num_classes=3).eval()
        features = torch.randn(2, 64)
        a = head(features, torch.zeros(2, 12))
        b = head(features, torch.ones(2, 12))
        assert not torch.allclose(a, b)

    @pytest.mark.parametrize("hidden", [[], [64], [128, 64, 32]])
    def test_arbitrary_depth(self, hidden):
        head = FusionHead(feature_dim=64, metadata_dim=12, num_classes=3,
                          hidden_dims=hidden)
        assert head(torch.randn(4, 64), torch.randn(4, 12)).shape == (4, 3)

    def test_too_few_classes_raises(self):
        with pytest.raises(ValueError, match="num_classes"):
            FusionHead(feature_dim=64, metadata_dim=12, num_classes=1)


class TestRiskClassifier:
    @pytest.fixture(scope="class")
    def model(self, fast_cfg):
        return RiskClassifier(build_backbone(fast_cfg), fast_cfg)

    def test_forward_shape(self, model, fast_cfg):
        images = torch.randn(2, 3, 224, 224)
        meta = torch.randn(2, metadata_dim(fast_cfg))
        assert model(images, meta).shape == (2, 3)

    def test_backbone_is_frozen_by_default(self, model):
        assert all(not p.requires_grad for p in model.backbone.parameters())
        assert any(p.requires_grad for p in model.head.parameters())

    def test_frozen_backbone_stays_in_eval_mode(self, model):
        model.train()
        assert model.backbone.training is False, \
            "A frozen backbone must keep its BatchNorm running stats fixed"
        assert model.head.training is True

    def test_trainable_parameters_excludes_backbone(self, model):
        trainable = list(model.trainable_parameters())
        head_params = list(model.head.parameters())
        assert len(trainable) == len(head_params)

    def test_predict_returns_a_valid_distribution(self, model, round_lesion):
        prediction = model.predict(round_lesion, age=50, sex="male", site="trunk")
        assert isinstance(prediction, RiskPrediction)
        assert prediction.label in model.classes
        assert 0.0 <= prediction.confidence <= 1.0
        assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
        assert prediction.confidence == pytest.approx(
            max(prediction.probabilities.values()))

    def test_predict_keeps_features_for_change_detection(self, model, round_lesion):
        prediction = model.predict(round_lesion, 50, "male", "trunk")
        assert prediction.features is not None
        assert prediction.features.shape == (model.backbone.feature_dim,)

    def test_predict_tolerates_missing_metadata(self, model, round_lesion):
        prediction = model.predict(round_lesion, None, None, None)
        assert prediction.label in model.classes

    def test_prediction_to_dict_excludes_features(self, model, round_lesion):
        payload = model.predict(round_lesion, 50, "male", "trunk").to_dict()
        assert "features" not in payload
        assert set(payload) == {"label", "confidence", "probabilities", "low_confidence"}

    def test_checkpoint_round_trip(self, fast_cfg, tmp_path, round_lesion):
        """A saved head must reproduce its predictions when reloaded.

        The checkpoint stores only the head and rebuilds the backbone from
        config, so this test seeds the RNG around each construction. With the
        real (pretrained) backbone that seeding is unnecessary -- the weights
        are identical every time -- but ``fast_cfg`` uses a randomly
        initialised ResNet-18 for speed.
        """
        torch.manual_seed(1234)
        model = RiskClassifier(build_backbone(fast_cfg), fast_cfg)
        path = tmp_path / "head.pt"
        model.save(path, extra={"note": "test"})
        before = model.predict(round_lesion, 50, "male", "trunk")

        torch.manual_seed(1234)
        restored = load_risk_classifier(path, cfg=fast_cfg, strict=True)
        after = restored.predict(round_lesion, 50, "male", "trunk")

        assert after.label == before.label
        assert after.confidence == pytest.approx(before.confidence, abs=1e-4)
        for name in before.probabilities:
            assert after.probabilities[name] == pytest.approx(
                before.probabilities[name], abs=1e-4)

    def test_non_reproducible_backbone_is_flagged(self, fast_cfg, tmp_path, caplog):
        """A head trained on a randomly initialised backbone cannot be restored
        faithfully, so both saving and loading must say so loudly."""
        import logging

        model = RiskClassifier(build_backbone(fast_cfg), fast_cfg)
        assert model.backbone.is_reproducible is False

        path = tmp_path / "head.pt"
        with caplog.at_level(logging.WARNING):
            model.save(path)
        assert "NON-REPRODUCIBLE" in caplog.text

        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["backbone_is_reproducible"] is False

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            load_risk_classifier(path, cfg=fast_cfg, strict=True)
        assert "will NOT match" in caplog.text

    def test_missing_checkpoint_raises_when_strict(self, fast_cfg, tmp_path):
        with pytest.raises(FileNotFoundError, match="train.py"):
            load_risk_classifier(tmp_path / "absent.pt", cfg=fast_cfg, strict=True)

    def test_missing_checkpoint_returns_untrained_when_lenient(self, fast_cfg, tmp_path):
        model = load_risk_classifier(tmp_path / "absent.pt", cfg=fast_cfg, strict=False)
        assert isinstance(model, RiskClassifier)

    def test_dimension_mismatch_is_detected(self, model, fast_cfg, tmp_path):
        """Loading a checkpoint trained on a different backbone must not
        silently produce garbage."""
        path = tmp_path / "head.pt"
        model.save(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["feature_dim"] = 99999
        payload["backbone_name"] = "some_other_model"
        torch.save(payload, path)

        with pytest.raises(ValueError, match="dim backbone"):
            load_risk_classifier(path, cfg=fast_cfg, strict=True)


class TestGradCAMTarget:
    def test_accepts_images_only(self, fast_cfg):
        model = RiskClassifier(build_backbone(fast_cfg), fast_cfg).eval()
        meta = torch.zeros(1, metadata_dim(fast_cfg))
        wrapped = GradCAMTarget(model, meta).eval()
        assert wrapped(torch.randn(1, 3, 224, 224)).shape == (1, 3)

    def test_broadcasts_metadata_across_a_batch(self, fast_cfg):
        model = RiskClassifier(build_backbone(fast_cfg), fast_cfg).eval()
        wrapped = GradCAMTarget(model, torch.zeros(1, metadata_dim(fast_cfg))).eval()
        assert wrapped(torch.randn(4, 3, 224, 224)).shape == (4, 3)


class TestTrainingHelpers:
    def test_class_weights_favour_rare_classes(self):
        import pandas as pd

        frame = pd.DataFrame({"risk_label": ["Low"] * 90 + ["Moderate"] * 9 + ["High"]})
        weights = compute_class_weights(frame, ["Low", "Moderate", "High"])
        assert weights[2] > weights[1] > weights[0], \
            "Rarer classes must receive larger loss weights"

    def test_absent_class_gets_zero_weight(self):
        import pandas as pd

        frame = pd.DataFrame({"risk_label": ["Low"] * 10})
        weights = compute_class_weights(frame, ["Low", "Moderate", "High"])
        assert weights[1] == 0.0 and weights[2] == 0.0

    def test_macro_f1_perfect_and_worst(self):
        truth = np.array([0, 1, 2, 0, 1, 2])
        assert macro_f1(truth, truth.copy(), 3) == pytest.approx(1.0)
        assert macro_f1(truth, np.zeros_like(truth), 3) < 0.4

    def test_macro_f1_penalises_ignoring_a_class(self):
        """Macro-F1 is the early-stopping criterion precisely because a model
        that never predicts the rare class must score badly."""
        truth = np.array([0] * 90 + [2] * 10)
        always_low = np.zeros_like(truth)
        assert macro_f1(truth, always_low, 3) < 0.5
