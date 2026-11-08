"""PanDerm dermatology foundation model wrapper.

PanDerm (Yan et al., *Nature Medicine* 2025, https://github.com/SiyuanYan1/PanDerm)
is a vision transformer pretrained on millions of dermatology images. Its
features are far better suited to skin lesions than generic ImageNet features.

    LICENCE NOTE: PanDerm's weights and code are released under CC-BY-NC
    (non-commercial) terms and may be used for non-commercial academic research
    with attribution. This project is a student research prototype and complies
    with that. Do not deploy it commercially. See README > "Licences and
    attribution".

Because the weights are gated behind a manual download, this wrapper is written
to fail *cleanly* -- raising :class:`BackboneUnavailable` -- so the factory can
fall back automatically instead of crashing the app.

Setup (see README for the full walkthrough)::

    git clone https://github.com/SiyuanYan1/PanDerm third_party/PanDerm
    # download the checkpoint from the repo's release links, then:
    mkdir -p checkpoints/panderm
    mv panderm_bb_data6_checkpoint-499.pth checkpoints/panderm/
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, List

import torch
import torch.nn as nn

from src.backbone.base import BackboneUnavailable, FeatureBackbone, default_transform
from src.config import Config
from src.utils import get_logger

logger = get_logger(__name__)

#: Maps the config's ``model_name`` to (builder function, embedding dim).
#: Names and builders mirror ``classification/models/builder.py`` upstream.
PANDERM_VARIANTS = {
    "PanDerm_Base_LP": ("panderm_base_patch16_224", 768),
    "PanDerm_Large_LP": ("panderm_large_patch16_224", 1024),
}


class PanDermBackbone(FeatureBackbone):
    """Loads a PanDerm ViT and exposes it through :class:`FeatureBackbone`."""

    is_fallback = False

    def __init__(self, cfg: Config) -> None:
        """Load PanDerm's source and checkpoint.

        Raises:
            BackboneUnavailable: If the repo has not been cloned, the checkpoint
                is missing, the model name is unknown, or loading fails for any
                other reason. The factory treats this as "fall back".
        """
        super().__init__()
        self.cfg = cfg

        repo_dir = cfg.path("backbone.panderm.repo_dir")
        ckpt_path = cfg.path("backbone.panderm.checkpoint")
        model_name = str(cfg.get("backbone.panderm.model_name", "PanDerm_Base_LP"))

        if model_name not in PANDERM_VARIANTS:
            raise BackboneUnavailable(
                f"Unknown PanDerm variant {model_name!r}. "
                f"Expected one of {sorted(PANDERM_VARIANTS)}."
            )
        if not repo_dir.exists():
            raise BackboneUnavailable(
                f"PanDerm repository not found at {repo_dir}. "
                "Clone it with: git clone https://github.com/SiyuanYan1/PanDerm "
                f"{repo_dir.as_posix()}"
            )
        if not ckpt_path.exists():
            # Deployment targets (Streamlit Cloud, containers) cannot ship a
            # 328 MB file in the git repo, so optionally pull it from a
            # Hugging Face model repo on first run and cache it there.
            ckpt_path = self._fetch_checkpoint(cfg, ckpt_path)

        builder_name, expected_dim = PANDERM_VARIANTS[model_name]
        build_fn = self._import_builder(repo_dir, builder_name)

        try:
            model = build_fn()
        except Exception as exc:  # noqa: BLE001
            raise BackboneUnavailable(
                f"PanDerm builder {builder_name}() failed: {exc}"
            ) from exc

        self._load_checkpoint(model, ckpt_path)

        # The linear-probe variants keep a classification head we do not want.
        if hasattr(model, "head"):
            model.head = nn.Identity()

        self.model = model
        self.name = model_name
        self.feature_dim = int(cfg.get("backbone.panderm.feature_dim", None) or expected_dim)
        logger.info("Loaded PanDerm backbone: %s", self.describe())

    # -- loading helpers ---------------------------------------------------
    @staticmethod
    def _fetch_checkpoint(cfg: Config, ckpt_path: Path) -> Path:
        """Download the checkpoint from Hugging Face when it is not on disk.

        Hosted deployments cannot keep a 328 MB file in git, so
        ``backbone.panderm.hf_repo`` names a Hugging Face **model** repo to pull
        it from on first run. The weights may be stored there in fp16 (half the
        download); ``load_state_dict`` casts them back to the model's fp32
        parameters on load, which costs about 7e-5 in output probability.

        Args:
            cfg: Loaded project config.
            ckpt_path: Where the checkpoint was expected locally.

        Returns:
            Path to the downloaded file.

        Raises:
            BackboneUnavailable: If no ``hf_repo`` is configured, or the
                download fails. The factory turns this into the usual fallback.
        """
        hf_repo = cfg.get("backbone.panderm.hf_repo", None)
        if not hf_repo:
            raise BackboneUnavailable(
                f"PanDerm checkpoint not found at {ckpt_path}, and no "
                "backbone.panderm.hf_repo is configured to download it from. "
                "Download it from the PanDerm repository's release links."
            )

        hf_filename = str(cfg.get("backbone.panderm.hf_filename", ckpt_path.name))
        logger.info("Checkpoint missing locally; downloading %s/%s from Hugging Face "
                    "(first run only)...", hf_repo, hf_filename)
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=str(hf_repo),
                filename=hf_filename,
                repo_type="model",
            )
        except Exception as exc:  # noqa: BLE001
            raise BackboneUnavailable(
                f"Could not download the PanDerm checkpoint from {hf_repo}: {exc}"
            ) from exc

        logger.info("Downloaded PanDerm checkpoint -> %s", downloaded)
        return Path(downloaded)

    @staticmethod
    def _import_builder(repo_dir: Path, builder_name: str) -> Callable[[], nn.Module]:
        """Import PanDerm's model builder from the cloned repository.

        Upstream's ``classification`` package imports its siblings by top-level
        name (``from models.modeling_finetune import ...``), so that directory
        must be on ``sys.path``, not just the repo root.

        Raises:
            BackboneUnavailable: If the module or builder cannot be imported.
        """
        search_paths = [repo_dir / "classification", repo_dir]
        for path in search_paths:
            path_str = str(path.resolve())
            if path.exists() and path_str not in sys.path:
                sys.path.insert(0, path_str)

        last_error: Exception | None = None
        for module_name in ("models.modeling_finetune",
                            "classification.models.modeling_finetune"):
            try:
                module = __import__(module_name, fromlist=[builder_name])
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
            builder = getattr(module, builder_name, None)
            if builder is not None:
                return builder
            last_error = AttributeError(
                f"{module_name} has no builder named {builder_name!r}"
            )

        raise BackboneUnavailable(
            f"Could not import PanDerm builder {builder_name!r} from {repo_dir}: "
            f"{last_error}. PanDerm has its own dependencies (e.g. timm, "
            "open_clip_torch) -- see its README."
        )

    @staticmethod
    def _load_checkpoint(model: nn.Module, ckpt_path: Path) -> None:
        """Load a PanDerm checkpoint into ``model``.

        Mirrors upstream's loading logic: checkpoints may be a bare state dict
        or wrapped under ``model``/``state_dict``/``module``, and the large
        variant's keys carry an ``encoder.`` prefix that must be stripped.
        ``strict=False`` is intentional -- the pretraining head does not exist
        on the feature-extraction model.

        Raises:
            BackboneUnavailable: If the file cannot be read or nothing matches.
        """
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            raise BackboneUnavailable(
                f"Could not read PanDerm checkpoint {ckpt_path}: {exc}"
            ) from exc

        state_dict = checkpoint
        for key in ("model", "state_dict", "module"):
            if isinstance(state_dict, dict) and key in state_dict:
                state_dict = state_dict[key]
                break
        if not isinstance(state_dict, dict):
            raise BackboneUnavailable(
                f"Unexpected checkpoint structure in {ckpt_path}: "
                f"got {type(state_dict).__name__}."
            )

        cleaned = {
            k.replace("encoder.", "").replace("module.", ""): v
            for k, v in state_dict.items()
        }
        result = model.load_state_dict(cleaned, strict=False)

        matched = len(model.state_dict()) - len(getattr(result, "missing_keys", []))
        if matched <= 0:
            raise BackboneUnavailable(
                f"No parameters in {ckpt_path.name} matched the model -- the "
                "checkpoint and the configured model_name probably disagree."
            )
        if getattr(result, "missing_keys", None):
            logger.info("PanDerm: %d key(s) not present in the checkpoint (expected "
                        "for the pretraining head).", len(result.missing_keys))

    # -- FeatureBackbone API ----------------------------------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Embed a ``(B, 3, H, W)`` batch into ``(B, feature_dim)``.

        PanDerm's ViT returns pooled features once ``head`` is an Identity; if a
        variant returns per-token features instead, we mean-pool the tokens.
        """
        features = self.model(images)
        if features.ndim == 3:  # (B, tokens, C) -> mean over tokens
            features = features.mean(dim=1)
        return features

    def build_transform(self, train: bool = False) -> Callable[[Any], torch.Tensor]:
        """Return PanDerm's transform (ImageNet normalisation, per upstream)."""
        return default_transform(self.cfg, train=train)

    def gradcam_target_layers(self) -> List[nn.Module]:
        """Return the last transformer block's first LayerNorm.

        For ViTs this is the conventional Grad-CAM hook point: it still carries
        per-token spatial structure, which :meth:`gradcam_reshape_transform`
        folds back into a 2-D grid.
        """
        blocks = getattr(self.model, "blocks", None)
        if blocks is not None and len(blocks) > 0:
            last = blocks[-1]
            return [getattr(last, "norm1", last)]
        norms = [m for m in self.model.modules() if isinstance(m, nn.LayerNorm)]
        if norms:
            return [norms[-1]]
        raise BackboneUnavailable(f"No Grad-CAM target layer found in {self.name}.")

    def gradcam_reshape_transform(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Fold ``(B, tokens, C)`` ViT activations back into ``(B, C, H, W)``."""
        def reshape(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.ndim != 3:
                return tensor
            n_tokens = tensor.shape[1]
            grid = int(round(n_tokens ** 0.5))
            # Drop leading class/register tokens that do not sit on the grid.
            if grid * grid != n_tokens:
                grid = int(round((n_tokens - 1) ** 0.5))
                tensor = tensor[:, n_tokens - grid * grid:, :]
            result = tensor.reshape(tensor.size(0), grid, grid, tensor.size(2))
            return result.permute(0, 3, 1, 2)
        return reshape
