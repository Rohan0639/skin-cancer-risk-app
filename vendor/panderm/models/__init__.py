"""Vendored subset of PanDerm (https://github.com/SiyuanYan1/PanDerm).

PanDerm is released under CC-BY-NC: non-commercial academic research only,
with attribution. Only ``modeling_finetune`` is copied here -- it is
self-contained (torch + timm) and is all that is needed to build the backbone.

This module is deliberately empty. Upstream's ``__init__`` imports
``builder.py``, which pulls in ``open_clip``; importing it would drag a large
dependency into every deployment for no benefit.

Citation: Yan et al., "A general-purpose multimodal foundation model for
dermatology", Nature Medicine (2025).
"""
