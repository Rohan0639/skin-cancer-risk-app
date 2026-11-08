# Vendored PanDerm subset

`models/modeling_finetune.py` is copied verbatim from
[PanDerm](https://github.com/SiyuanYan1/PanDerm) (`classification/models/`).

It is vendored rather than cloned so that hosted deployments (Streamlit Cloud,
containers) can build the backbone without fetching the full upstream repo and
its `open_clip` dependency. Nothing in it has been modified.

**Licence:** CC-BY-NC — non-commercial academic research only, with
attribution. See the project README under "Licences and attribution".

**Citation:** Yan et al., "A general-purpose multimodal foundation model for
dermatology", *Nature Medicine* (2025).

The model **weights** are not vendored. They are downloaded at runtime from the
Hugging Face repo named by `backbone.panderm.hf_repo` in `configs/default.yaml`.
