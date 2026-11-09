"""Export PanDerm for in-browser inference: ViT trunk to ONNX, head to JSON.

Two artefacts land in ``--out``:

* ``panderm_vit.onnx``  -- the ViT trunk. Input a preprocessed image, output
  all 197 tokens ``(1, 197, 768)``: the CLS token followed by a 14x14 grid of
  patch tokens.
* ``head.json``         -- the fusion head's weights, a few hundred KB.

**Why the split.** The head is tiny but has to run twice (once on the CLS token
for the prediction, once per patch token for the relevance map). Keeping both
copies inside the ONNX graph produced duplicate Gemm nodes that the dynamic
quantiser turned into a MatMulInteger graph with mismatched dimensions and
would not load. Exporting only the trunk leaves a clean, uniform transformer
that quantises reliably, and the head is small enough to reimplement in a
dozen lines of JavaScript.

**How the browser gets a saliency map.** Grad-CAM needs a backward pass, and
ONNX Runtime Web does inference only. Because the head is an MLP over a
CLS-pooled ViT, applying it to each patch token instead yields a
class-discriminative per-patch relevance map from the same forward pass. It
answers Grad-CAM's question without gradients, and is labelled a token
relevance map rather than Grad-CAM.

Usage::

    python scripts/export_onnx.py --out web/model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.dataset import (  # noqa: E402
    diagnosis_classes,
    diagnosis_description,
    diagnosis_name,
    metadata_dim,
    risk_for,
)
from src.model.risk_classifier import load_risk_classifier  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("export_onnx")


def _attention_matrix(attn: nn.Module, x: torch.Tensor,
                      rel_pos_bias: torch.Tensor | None) -> torch.Tensor:
    """Recompute one block's attention matrix from its own weights.

    The vendored ``Attention.forward`` does not return its attention, and
    editing third-party code to make it do so would be worse than recomputing
    it here. Mirrors that forward exactly, stopping after the softmax.
    """
    B, N, _ = x.shape
    qkv_bias = None
    if attn.q_bias is not None:
        qkv_bias = torch.cat(
            (attn.q_bias, torch.zeros_like(attn.v_bias), attn.v_bias))
    qkv = F.linear(x, attn.qkv.weight, qkv_bias)
    qkv = qkv.reshape(B, N, 3, attn.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    a = (q * attn.scale) @ k.transpose(-2, -1)
    if attn.relative_position_bias_table is not None:
        bias = attn.relative_position_bias_table[
            attn.relative_position_index.view(-1)].view(
            attn.window_size[0] * attn.window_size[1] + 1,
            attn.window_size[0] * attn.window_size[1] + 1, -1)
        a = a + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    if rel_pos_bias is not None:
        a = a + rel_pos_bias
    return a.softmax(dim=-1)


class TrunkWrapper(nn.Module):
    """ViT trunk: image -> all tokens, plus an attention-rollout saliency map.

    Returns ``(tokens, rollout)``:

    * ``tokens``  ``(B, 197, 768)`` — CLS token then the 14x14 patch grid.
    * ``rollout`` ``(B, 196)`` — attention rollout (Abnar & Zuidema 2020):
      how much of the CLS token's effective attention reaches each patch,
      accumulated through all 12 blocks with the residual path included.

    Rollout replaces the earlier "run the head on each patch token" saliency,
    which measured at a focus ratio of 1.03 — statistically indistinguishable
    from a uniform heatmap. On the same test images rollout scores 1.38 median
    and beats chance on 70% of them, and unlike Grad-CAM it needs no backward
    pass, so it runs in ONNX Runtime Web.

    The block forward is left untouched; attention is recomputed as a side
    branch, so the tokens -- and therefore every prediction -- are bit-identical
    to the previous export.
    """

    def __init__(self, vit: nn.Module) -> None:
        super().__init__()
        self.vit = vit

    def forward(self, image: torch.Tensor):
        """Mirror PanDerm's ``forward_features``, also accumulating rollout."""
        vit = self.vit
        x = vit.patch_embed(image)
        batch = x.shape[0]
        cls = vit.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls, x), dim=1)
        if vit.pos_embed is not None:
            x = x + vit.pos_embed.expand(batch, -1, -1).type_as(x)
        x = vit.pos_drop(x)
        rel_pos_bias = vit.rel_pos_bias() if vit.rel_pos_bias is not None else None

        n = x.shape[1]
        eye = torch.eye(n, dtype=x.dtype, device=x.device).unsqueeze(0)
        roll = eye
        for blk in vit.blocks:
            a = _attention_matrix(blk.attn, blk.norm1(x), rel_pos_bias)
            a = a.mean(dim=1) + eye                 # average heads, add residual
            a = a / a.sum(dim=-1, keepdim=True)
            roll = a @ roll
            x = blk(x, rel_pos_bias=rel_pos_bias)   # unchanged original path

        return vit.norm(x), roll[:, 0, 1:]


def dump_head(head: nn.Module, cfg) -> dict:
    """Serialise the fusion head's weights into a JSON-safe dict.

    The JS side replays exactly these ops:
    ``image_proj`` (Linear -> ReLU -> LayerNorm), ``metadata_encoder``
    (Linear -> ReLU -> LayerNorm), concat, then the MLP's
    (Linear -> ReLU -> BatchNorm) blocks and a final Linear.
    BatchNorm is folded to a plain scale/shift since inference uses running
    statistics.
    """
    def arr(t: torch.Tensor) -> list:
        return t.detach().cpu().numpy().astype(np.float32).ravel().tolist()

    def linear(mod: nn.Linear) -> dict:
        return {"w": arr(mod.weight), "b": arr(mod.bias),
                "in": mod.in_features, "out": mod.out_features}

    def layernorm(mod: nn.LayerNorm) -> dict:
        return {"g": arr(mod.weight), "b": arr(mod.bias), "eps": float(mod.eps)}

    def batchnorm(mod: nn.BatchNorm1d) -> dict:
        # Fold to y = x * scale + shift using the running statistics.
        var = mod.running_var.detach().cpu().numpy().astype(np.float64)
        mean = mod.running_mean.detach().cpu().numpy().astype(np.float64)
        gamma = mod.weight.detach().cpu().numpy().astype(np.float64)
        beta = mod.bias.detach().cpu().numpy().astype(np.float64)
        scale = gamma / np.sqrt(var + mod.eps)
        shift = beta - mean * scale
        return {"scale": scale.astype(np.float32).tolist(),
                "shift": shift.astype(np.float32).tolist()}

    codes = diagnosis_classes(cfg)
    out: dict = {
        # The head predicts lesion types; the browser derives the risk tier
        # from `risk_of`, so the two can never disagree.
        "classes": codes,
        "names": {c: diagnosis_name(cfg, c) for c in codes},
        "descriptions": {c: diagnosis_description(cfg, c) for c in codes},
        "risk_of": {c: risk_for(cfg, c) for c in codes},
        "risk_classes": list(cfg.get("labels.risk_classes")),
        "recommendations": dict(cfg.get("report.recommendations")),
        "metadata_dim": metadata_dim(cfg),
        "use_metadata": bool(head.use_metadata),
        "image_proj": {
            "linear": linear(head.image_proj[0]),
            "norm": layernorm(head.image_proj[2]),
        },
        "mlp": [],
    }

    if head.use_metadata:
        enc = head.metadata_encoder.net
        out["metadata_encoder"] = {
            "linear": linear(enc[0]),
            "norm": layernorm(enc[2]),
        }

    # Walk the MLP: Linear, ReLU, BatchNorm1d, Dropout, ... , final Linear.
    layers = list(head.mlp)
    i = 0
    while i < len(layers):
        mod = layers[i]
        if isinstance(mod, nn.Linear):
            block: dict = {"linear": linear(mod), "relu": False, "bn": None}
            j = i + 1
            while j < len(layers) and not isinstance(layers[j], nn.Linear):
                if isinstance(layers[j], nn.ReLU):
                    block["relu"] = True
                elif isinstance(layers[j], nn.BatchNorm1d):
                    block["bn"] = batchnorm(layers[j])
                j += 1
            out["mlp"].append(block)
            i = j
        else:
            i += 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="web/model")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    cfg = load_config()
    model = load_risk_classifier(cfg=cfg, strict=True).eval()
    trunk = TrunkWrapper(model.backbone.model).eval()

    size = int(cfg.get("backbone.image_size", 224))
    image = torch.randn(1, 3, size, size)
    with torch.no_grad():
        ref_tokens, ref_roll = trunk(image)
    logger.info("tokens %s  rollout %s", tuple(ref_tokens.shape), tuple(ref_roll.shape))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "panderm_vit.onnx"

    torch.onnx.export(
        trunk, (image,), str(onnx_path),
        input_names=["image"], output_names=["tokens", "rollout"],
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
    )
    logger.info("Wrote %s (%.0f MB)", onnx_path.name, onnx_path.stat().st_size / 1048576)

    head_path = out_dir / "head.json"
    head_path.write_text(json.dumps(dump_head(model.head, cfg)), encoding="utf-8")
    logger.info("Wrote %s (%.1f MB)", head_path.name, head_path.stat().st_size / 1048576)

    # Parity: ONNX trunk vs torch trunk.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": image.numpy()})
    diff = float(np.abs(got[0] - ref_tokens.numpy()).max())
    rdiff = float(np.abs(got[1] - ref_roll.numpy()).max())
    logger.info("trunk max abs diff vs torch: %.2e  (rollout %.2e)", diff, rdiff)
    if diff > 1e-3:
        logger.error("ONNX trunk diverges from PyTorch.")
        return 1

    print(f"\nOK  {onnx_path} ({onnx_path.stat().st_size/1048576:.0f} MB)")
    print(f"    {head_path} ({head_path.stat().st_size/1048576:.1f} MB)")
    print(f"    trunk diff {diff:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
