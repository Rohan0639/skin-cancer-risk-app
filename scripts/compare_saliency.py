"""Compare saliency methods by how well they actually land on the lesion.

The browser build shipped a "token relevance" map: the classifier head applied
to each patch token. It localises badly -- measured focus ratios were often
below 1.0, i.e. worse than a uniform heatmap. The likely reason is that the
head was trained only on the CLS token, so patch tokens are out of
distribution for it and its output on them is close to meaningless.

This script scores candidate replacements on real test images using

    focus ratio = (heat inside lesion / total heat) / (lesion area fraction)

so 1.0 means "no better than chance" and higher is better. Grad-CAM is
included as a reference upper bound; it needs gradients and therefore cannot
run in ONNX Runtime Web, but it shows what a good method achieves here.

Usage::

    python scripts/compare_saliency.py --n 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config  # noqa: E402
from src.dataset import encode_metadata  # noqa: E402
from src.model.risk_classifier import load_risk_classifier  # noqa: E402
from src.segmentation import segment_lesion  # noqa: E402
from src.utils import get_logger, load_image_rgb  # noqa: E402

logger = get_logger("compare_saliency")
GRID = 14


def attention_maps(vit: torch.nn.Module, image: torch.Tensor) -> List[torch.Tensor]:
    """Run the trunk and return each block's attention matrix.

    The vendored ``Attention.forward`` does not return its attention, so this
    hooks each module's *input* and recomputes the matrix from the module's own
    weights. That avoids editing third-party code and stays exact.
    """
    attns: List[torch.Tensor] = []
    handles = []

    def make_hook(module):
        def hook(mod, inputs):
            x = inputs[0]
            B, N, C = x.shape
            qkv_bias = None
            if mod.q_bias is not None:
                qkv_bias = torch.cat(
                    (mod.q_bias, torch.zeros_like(mod.v_bias), mod.v_bias))
            qkv = F.linear(x, mod.qkv.weight, qkv_bias)
            qkv = qkv.reshape(B, N, 3, mod.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            attn = (q * mod.scale) @ k.transpose(-2, -1)
            if mod.relative_position_bias_table is not None:
                bias = mod.relative_position_bias_table[
                    mod.relative_position_index.view(-1)].view(
                    mod.window_size[0] * mod.window_size[1] + 1,
                    mod.window_size[0] * mod.window_size[1] + 1, -1)
                attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            attns.append(attn.softmax(dim=-1).detach())
        return hook

    for blk in vit.blocks:
        handles.append(blk.attn.register_forward_pre_hook(make_hook(blk.attn)))
    try:
        with torch.no_grad():
            _ = trunk_forward(vit, image)
    finally:
        for h in handles:
            h.remove()
    return attns


def trunk_forward(vit: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Replay PanDerm's forward_features up to the final norm."""
    x = vit.patch_embed(image)
    b = x.shape[0]
    x = torch.cat((vit.cls_token.expand(b, -1, -1), x), dim=1)
    if vit.pos_embed is not None:
        x = x + vit.pos_embed.expand(b, -1, -1).type_as(x)
    x = vit.pos_drop(x)
    rpb = vit.rel_pos_bias() if vit.rel_pos_bias is not None else None
    for blk in vit.blocks:
        x = blk(x, rel_pos_bias=rpb)
    return vit.norm(x)


def rollout(attns: List[torch.Tensor]) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema 2020): what the CLS token attends to.

    Averages heads, adds the residual connection as an identity, renormalises,
    and multiplies the per-layer matrices together. Row 0 of the product is the
    CLS token's effective attention over the patches.
    """
    result = None
    for a in attns:
        a = a.mean(dim=1)                       # average heads -> (B, N, N)
        a = a + torch.eye(a.shape[-1], device=a.device)   # residual path
        a = a / a.sum(dim=-1, keepdim=True)
        result = a if result is None else a @ result
    cls_row = result[0, 0, 1:]                  # CLS -> patches
    return cls_row.cpu().numpy()


def last_layer_attention(attns: List[torch.Tensor]) -> np.ndarray:
    """CLS attention from the final block only -- the cheapest baseline."""
    return attns[-1].mean(dim=1)[0, 0, 1:].cpu().numpy()


def head_patch_logits(model, tokens: torch.Tensor, meta: torch.Tensor,
                      target: int) -> np.ndarray:
    """The method currently shipped: run the head on every patch token."""
    patches = tokens[:, 1:, :]
    b, n, d = patches.shape
    flat = patches.reshape(b * n, d)
    meta_rep = meta.expand(b, n, meta.shape[1]).reshape(b * n, -1)
    with torch.no_grad():
        logits = model.head(flat, meta_rep).reshape(n, -1)
    other = (logits.sum(dim=1) - logits[:, target]) / (logits.shape[1] - 1)
    return (logits[:, target] - other).cpu().numpy()


def grad_cam(model, image: torch.Tensor, meta: torch.Tensor, target: int) -> np.ndarray:
    """Real Grad-CAM on the last block -- reference only; needs gradients."""
    vit = model.backbone.model
    acts: Dict[str, torch.Tensor] = {}
    grads: Dict[str, torch.Tensor] = {}
    layer = vit.blocks[-1].norm1

    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o))
    h2 = layer.register_full_backward_hook(
        lambda m, gi, go: grads.__setitem__("v", go[0]))
    try:
        for p in model.backbone.parameters():
            p.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        tokens = trunk_forward_grad(vit, image)
        logits = model.head(tokens[:, 0], meta)
        logits[0, target].backward()
        a = acts["v"][:, 1:, :]
        g = grads["v"][:, 1:, :]
        weights = g.mean(dim=1, keepdim=True)
        cam = torch.relu((weights * a).sum(dim=-1))[0]
        return cam.detach().cpu().numpy()
    finally:
        h1.remove(); h2.remove()
        for p in model.backbone.parameters():
            p.requires_grad_(False)


def trunk_forward_grad(vit, image):
    """Same as trunk_forward but keeps the graph for backprop."""
    x = vit.patch_embed(image)
    b = x.shape[0]
    x = torch.cat((vit.cls_token.expand(b, -1, -1), x), dim=1)
    if vit.pos_embed is not None:
        x = x + vit.pos_embed.expand(b, -1, -1).type_as(x)
    x = vit.pos_drop(x)
    rpb = vit.rel_pos_bias() if vit.rel_pos_bias is not None else None
    for blk in vit.blocks:
        x = blk(x, rel_pos_bias=rpb)
    return vit.norm(x)


def focus_ratio(heat: np.ndarray, mask: np.ndarray) -> float | None:
    """Heat mass inside the lesion, divided by what chance would give."""
    h = heat.reshape(GRID, GRID).astype(np.float64)
    h = h - h.min()
    if h.max() <= 1e-12:
        return None
    h = h / h.max()

    H, W = mask.shape
    small = np.zeros((GRID, GRID), dtype=bool)
    for gy in range(GRID):
        for gx in range(GRID):
            y0, y1 = int(gy * H / GRID), max(int((gy + 1) * H / GRID), int(gy * H / GRID) + 1)
            x0, x1 = int(gx * W / GRID), max(int((gx + 1) * W / GRID), int(gx * W / GRID) + 1)
            small[gy, gx] = mask[y0:y1, x0:x1].mean() > 0.5

    area = small.mean()
    if area <= 0 or area >= 1:
        return None
    total = h.sum()
    if total <= 1e-12:
        return None
    return float((h[small].sum() / total) / area)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=25)
    args = parser.parse_args()

    import pandas as pd
    from PIL import Image

    cfg = load_config()
    model = load_risk_classifier(cfg=cfg, strict=True).eval()
    vit = model.backbone.model
    transform = model.backbone.build_transform(train=False)

    splits = pd.read_csv(cfg.path("paths.output_dir") / "splits.csv")
    test = set(splits[splits.split == "test"].image_path)
    df = pd.read_csv(cfg.path("paths.unified_csv"))
    df = df[df.image_path.isin(test)].sample(n=args.n, random_state=0)

    scores: Dict[str, List[float]] = {
        "head-on-patches (shipped)": [], "last-layer attention": [],
        "attention rollout": [], "Grad-CAM (reference)": []}

    for row in df.to_dict("records"):
        rgb = load_image_rgb(Config.resolve(row["image_path"]))
        seg = segment_lesion(rgb)
        if not seg.found:
            continue
        image = transform(Image.fromarray(rgb)).unsqueeze(0)
        meta = torch.from_numpy(
            encode_metadata(row["age"], row["sex"], row["site"], cfg)).unsqueeze(0)

        with torch.no_grad():
            tokens = trunk_forward(vit, image)
            target = int(model.head(tokens[:, 0], meta).argmax(dim=1).item())

        attns = attention_maps(vit, image)
        mask = (seg.mask > 0)
        for name, heat in (
            ("head-on-patches (shipped)", head_patch_logits(model, tokens, meta, target)),
            ("last-layer attention", last_layer_attention(attns)),
            ("attention rollout", rollout(attns)),
            ("Grad-CAM (reference)", grad_cam(model, image, meta, target)),
        ):
            fr = focus_ratio(heat, mask)
            if fr is not None:
                scores[name].append(fr)

    print()
    print("=" * 62)
    print(f"  focus ratio on {len(scores['attention rollout'])} test images")
    print("  (1.0 = no better than chance; higher is better)")
    print("-" * 62)
    for name, vals in scores.items():
        if not vals:
            continue
        v = np.array(vals)
        print(f"  {name:28s} mean {v.mean():5.2f}  median {np.median(v):5.2f}"
              f"  >1: {(v > 1).mean():4.0%}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
