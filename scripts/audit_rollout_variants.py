"""Why does the attention peak miss the lesion, and is it fixable?

The rollout scores 0.127 on Pointing Game against a 0.287 random baseline: its
single hottest pixel avoids the lesion. Meanwhile focus ratio is 1.245, so the
bulk of the heat IS on the lesion. Both can be true if a few isolated
high-attention patches sit on background skin and outrank the lesion's broad,
moderate response.

That pattern is the documented ViT artifact-token effect (Darcet et al. 2023,
"Vision Transformers Need Registers"): ViTs repurpose a handful of
low-information background patches as global scratch space, and those tokens
acquire very high attention while carrying no local signal.

This tests that explanation directly by scoring variants that each target one
suspected cause, against the same expert masks:
  max-fuse / mean-fuse over heads   -- max propagates a single rogue head
  start layer 0 / 3 / 6 / 9         -- artifacts concentrate in deeper blocks
  percentile clip at 98             -- caps the outlier spikes
  3x3 median filter on the grid     -- isolated single-patch spikes removed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

from audit_localization import load_mask, score  # noqa: E402
from export_onnx import _attention_matrix  # noqa: E402
from src.config import load_config  # noqa: E402
from src.model.risk_classifier import load_risk_classifier  # noqa: E402
from src.utils import get_logger, load_image_rgb  # noqa: E402

logger = get_logger("variants")
ANN = Path("data/raw/ham10000_dn/ds/ann")
OUT = Path("audit_reports")
N = 200


def rollout(vit, image, start, fuse):
    """Attention rollout over the trunk, fusing heads with max or mean."""
    x = vit.patch_embed(image)
    b = x.shape[0]
    x = torch.cat([vit.cls_token.expand(b, -1, -1), x], dim=1)
    if vit.pos_embed is not None:
        x = x + vit.pos_embed.expand(b, -1, -1).type_as(x)
    x = vit.pos_drop(x)
    bias = vit.rel_pos_bias() if vit.rel_pos_bias is not None else None
    n = x.shape[1]
    roll = torch.eye(n, device=x.device).unsqueeze(0).expand(b, -1, -1).clone()
    eye = torch.eye(n, device=x.device).unsqueeze(0)
    for i, blk in enumerate(vit.blocks):
        if i >= start:
            a = _attention_matrix(blk.attn, blk.norm1(x), bias)
            a = (a.max(dim=1).values if fuse == "max" else a.mean(dim=1)) + eye
            a = a / a.sum(dim=-1, keepdim=True)
            roll = a @ roll
        x = blk(x, rel_pos_bias=bias)
    return roll[:, 0, 1:]


def median3(g: np.ndarray) -> np.ndarray:
    """3x3 median filter: removes single-patch spikes, keeps real regions."""
    p = np.pad(g, 1, mode="edge")
    return np.median(np.stack([p[i:i + g.shape[0], j:j + g.shape[1]]
                               for i in range(3) for j in range(3)]), axis=0)


def to_heat(v, shape, clip=None, med=False):
    g = int(np.sqrt(v.size))
    a = v.reshape(g, g).astype(np.float64)
    if med:
        a = median3(a)
    if clip is not None:
        a = np.clip(a, 0, np.percentile(a, clip))
    a = a / max(a.max(), 1e-8)
    return np.array(Image.fromarray((a * 255).astype(np.uint8))
                    .resize((shape[1], shape[0]), Image.BILINEAR)).astype(np.float64)


def main() -> int:
    cfg = load_config()
    model = load_risk_classifier(cfg=cfg, strict=True).eval()
    dev = next(model.head.parameters()).device
    vit = model.backbone.model
    tf = model.backbone.build_transform(train=False)

    df = pd.read_csv("data/processed/ham10000.csv")
    df["stem"] = [Path(str(p)).stem for p in df.image_path]
    df = df[[(ANN / f"{s}.jpg.json").exists() for s in df.stem]]
    df = df.sample(n=min(N, len(df)), random_state=0)

    variants = {
        "deployed: max-fuse, block 6":        dict(start=6, fuse="max"),
        "mean-fuse, block 6":                 dict(start=6, fuse="mean"),
        "mean-fuse, block 0 (all blocks)":    dict(start=0, fuse="mean"),
        "mean-fuse, block 3":                 dict(start=3, fuse="mean"),
        "mean-fuse, block 9":                 dict(start=9, fuse="mean"),
        "max-fuse b6 + clip p98":             dict(start=6, fuse="max", clip=98),
        "max-fuse b6 + 3x3 median":           dict(start=6, fuse="max", med=True),
        "mean-fuse b6 + 3x3 median":          dict(start=6, fuse="mean", med=True),
    }
    acc = {k: [] for k in variants}
    cached = {}

    for _, r in df.iterrows():
        rgb = load_image_rgb(cfg.resolve(r.image_path))
        mask = load_mask(r.stem, rgb.shape[:2])
        if mask is None:
            continue
        x = tf(Image.fromarray(rgb)).unsqueeze(0).to(dev)
        for name, kw in variants.items():
            key = (kw["start"], kw["fuse"])
            if key not in cached:
                with torch.no_grad():
                    cached[key] = rollout(vit, x, kw["start"], kw["fuse"])[0].float().cpu().numpy()
            heat = to_heat(cached[key], rgb.shape[:2],
                           clip=kw.get("clip"), med=kw.get("med", False))
            acc[name].append(score(heat, mask))
        cached.clear()

    rows = {}
    for name, lst in acc.items():
        d = pd.DataFrame(lst)
        rows[name] = {
            "pointing_game": float(d.pointing_hit.mean()),
            "iou": float(d.iou.mean()),
            "dice": float(d.dice.mean()),
            "focus_ratio": float(d.focus_ratio.mean()),
            "frac_above_chance": float((d.focus_ratio > 1.0).mean()),
        }
    n_eval = len(acc["deployed: max-fuse, block 6"])
    out = {"n_images": n_eval,
           "random_pointing_baseline_equals_mean_lesion_area": True,
           "variants": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rollout_variants.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n{'=' * 86}")
    print(f"  ATTENTION ROLLOUT VARIANTS vs EXPERT MASKS  (n={n_eval})")
    print(f"{'-' * 86}")
    print(f"  {'variant':34s} {'Pointing':>9s} {'IoU':>7s} {'Dice':>7s} "
          f"{'Focus':>7s} {'>chance':>9s}")
    base = rows["deployed: max-fuse, block 6"]
    for k, v in sorted(rows.items(), key=lambda kv: -kv[1]["pointing_game"]):
        mark = "  <- deployed" if k.startswith("deployed") else ""
        print(f"  {k:34s} {v['pointing_game']:9.3f} {v['iou']:7.3f} {v['dice']:7.3f} "
              f"{v['focus_ratio']:7.3f} {v['frac_above_chance']:8.1%}{mark}")
    print(f"{'=' * 86}")
    best = max(rows.items(), key=lambda kv: kv[1]["pointing_game"])
    print(f"  best Pointing: {best[0]} ({best[1]['pointing_game']:.3f} vs "
          f"deployed {base['pointing_game']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
