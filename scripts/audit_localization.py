"""Does the attention map actually land on the lesion? Measured, not eyeballed.

The app's heatmap is attention rollout taken from the FROZEN PanDerm trunk, so
it is a property of the backbone and does not depend on which head sits on top.
HAM10000 ships expert segmentation masks (DatasetNinja bitmap format), which
lets this be scored properly instead of judged by eye.

Reported against those masks:
  Pointing Game  does the hottest pixel fall inside the lesion?
  IoU / Dice     overlap of the thresholded heatmap with the mask
  Focus ratio    (heat inside lesion / total heat) / (lesion area fraction).
                 1.0 is exactly chance, so this is the number that says whether
                 the map carries information at all.

A random-uniform map is scored alongside as the chance baseline, because IoU
and Dice on their own look respectable for any blob that covers the middle of
the frame.

Caveat recorded in the output: the masks are dermoscopic HAM10000 images, while
the deployed classifier is trained on clinical photographs. This measures the
backbone's attention, which is shared, not the clinical classifier's accuracy.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

from export_onnx import TrunkWrapper  # noqa: E402
from src.config import load_config  # noqa: E402
from src.model.risk_classifier import load_risk_classifier  # noqa: E402
from src.utils import get_logger, load_image_rgb  # noqa: E402

logger = get_logger("localization")
ANN = Path("data/raw/ham10000_dn/ds/ann")
OUT = Path("audit_reports")
N_IMAGES = 300


def load_mask(stem: str, shape) -> np.ndarray | None:
    """Decode a DatasetNinja bitmap annotation into a full-frame boolean mask."""
    p = ANN / f"{stem}.jpg.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text(encoding="utf-8"))
    full = np.zeros((j["size"]["height"], j["size"]["width"]), dtype=bool)
    for obj in j.get("objects", []):
        bm = obj.get("bitmap")
        if not bm:
            continue
        sub = np.array(Image.open(io.BytesIO(
            zlib.decompress(base64.b64decode(bm["data"]))))) > 0
        x0, y0 = bm.get("origin", [0, 0])
        h, w = sub.shape
        y1, x1 = min(y0 + h, full.shape[0]), min(x0 + w, full.shape[1])
        if y1 > y0 and x1 > x0:
            full[y0:y1, x0:x1] |= sub[:y1 - y0, :x1 - x0]
    if not full.any():
        return None
    if full.shape != shape:
        full = np.array(Image.fromarray(full.astype(np.uint8) * 255)
                        .resize((shape[1], shape[0]), Image.NEAREST)) > 127
    return full


def score(heat: np.ndarray, mask: np.ndarray) -> dict:
    """Pointing game, IoU, Dice and the scale-free focus ratio."""
    h = heat / max(heat.sum(), 1e-8)
    area = mask.mean()
    inside = h[mask].sum()
    yx = np.unravel_index(np.argmax(heat), heat.shape)
    thr = heat >= np.percentile(heat, 100 * (1 - area))  # same area as the mask
    inter = np.logical_and(thr, mask).sum()
    return {
        "pointing_hit": bool(mask[yx]),
        "iou": float(inter / max(np.logical_or(thr, mask).sum(), 1)),
        "dice": float(2 * inter / max(thr.sum() + mask.sum(), 1)),
        "focus_ratio": float(inside / max(area, 1e-8)),
        "lesion_area_fraction": float(area),
    }


def main() -> int:
    cfg = load_config()
    model = load_risk_classifier(cfg=cfg, strict=True).eval()
    device = next(model.head.parameters()).device
    trunk = TrunkWrapper(model.backbone.model).eval().to(device)
    tf = model.backbone.build_transform(train=False)

    df = pd.read_csv("data/processed/ham10000.csv")
    df["stem"] = [Path(str(p)).stem for p in df.image_path]
    df = df[[(ANN / f"{s}.jpg.json").exists() for s in df.stem]]
    logger.info("%d dermoscopic images carry an expert mask", len(df))
    df = df.sample(n=min(N_IMAGES, len(df)), random_state=0)

    rng = np.random.RandomState(0)
    rows, chance = [], []
    for _, r in df.iterrows():
        rgb = load_image_rgb(cfg.resolve(r.image_path))
        mask = load_mask(r.stem, rgb.shape[:2])
        if mask is None:
            continue
        x = tf(Image.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            _, roll = trunk(x)
        v = roll[0].float().cpu().numpy()
        g = int(np.sqrt(v.size))
        heat = np.array(Image.fromarray(
            (v.reshape(g, g) / max(v.max(), 1e-8) * 255).astype(np.uint8))
            .resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR)).astype(np.float64)
        s = score(heat, mask); s["image_id"] = r.stem; s["diagnosis"] = r.diagnosis
        rows.append(s)
        chance.append(score(rng.rand(*mask.shape), mask))

    res = pd.DataFrame(rows)
    ch = pd.DataFrame(chance)
    summary = {
        "note": ("Masks are dermoscopic HAM10000 (expert annotations). The attention "
                 "rollout comes from the FROZEN PanDerm trunk, which is identical "
                 "whatever head sits on it, so this measures the backbone's "
                 "localization. It is NOT a measure of the clinical classifier's "
                 "accuracy, and these images are not the clinical test split."),
        "n_images": int(len(res)),
        "method": "attention rollout, max-fused over heads from block 6",
        "attention_rollout": {
            "pointing_game_accuracy": float(res.pointing_hit.mean()),
            "mean_iou": float(res.iou.mean()),
            "mean_dice": float(res.dice.mean()),
            "mean_focus_ratio": float(res.focus_ratio.mean()),
            "median_focus_ratio": float(res.focus_ratio.median()),
            "frac_above_chance_focus": float((res.focus_ratio > 1.0).mean()),
        },
        "random_baseline": {
            "pointing_game_accuracy": float(ch.pointing_hit.mean()),
            "mean_iou": float(ch.iou.mean()),
            "mean_dice": float(ch.dice.mean()),
            "mean_focus_ratio": float(ch.focus_ratio.mean()),
        },
        "mean_lesion_area_fraction": float(res.lesion_area_fraction.mean()),
        "per_class_focus_ratio": {
            k: round(float(v), 3) for k, v in res.groupby("diagnosis").focus_ratio.mean().items()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "localization_metrics.json").write_text(json.dumps(summary, indent=2),
                                                   encoding="utf-8")
    res.to_csv(OUT / "localization_per_image.csv", index=False)

    a, b = summary["attention_rollout"], summary["random_baseline"]
    print(f"\n{'=' * 68}")
    print(f"  ATTENTION LOCALIZATION vs EXPERT MASKS  (n={len(res)})")
    print(f"{'-' * 68}")
    print(f"  {'metric':24s} {'rollout':>10s} {'random':>10s}")
    for k, lbl in [("pointing_game_accuracy", "Pointing Game"), ("mean_iou", "IoU"),
                   ("mean_dice", "Dice"), ("mean_focus_ratio", "Focus ratio")]:
        print(f"  {lbl:24s} {a[k]:10.3f} {b[k]:10.3f}")
    print(f"  {'above-chance focus':24s} {a['frac_above_chance_focus']:10.1%}")
    print(f"  {'mean lesion area':24s} {summary['mean_lesion_area_fraction']:10.1%}")
    print(f"{'=' * 68}")
    print(f"  wrote {OUT}/localization_metrics.json + localization_per_image.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
