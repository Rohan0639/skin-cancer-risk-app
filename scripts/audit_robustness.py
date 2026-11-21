"""Phase 7 + 8: robustness to image perturbations, and GPU/inference timing.

Each perturbation is applied to the whole held-out test split and scored with
the same deployed model, so the drop is attributable to the perturbation alone.
Perturbations are synthetic: a drop here indicates brittleness, but holding up
is NOT clinical validation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageEnhance, ImageFilter  # noqa: E402

from src.config import load_config  # noqa: E402
from src.dataset import (  # noqa: E402
    TARGET_COLUMN, build_metadata_matrix, diagnosis_classes,
    group_aware_split, load_unified_csv, risk_for,
)
from src.model.risk_classifier import load_risk_classifier  # noqa: E402
from src.utils import get_logger, load_image_rgb  # noqa: E402

logger = get_logger("robustness")
OUT = Path("audit_reports")

PERTURBATIONS = {
    "none (baseline)":      lambda im: im,
    "rotate 10 deg":        lambda im: im.rotate(10, resample=Image.BILINEAR, expand=False),
    "rotate 90 deg":        lambda im: im.rotate(90, expand=True),
    "horizontal flip":      lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    "gaussian blur r=2":    lambda im: im.filter(ImageFilter.GaussianBlur(2)),
    "brightness x1.4":      lambda im: ImageEnhance.Brightness(im).enhance(1.4),
    "brightness x0.6":      lambda im: ImageEnhance.Brightness(im).enhance(0.6),
    "contrast x0.5":        lambda im: ImageEnhance.Contrast(im).enhance(0.5),
    "saturation x0.0 (grey)": lambda im: ImageEnhance.Color(im).enhance(0.0),
    "downscale to 64px":    lambda im: im.resize((64, 64), Image.BILINEAR).resize(im.size, Image.BILINEAR),
    "jpeg quality 20":      None,  # handled separately
}


def jpeg20(im: Image.Image) -> Image.Image:
    import io
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=20)
    b.seek(0)
    return Image.open(b).convert("RGB")


def main() -> int:
    cfg = load_config()
    classes = diagnosis_classes(cfg)
    c2i = {c: i for i, c in enumerate(classes)}
    df = load_unified_csv(cfg)
    tr, va, te = group_aware_split(df, cfg)

    model = load_risk_classifier(cfg=cfg, strict=True).eval()
    dev = next(model.head.parameters()).device
    tf = model.backbone.build_transform(train=False)
    M = torch.from_numpy(build_metadata_matrix(te, cfg).astype(np.float32)).to(dev)
    y = np.array([c2i[v] for v in te[TARGET_COLUMN]])
    mel = c2i["MEL"]

    raw = [Image.fromarray(load_image_rgb(cfg.resolve(p))) for p in te.image_path]
    logger.info("loaded %d test images", len(raw))

    def evaluate(fn):
        X = torch.stack([tf(fn(im)) for im in raw])
        P = []
        with torch.no_grad():
            for i in range(0, len(X), 32):
                f = model.backbone(X[i:i + 32].to(dev)).float()
                P.append(torch.softmax(model.head(f, M[i:i + 32]), 1).cpu().numpy())
        P = np.concatenate(P)
        pred = P.argmax(1)
        shown_low = np.mean([risk_for(cfg, classes[c]) == "Low" for c in pred[y == mel]])
        return {
            "accuracy": float((pred == y).mean()),
            "macro_recall": float(np.mean([(pred[y == i] == i).mean()
                                           for i in range(len(classes)) if (y == i).sum()])),
            "melanoma_recall": float((pred[y == mel] == mel).mean()),
            "melanomas_shown_low_risk": float(shown_low),
            "mean_confidence": float(P.max(1).mean()),
        }

    results = {}
    for name, fn in PERTURBATIONS.items():
        f = jpeg20 if fn is None else fn
        results[name] = evaluate(f)
        logger.info("%-24s acc %.3f", name, results[name]["accuracy"])

    # ---- Phase 8: device + timing -------------------------------------------
    perf = {
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "model_device": str(dev),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        if torch.cuda.is_available() else None,
    }
    one = tf(raw[0]).unsqueeze(0).to(dev)
    m1 = M[:1]
    with torch.no_grad():                      # warm up kernels first
        for _ in range(5):
            model.head(model.backbone(one).float(), m1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(30):
            model.head(model.backbone(one).float(), m1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    perf["single_image_ms"] = round((time.perf_counter() - t0) / 30 * 1000, 2)

    batch = torch.stack([tf(im) for im in raw[:32]]).to(dev)
    with torch.no_grad():
        model.backbone(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(5):
            model.head(model.backbone(batch).float(), M[:32])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    el = (time.perf_counter() - t0) / 5
    perf["batch32_ms"] = round(el * 1000, 2)
    perf["throughput_img_per_s"] = round(32 / el, 1)
    if torch.cuda.is_available():
        perf["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1048576, 1)

    base = results["none (baseline)"]["accuracy"]
    out = {"n_test_images": len(y),
           "caveat": ("Synthetic perturbations of the held-out split. A drop shows "
                      "brittleness; holding up is not clinical validation."),
           "baseline_accuracy": base, "perturbations": results, "performance": perf}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "robustness_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n{'=' * 84}")
    print(f"  ROBUSTNESS - held-out test split, n={len(y)}")
    print(f"{'-' * 84}")
    print(f"  {'perturbation':26s} {'acc':>7s} {'delta':>8s} {'MEL rec':>9s} "
          f"{'MEL->Low':>9s} {'conf':>7s}")
    for k, v in results.items():
        d = "" if k.startswith("none") else f"{v['accuracy'] - base:+.3f}"
        print(f"  {k:26s} {v['accuracy']:7.3f} {d:>8s} {v['melanoma_recall']:9.3f} "
              f"{v['melanomas_shown_low_risk']:9.1%} {v['mean_confidence']:7.3f}")
    print(f"{'=' * 84}")
    print(f"  device {perf['model_device']} | {perf.get('gpu_name')} | "
          f"CUDA {perf['cuda_available']}")
    print(f"  single image {perf['single_image_ms']} ms | batch-32 "
          f"{perf['throughput_img_per_s']} img/s | peak VRAM {perf.get('peak_vram_mb')} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
