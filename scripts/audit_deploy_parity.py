"""TASK 1 - full classification parity, deployed model vs new model.

Compares the complete chain on every held-out test image:
  embeddings -> logits -> softmax -> predicted class -> risk tier

The rollout fix touches a side branch only, so all five should be identical.
Anything else blocks deployment.

Also records the fp32->int8 quantisation cost, which is a PRE-EXISTING property
of the deployed artifact (old and new int8 tokens are bit-identical), reported
here so it is not mistaken for a regression introduced by this change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config  # noqa: E402
from src.dataset import (  # noqa: E402
    TARGET_COLUMN, build_metadata_matrix, diagnosis_classes,
    group_aware_split, load_unified_csv, risk_for,
)
from src.model.risk_classifier import load_risk_classifier  # noqa: E402
from src.utils import get_logger, load_image_rgb  # noqa: E402

logger = get_logger("deploy_parity")
DEPLOYED = Path("web/model/panderm_vit_int8.onnx")
OUT = Path("audit_reports")


def preprocess(rgb, cfg):
    from PIL import Image
    size = int(cfg.get("backbone.image_size", 224))
    mean = np.array(cfg.get("backbone.normalize_mean"), dtype=np.float32)
    std = np.array(cfg.get("backbone.normalize_std"), dtype=np.float32)
    im = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)
    a = (np.asarray(im, dtype=np.float32) / 255.0 - mean) / std
    return a.transpose(2, 0, 1)[None].astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", required=True, help="new int8 model")
    ap.add_argument("--new-fp32", default="", help="optional fp32 trunk, for quant cost")
    args = ap.parse_args()

    import onnxruntime as ort

    cfg = load_config()
    classes = diagnosis_classes(cfg)
    model = load_risk_classifier(cfg=cfg, strict=True).eval()
    head = model.head.cpu().eval()

    df = load_unified_csv(cfg)
    _, _, te = group_aware_split(df, cfg)
    meta = build_metadata_matrix(te, cfg).astype(np.float32)
    logger.info("comparing on all %d held-out test images", len(te))

    s_old = ort.InferenceSession(str(DEPLOYED), providers=["CPUExecutionProvider"])
    s_new = ort.InferenceSession(str(args.new), providers=["CPUExecutionProvider"])
    s_f32 = (ort.InferenceSession(args.new_fp32, providers=["CPUExecutionProvider"])
             if args.new_fp32 else None)

    d_emb = d_log = d_prob = 0.0
    n_cls = n_tier = n = 0
    q_cls = q_tier = 0
    q_emb = q_prob = 0.0

    for i, row in enumerate(te.to_dict("records")):
        img = preprocess(load_image_rgb(Config.resolve(row["image_path"])), cfg)
        m = torch.from_numpy(meta[i])[None]

        e_old = torch.from_numpy(s_old.run(None, {"image": img})[0].mean(axis=1))
        e_new = torch.from_numpy(s_new.run(None, {"image": img})[0].mean(axis=1))
        with torch.no_grad():
            l_old, l_new = head(e_old, m), head(e_new, m)
            p_old = torch.softmax(l_old, 1)[0].numpy()
            p_new = torch.softmax(l_new, 1)[0].numpy()

        d_emb = max(d_emb, float((e_old - e_new).abs().max()))
        d_log = max(d_log, float((l_old - l_new).abs().max()))
        d_prob = max(d_prob, float(np.abs(p_old - p_new).max()))
        c_old, c_new = classes[int(p_old.argmax())], classes[int(p_new.argmax())]
        n_cls += int(c_old == c_new)
        n_tier += int(risk_for(cfg, c_old) == risk_for(cfg, c_new))
        n += 1

        if s_f32 is not None:
            e_f = torch.from_numpy(s_f32.run(None, {"image": img})[0].mean(axis=1))
            with torch.no_grad():
                p_f = torch.softmax(head(e_f, m), 1)[0].numpy()
            q_emb = max(q_emb, float((e_f - e_new).abs().max()))
            q_prob = max(q_prob, float(np.abs(p_f - p_new).max()))
            c_f = classes[int(p_f.argmax())]
            q_cls += int(c_f == c_new)
            q_tier += int(risk_for(cfg, c_f) == risk_for(cfg, c_new))

    res = {
        "n_images": n,
        "comparison": "deployed int8 (old rollout) vs new int8 (fixed rollout)",
        "pooled_embedding_max_abs_diff": d_emb,
        "logits_max_abs_diff": d_log,
        "softmax_max_abs_diff": d_prob,
        "predicted_class_agreement": f"{n_cls}/{n}",
        "risk_tier_agreement": f"{n_tier}/{n}",
        "verdict": ("IDENTICAL" if (d_emb == 0.0 and n_cls == n and n_tier == n)
                    else "DIFFERS - BLOCKS DEPLOYMENT"),
    }
    if s_f32 is not None:
        res["quantisation_cost_fp32_vs_int8"] = {
            "note": ("Pre-existing property of int8 dynamic quantisation, NOT "
                     "introduced by the rollout fix: old and new int8 tokens are "
                     "bit-identical, so the deployed model carries the same cost."),
            "pooled_embedding_max_abs_diff": q_emb,
            "softmax_max_abs_diff": q_prob,
            "predicted_class_agreement": f"{q_cls}/{n}",
            "risk_tier_agreement": f"{q_tier}/{n}",
        }

    OUT.mkdir(exist_ok=True)
    (OUT / "deploy_parity.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"\n{'=' * 68}")
    print(f"  TASK 1 - CLASSIFICATION PARITY   (all {n} held-out test images)")
    print(f"{'-' * 68}")
    print(f"  embedding   max |diff|      {d_emb:.3e}")
    print(f"  logits      max |diff|      {d_log:.3e}")
    print(f"  softmax     max |diff|      {d_prob:.3e}")
    print(f"  predicted class agreement   {n_cls}/{n}  ({n_cls / n:.1%})")
    print(f"  risk tier agreement         {n_tier}/{n}  ({n_tier / n:.1%})")
    print(f"{'-' * 68}")
    print(f"  VERDICT: {res['verdict']}")
    if s_f32 is not None:
        q = res["quantisation_cost_fp32_vs_int8"]
        print(f"\n  fp32 vs int8 (pre-existing quantisation cost):")
        print(f"    embedding drift {q_emb:.5f} | softmax drift {q_prob:.5f}")
        print(f"    class agreement {q['predicted_class_agreement']} | "
              f"tier agreement {q['risk_tier_agreement']}")
    print(f"{'=' * 68}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
