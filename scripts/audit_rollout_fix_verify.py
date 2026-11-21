"""Quantize the re-exported trunk and prove the classifier did not move.

The rollout fix changes only a side branch: `x = blk(...)` is untouched, so
tokens -- and therefore every embedding, logit and prediction -- should be
unchanged. "Should be" is not evidence, so this runs the OLD deployed int8
model and the NEW int8 model over real images and compares them directly.

Writes the int8 model next to the fp32 one and prints the comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config  # noqa: E402
from src.dataset import group_aware_split, load_unified_csv  # noqa: E402
from src.utils import get_logger, load_image_rgb  # noqa: E402

logger = get_logger("rollout_verify")
OLD = Path("web/model/panderm_vit_int8.onnx")


def preprocess(rgb, cfg):
    """Match the deployed pipeline exactly: 224x224, ImageNet normalisation."""
    from PIL import Image
    size = int(cfg.get("backbone.image_size", 224))
    im = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    mean = np.array(cfg.get("backbone.normalize_mean"), dtype=np.float32)
    std = np.array(cfg.get("backbone.normalize_std"), dtype=np.float32)
    a = (a - mean) / std
    return a.transpose(2, 0, 1)[None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="newly exported fp32 trunk")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src = Path(args.model)
    dst = src.with_name(src.stem + "_int8.onnx")

    # MatMul ONLY, which is what the deployed artifact was built with. Letting
    # the quantiser touch the patch_embed Conv produces a ConvInteger node, and
    # ONNX Runtime Web has no kernel for it -- the model would load and then
    # fail at inference in the browser. It also measurably degrades the
    # embedding (0.18 max abs drift), which moves predictions.
    # Note: scripts/quantize_onnx.py uses nodes_to_exclude instead and does NOT
    # reproduce the deployed file. See the fix report.
    logger.info("quantising %s (MatMul only) ...", src.name)
    quantize_dynamic(
        model_input=str(src), model_output=str(dst),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
        extra_options={"MatMulConstBOnly": True,
                       "DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    mb_src, mb_dst = src.stat().st_size / 1048576, dst.stat().st_size / 1048576
    logger.info("fp32 %.0f MB -> int8 %.0f MB", mb_src, mb_dst)

    if not OLD.exists():
        logger.error("deployed model %s not found; cannot compare", OLD)
        return 1

    cfg = load_config()
    df = load_unified_csv(cfg)
    _, _, te = group_aware_split(df, cfg)
    sel = te.sample(n=min(args.n, len(te)), random_state=7)

    s_old = ort.InferenceSession(str(OLD), providers=["CPUExecutionProvider"])
    s_new = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])

    tok_d, roll_d, emb_d = [], [], []
    for p in sel.image_path:
        img = preprocess(load_image_rgb(Config.resolve(p)), cfg)
        to, ro = s_old.run(None, {"image": img})
        tn, rn = s_new.run(None, {"image": img})
        tok_d.append(float(np.abs(to - tn).max()))
        emb_d.append(float(np.abs(to.mean(axis=1) - tn.mean(axis=1)).max()))
        # Rollout is compared after the same percentile normalisation the app
        # applies, since that is the form the user actually sees.
        def norm(v):
            v = v[0]
            lo, hi = np.percentile(v, 50), np.percentile(v, 99)
            return np.clip((v - lo) / max(hi - lo, 1e-8), 0, 1) ** 1.2
        roll_d.append(float(np.abs(norm(ro) - norm(rn)).max()))

    res = {
        "n_images": len(tok_d),
        "fp32_mb": round(mb_src), "int8_mb": round(mb_dst),
        "tokens_max_abs_diff": max(tok_d),
        "tokens_mean_abs_diff": float(np.mean(tok_d)),
        "pooled_embedding_max_abs_diff": max(emb_d),
        "rollout_max_abs_diff_normalised": max(roll_d),
        "rollout_mean_abs_diff_normalised": float(np.mean(roll_d)),
        "interpretation": (
            "tokens/embedding diffs at 1e-5 or below mean the classification "
            "path is numerically unchanged; the rollout diff should be LARGE, "
            "since that is the branch that was fixed."),
    }
    print(f"\n{'=' * 66}")
    print(f"  OLD deployed int8  vs  NEW int8   (n={len(tok_d)} test images)")
    print(f"{'-' * 66}")
    print(f"  tokens      max |diff|  {res['tokens_max_abs_diff']:.3e}   "
          f"mean {res['tokens_mean_abs_diff']:.3e}")
    print(f"  embedding   max |diff|  {res['pooled_embedding_max_abs_diff']:.3e}")
    print(f"  rollout     max |diff|  {res['rollout_max_abs_diff_normalised']:.3e}   "
          f"mean {res['rollout_mean_abs_diff_normalised']:.3e}  (expected large)")
    print(f"{'=' * 66}")
    verdict = ("UNCHANGED" if res["tokens_max_abs_diff"] < 1e-4 else "CHANGED - INVESTIGATE")
    print(f"  classification path: {verdict}")
    print(f"  wrote {dst.name} ({mb_dst:.0f} MB)")

    Path("audit_reports").mkdir(exist_ok=True)
    Path("audit_reports/rollout_parity.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
