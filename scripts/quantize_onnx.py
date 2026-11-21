"""Quantise the exported ONNX trunk for browser delivery, and measure the cost.

A 327 MB fp32 graph is too heavy to download into a browser, so this produces
an int8 dynamically-quantised copy and reports, on real images, how far its
predictions drift from the fp32 original. Size is only worth trading for if
the accuracy cost is measured, not assumed.

**Quantise MatMul only.** Letting the quantiser touch the patch-embedding Conv
produces a ``ConvInteger`` node, and ONNX Runtime Web has no kernel for it: the
model loads in the browser and then fails at inference. It also costs real
accuracy -- measured at 0.18 max absolute drift on the pooled embedding, which
moves predictions -- and it destroys the property that ``patch_embed`` stays
bit-identical to PanDerm's released weights.

An earlier version of this script used ``nodes_to_exclude`` on head nodes
instead, which did not reproduce the deployed artifact: it emitted ConvInteger
and drifted tokens by 7.2. The guard in :func:`assert_web_compatible` exists so
that regression cannot ship silently.

Usage::

    python scripts/quantize_onnx.py --model web/model/panderm_vit.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config  # noqa: E402
from src.utils import get_logger, load_image_rgb  # noqa: E402

logger = get_logger("quantize_onnx")

#: Ops ONNX Runtime Web cannot execute. Emitting one of these means a model
#: that loads fine and then throws on the first inference in the browser.
UNSUPPORTED_WEB_OPS = {"ConvInteger"}


def preprocess(rgb: np.ndarray, cfg: Config) -> np.ndarray:
    """Resize to the model's input size and normalise, returning (1,3,H,W)."""
    from PIL import Image

    size = int(cfg.get("backbone.image_size", 224))
    mean = np.array(cfg.get("backbone.normalize_mean"), dtype=np.float32)
    std = np.array(cfg.get("backbone.normalize_std"), dtype=np.float32)

    img = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


def assert_web_compatible(path: Path) -> dict:
    """Raise if the graph contains an op ONNX Runtime Web cannot run."""
    import collections
    import onnx

    counts = collections.Counter(
        n.op_type for n in onnx.load(str(path), load_external_data=False).graph.node)
    bad = {op: counts[op] for op in UNSUPPORTED_WEB_OPS if counts.get(op)}
    if bad:
        raise RuntimeError(
            f"{path.name} contains ops ONNX Runtime Web cannot execute: {bad}. "
            "This model would fail at inference in the browser. Check that "
            "op_types_to_quantize is restricted to MatMul.")
    if not counts.get("Conv"):
        raise RuntimeError(
            f"{path.name} has no fp32 Conv node. The patch embedding must stay "
            "in fp32 to remain bit-identical to PanDerm's released weights.")
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="web/model/panderm_vit.onnx",
                        help="fp32 trunk exported by scripts/export_onnx.py")
    parser.add_argument("--n", type=int, default=24, help="Images to compare on.")
    args = parser.parse_args()

    import onnx
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from src.dataset import (build_metadata_matrix, diagnosis_classes,
                             group_aware_split, load_unified_csv, risk_for)
    from src.model.risk_classifier import load_risk_classifier

    cfg = load_config()
    src = Path(args.model)
    if not src.exists():
        logger.error("%s not found. Run scripts/export_onnx.py first.", src)
        return 1
    dst = src.with_name(src.stem + "_int8.onnx")

    logger.info("Quantising %s -> %s (MatMul only) ...", src.name, dst.name)
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        # MatMul only: see the module docstring. Conv must stay fp32.
        op_types_to_quantize=["MatMul"],
        extra_options={
            "MatMulConstBOnly": True,
            # Shape inference cannot type every intermediate in this graph;
            # without this the MatMul quantizer aborts.
            "DefaultTensorType": onnx.TensorProto.FLOAT,
        },
    )
    counts = assert_web_compatible(dst)
    mb_src = src.stat().st_size / 1048576
    mb_dst = dst.stat().st_size / 1048576
    logger.info("fp32 %.0f MB -> int8 %.0f MB (%.1fx smaller)",
                mb_src, mb_dst, mb_src / mb_dst)
    logger.info("ops: Conv(fp32)=%d  MatMulInteger=%d  MatMul=%d  ConvInteger=%d",
                counts.get("Conv", 0), counts.get("MatMulInteger", 0),
                counts.get("MatMul", 0), counts.get("ConvInteger", 0))

    # ---- cost of quantisation, through the real head -----------------------
    model = load_risk_classifier(cfg=cfg, strict=True).eval()
    classes = diagnosis_classes(cfg)
    df = load_unified_csv(cfg)
    _, _, te = group_aware_split(df, cfg)
    sel = te.sample(n=min(args.n, len(te)), random_state=0)
    meta = build_metadata_matrix(sel, cfg).astype(np.float32)

    s32 = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    head = model.head.cpu().eval()

    agree = 0
    tier_agree = 0
    emb_drift = 0.0
    prob_drift = 0.0
    for i, row in enumerate(sel.to_dict("records")):
        image = preprocess(load_image_rgb(Config.resolve(row["image_path"])), cfg)
        t32 = s32.run(None, {"image": image})[0]
        t8 = s8.run(None, {"image": image})[0]
        e32, e8 = t32.mean(axis=1), t8.mean(axis=1)
        emb_drift = max(emb_drift, float(np.abs(e32 - e8).max()))

        m = torch.from_numpy(meta[i])[None]
        with torch.no_grad():
            p32 = torch.softmax(head(torch.from_numpy(e32), m), 1)[0].numpy()
            p8 = torch.softmax(head(torch.from_numpy(e8), m), 1)[0].numpy()
        prob_drift = max(prob_drift, float(np.abs(p32 - p8).max()))
        c32, c8 = classes[int(p32.argmax())], classes[int(p8.argmax())]
        agree += int(c32 == c8)
        tier_agree += int(risk_for(cfg, c32) == risk_for(cfg, c8))

    n = len(sel)
    print()
    print("=" * 62)
    print(f"  fp32 : {mb_src:6.0f} MB")
    print(f"  int8 : {mb_dst:6.0f} MB   ({mb_src / mb_dst:.1f}x smaller)")
    print(f"  ONNX Runtime Web compatible : yes (no ConvInteger, Conv stays fp32)")
    print(f"  predicted class agreement   : {agree}/{n}  ({agree / n:.1%})")
    print(f"  risk tier agreement         : {tier_agree}/{n}  ({tier_agree / n:.1%})")
    print(f"  max embedding drift         : {emb_drift:.5f}")
    print(f"  max probability drift       : {prob_drift:.5f}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
