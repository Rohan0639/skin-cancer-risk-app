"""Quantise the exported ONNX model for browser delivery, and measure the cost.

A 329 MB fp32 graph is too heavy to download into a browser, so this produces
an int8 dynamically-quantised copy and reports, on real images, how far its
predictions drift from the fp32 original. Size is only worth trading for if the
accuracy cost is measured, not assumed.

Usage::

    python scripts/quantize_onnx.py --model web/model/panderm_risk.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config  # noqa: E402
from src.dataset import encode_metadata  # noqa: E402
from src.utils import get_logger, load_image_rgb  # noqa: E402

logger = get_logger("quantize_onnx")


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="web/model/panderm_risk.onnx")
    parser.add_argument("--n", type=int, default=24, help="Images to compare on.")
    args = parser.parse_args()

    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import pandas as pd

    cfg = load_config()
    src = Path(args.model)
    dst = src.with_name(src.stem + "_int8.onnx")

    # Quantise the ViT trunk only. The fusion head is a few hundred KB, so
    # there is nothing to gain from quantising it, and its Gemm nodes produce
    # a MatMulInteger graph with mismatched dimensions that fails to load.
    graph = onnx.load(str(src)).graph
    exclude = [n.name for n in graph.node if "/head/" in n.name]
    logger.info("Excluding %d head node(s) from quantisation.", len(exclude))

    logger.info("Quantising %s -> %s ...", src.name, dst.name)
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        nodes_to_exclude=exclude,
        # DefaultTensorType is required because shape inference cannot type
        # every intermediate in this graph; without it the MatMul quantizer
        # aborts on the fusion head's Gemm outputs.
        extra_options={
            "MatMulConstBOnly": True,
            "DefaultTensorType": onnx.TensorProto.FLOAT,
        },
    )
    mb_src = src.stat().st_size / 1048576
    mb_dst = dst.stat().st_size / 1048576
    logger.info("fp32 %.0f MB -> int8 %.0f MB (%.1fx smaller)", mb_src, mb_dst, mb_src / mb_dst)

    # Compare on real prepared images.
    df = pd.read_csv(cfg.path("paths.unified_csv")).sample(
        n=min(args.n, 9999), random_state=0)
    s32 = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])

    agree = 0
    max_prob_delta = 0.0
    total = 0
    for row in df.to_dict("records"):
        rgb = load_image_rgb(Config.resolve(row["image_path"]))
        image = preprocess(rgb, cfg)
        meta = encode_metadata(row["age"], row["sex"], row["site"], cfg)[None]

        l32 = s32.run(None, {"image": image, "metadata": meta})[0][0]
        l8 = s8.run(None, {"image": image, "metadata": meta})[0][0]

        def softmax(v):
            e = np.exp(v - v.max())
            return e / e.sum()

        p32, p8 = softmax(l32), softmax(l8)
        agree += int(p32.argmax() == p8.argmax())
        max_prob_delta = max(max_prob_delta, float(np.abs(p32 - p8).max()))
        total += 1

    print()
    print("=" * 60)
    print(f"  fp32 : {mb_src:6.0f} MB")
    print(f"  int8 : {mb_dst:6.0f} MB   ({mb_src / mb_dst:.1f}x smaller)")
    print(f"  predicted class agreement : {agree}/{total}  ({agree / total:.1%})")
    print(f"  max probability drift     : {max_prob_delta:.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
