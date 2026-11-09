"""Verify the browser model: int8 ONNX trunk + hand-rolled head vs PyTorch.

Reimplements the fusion head in NumPy exactly as the JavaScript will, runs it
on top of the quantised ONNX trunk, and compares the resulting probabilities
against the full PyTorch pipeline on real images.

This is the gate for shipping the web build: it proves both that int8
quantisation is acceptable and that the JS head reimplementation is correct,
before a single line of JavaScript is written.

Usage::

    python scripts/verify_web_model.py --n 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config  # noqa: E402
from src.dataset import encode_metadata  # noqa: E402
from src.utils import get_logger, load_image_rgb  # noqa: E402

logger = get_logger("verify_web_model")


# --- the head, exactly as JS will implement it -----------------------------
def _linear(x: np.ndarray, spec: dict) -> np.ndarray:
    w = np.asarray(spec["w"], dtype=np.float32).reshape(spec["out"], spec["in"])
    b = np.asarray(spec["b"], dtype=np.float32)
    return x @ w.T + b


def _layernorm(x: np.ndarray, spec: dict) -> np.ndarray:
    g = np.asarray(spec["g"], dtype=np.float32)
    b = np.asarray(spec["b"], dtype=np.float32)
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + spec["eps"]) * g + b


def run_head(features: np.ndarray, meta: np.ndarray, head: dict) -> np.ndarray:
    """Apply the fusion head to ``(N, 768)`` features, returning ``(N, C)``."""
    x = _linear(features, head["image_proj"]["linear"])
    x = np.maximum(x, 0.0)
    x = _layernorm(x, head["image_proj"]["norm"])

    if head.get("use_metadata") and "metadata_encoder" in head:
        m = _linear(meta, head["metadata_encoder"]["linear"])
        m = np.maximum(m, 0.0)
        m = _layernorm(m, head["metadata_encoder"]["norm"])
        if m.shape[0] == 1 and x.shape[0] > 1:
            m = np.repeat(m, x.shape[0], axis=0)
        x = np.concatenate([x, m], axis=-1)

    for i, block in enumerate(head["mlp"]):
        x = _linear(x, block["linear"])
        if block.get("relu"):
            x = np.maximum(x, 0.0)
        if block.get("bn"):
            scale = np.asarray(block["bn"]["scale"], dtype=np.float32)
            shift = np.asarray(block["bn"]["shift"], dtype=np.float32)
            x = x * scale + shift
    return x


def softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def preprocess(rgb: np.ndarray, cfg: Config) -> np.ndarray:
    """Resize and normalise to ``(1, 3, H, W)`` -- mirrors the JS path."""
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
    parser.add_argument("--model-dir", default="web/model")
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()

    import onnxruntime as ort
    import pandas as pd

    cfg = load_config()
    mdir = Path(args.model_dir)
    head = json.loads((mdir / "head.json").read_text(encoding="utf-8"))
    classes = head["classes"]

    sess_fp32 = ort.InferenceSession(str(mdir / "panderm_vit.onnx"),
                                     providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(mdir / "panderm_vit_int8.onnx"),
                                     providers=["CPUExecutionProvider"])

    from src.model.risk_classifier import load_risk_classifier

    torch_model = load_risk_classifier(cfg=cfg, strict=True)

    df = pd.read_csv(cfg.path("paths.unified_csv")).sample(
        n=min(args.n, 99999), random_state=0)

    agree32 = agree8 = 0
    d32 = d8 = 0.0
    total = 0

    for row in df.to_dict("records"):
        rgb = load_image_rgb(Config.resolve(row["image_path"]))
        image = preprocess(rgb, cfg)
        meta = encode_metadata(row["age"], row["sex"], row["site"], cfg)[None]

        ref = torch_model.predict(rgb, row["age"], row["sex"], row["site"])
        ref_p = np.array([ref.diagnosis_probabilities[c] for c in classes], dtype=np.float32)

        for sess, tag in ((sess_fp32, "fp32"), (sess_int8, "int8")):
            tokens = sess.run(None, {"image": image})[0]
            cls_tok = tokens[:, 0, :]
            probs = softmax(run_head(cls_tok, meta, head))[0]
            delta = float(np.abs(probs - ref_p).max())
            if tag == "fp32":
                agree32 += int(probs.argmax() == ref_p.argmax())
                d32 = max(d32, delta)
            else:
                agree8 += int(probs.argmax() == ref_p.argmax())
                d8 = max(d8, delta)
        total += 1

    print()
    print("=" * 66)
    print(f"  compared against the PyTorch pipeline on {total} real images")
    print("-" * 66)
    print(f"  ONNX fp32 trunk + NumPy head : {agree32}/{total} agree"
          f"   max prob drift {d32:.5f}")
    print(f"  ONNX int8 trunk + NumPy head : {agree8}/{total} agree"
          f"   max prob drift {d8:.5f}")
    print("=" * 66)

    if agree32 < total:
        logger.error("fp32 path disagrees with PyTorch -- the head "
                     "reimplementation is wrong.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
