"""Representative visual evidence for the audit.

gradcam_examples/     photo | deployed rollout | best variant | expert mask
misclassified_examples/  the most confident errors, with the heatmap

No patient identifiers are written: ISIC accession stems only, which are
public dataset IDs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from audit_localization import load_mask, score  # noqa: E402
from audit_rollout_variants import rollout, to_heat  # noqa: E402
from src.config import load_config  # noqa: E402
from src.dataset import (  # noqa: E402
    TARGET_COLUMN, build_metadata_matrix, diagnosis_classes,
    group_aware_split, load_unified_csv, risk_for,
)
from src.model.risk_classifier import load_risk_classifier  # noqa: E402
from src.utils import get_logger, load_image_rgb  # noqa: E402

logger = get_logger("examples")
ANN = Path("data/raw/ham10000_dn/ds/ann")
OUT = Path("audit_reports")


def main() -> int:
    cfg = load_config()
    classes = diagnosis_classes(cfg)
    c2i = {c: i for i, c in enumerate(classes)}
    model = load_risk_classifier(cfg=cfg, strict=True).eval()
    dev = next(model.head.parameters()).device
    vit = model.backbone.model
    tf = model.backbone.build_transform(train=False)

    # ---- Grad-CAM evidence: deployed vs best variant vs expert mask ---------
    gd = OUT / "gradcam_examples"
    gd.mkdir(parents=True, exist_ok=True)
    dm = pd.read_csv("data/processed/ham10000.csv")
    dm["stem"] = [Path(str(p)).stem for p in dm.image_path]
    dm = dm[[(ANN / f"{s}.jpg.json").exists() for s in dm.stem]].sample(n=6, random_state=1)

    fig, axes = plt.subplots(len(dm), 4, figsize=(13, 3.1 * len(dm)))
    for n, (_, r) in enumerate(dm.iterrows()):
        rgb = load_image_rgb(cfg.resolve(r.image_path))
        mask = load_mask(r.stem, rgb.shape[:2])
        x = tf(Image.fromarray(rgb)).unsqueeze(0).to(dev)
        with torch.no_grad():
            dep = rollout(vit, x, 6, "max")[0].float().cpu().numpy()
            bst = rollout(vit, x, 3, "mean")[0].float().cpu().numpy()
        hd = to_heat(dep, rgb.shape[:2])
        hb = to_heat(bst, rgb.shape[:2])
        sd, sb = score(hd, mask), score(hb, mask)

        axes[n, 0].imshow(rgb); axes[n, 0].set_title(f"{r.stem} · {r.diagnosis}", fontsize=8)
        axes[n, 1].imshow(rgb); axes[n, 1].imshow(hd, cmap="jet", alpha=.45)
        axes[n, 1].set_title(f"DEPLOYED max/b6 · point {'HIT' if sd['pointing_hit'] else 'MISS'}"
                             f" · IoU {sd['iou']:.2f}", fontsize=8)
        axes[n, 2].imshow(rgb); axes[n, 2].imshow(hb, cmap="jet", alpha=.45)
        axes[n, 2].set_title(f"mean/b3 · point {'HIT' if sb['pointing_hit'] else 'MISS'}"
                             f" · IoU {sb['iou']:.2f}", fontsize=8)
        axes[n, 3].imshow(mask, cmap="gray")
        axes[n, 3].set_title("expert mask (ground truth)", fontsize=8)
        for a in axes[n]:
            a.axis("off")
    fig.suptitle("Attention localization vs expert masks — deployed config vs best variant",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(gd / "rollout_vs_mask.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", gd / "rollout_vs_mask.png")

    # ---- most confident misclassifications ----------------------------------
    md = OUT / "misclassified_examples"
    md.mkdir(parents=True, exist_ok=True)
    df = load_unified_csv(cfg)
    tr, va, te = group_aware_split(df, cfg)
    M = torch.from_numpy(build_metadata_matrix(te, cfg).astype(np.float32)).to(dev)
    X = torch.stack([tf(Image.fromarray(load_image_rgb(cfg.resolve(p)))) for p in te.image_path])
    P = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            f = model.backbone(X[i:i + 32].to(dev)).float()
            P.append(torch.softmax(model.head(f, M[i:i + 32]), 1).cpu().numpy())
    P = np.concatenate(P)
    y = np.array([c2i[v] for v in te[TARGET_COLUMN]])
    pred, conf = P.argmax(1), P.max(1)

    # Dangerous downgrades first: truly High risk, confidently shown as Low.
    danger = [i for i in range(len(y))
              if pred[i] != y[i]
              and risk_for(cfg, classes[y[i]]) == "High"
              and risk_for(cfg, classes[pred[i]]) == "Low"]
    danger.sort(key=lambda i: -conf[i])
    pick = danger[:6]
    logger.info("%d dangerous downgrades; plotting the %d most confident",
                len(danger), len(pick))

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
    for ax, i in zip(axes.ravel(), pick):
        row = te.iloc[i]
        rgb = load_image_rgb(cfg.resolve(row.image_path))
        ax.imshow(rgb); ax.axis("off")
        ax.set_title(f"{Path(str(row.image_path)).stem}\n"
                     f"true {classes[y[i]]} (High) → said {classes[pred[i]]} (Low)\n"
                     f"confidence {conf[i]:.0%} · true-class prob {P[i, y[i]]:.1%}",
                     fontsize=8)
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle("Dangerous downgrades: High-risk lesions reported as Low risk", fontsize=11)
    fig.tight_layout()
    fig.savefig(md / "dangerous_downgrades.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", md / "dangerous_downgrades.png")

    print(f"\n  gradcam_examples/rollout_vs_mask.png")
    print(f"  misclassified_examples/dangerous_downgrades.png")
    print(f"  dangerous downgrades total: {len(danger)} / {len(y)} "
          f"({len(danger) / len(y):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
