"""Confirm the rollout fix on images that were NOT used to choose it.

The variant sweep picked mean-fusion/block-3 by scoring eight configurations
on a 200-image sample. Reporting that same 0.305 as the result would be
reporting a selection score, not a validation: with eight candidates the best
one is partly lucky.

So this excludes every image used during selection (the n=300 localization run
and the n=200 variant sweep, both random_state=0) and scores the old and new
configurations on a disjoint sample drawn from what remains.

Three metrics together -- Pointing Game, IoU and Dice -- because focus ratio
alone is what chose the broken configuration in the first place.

Attention is not causal attribution: these numbers bound where the CLS token's
attention lands, not what drove the prediction.
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

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from audit_localization import load_mask, score  # noqa: E402
from audit_rollout_variants import rollout, to_heat  # noqa: E402
from src.config import load_config  # noqa: E402
from src.model.risk_classifier import load_risk_classifier  # noqa: E402
from src.utils import get_logger, load_image_rgb  # noqa: E402

logger = get_logger("validate")
ANN = Path("data/raw/ham10000_dn/ds/ann")
OUT = Path("audit_reports")
N_VAL = 400

OLD = {"name": "old: max-fuse, block 6", "start": 6, "fuse": "max"}
NEW = {"name": "new: mean-fuse, block 3", "start": 3, "fuse": "mean"}


def main() -> int:
    cfg = load_config()
    model = load_risk_classifier(cfg=cfg, strict=True).eval()
    dev = next(model.head.parameters()).device
    vit = model.backbone.model
    tf = model.backbone.build_transform(train=False)

    df = pd.read_csv("data/processed/ham10000.csv")
    df["stem"] = [Path(str(p)).stem for p in df.image_path]
    masked = df[[(ANN / f"{s}.jpg.json").exists() for s in df.stem]]
    logger.info("%d images carry an expert mask", len(masked))

    # Exactly the rows the selection runs consumed.
    used = set(masked.sample(n=min(300, len(masked)), random_state=0).stem)
    used |= set(masked.sample(n=min(200, len(masked)), random_state=0).stem)
    pool = masked[~masked.stem.isin(used)]
    logger.info("%d used during selection; %d remain for validation",
                len(used), len(pool))

    val = pool.sample(n=min(N_VAL, len(pool)), random_state=99)
    rng = np.random.RandomState(0)
    rows = []
    for _, r in val.iterrows():
        rgb = load_image_rgb(cfg.resolve(r.image_path))
        mask = load_mask(r.stem, rgb.shape[:2])
        if mask is None:
            continue
        x = tf(Image.fromarray(rgb)).unsqueeze(0).to(dev)
        rec = {"image_id": r.stem, "diagnosis": r.diagnosis}
        for tag, v in (("old", OLD), ("new", NEW)):
            with torch.no_grad():
                raw = rollout(vit, x, v["start"], v["fuse"])[0].float().cpu().numpy()
            s = score(to_heat(raw, rgb.shape[:2]), mask)
            for k in ("pointing_hit", "iou", "dice", "focus_ratio"):
                rec[f"{tag}_{k}"] = s[k]
        rec["lesion_area_fraction"] = float(mask.mean())
        sc = score(rng.rand(*mask.shape), mask)
        for k in ("pointing_hit", "iou", "dice"):
            rec[f"rand_{k}"] = sc[k]
        rows.append(rec)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "rollout_validation_per_image.csv", index=False)

    def agg(tag):
        return {
            "pointing_game": float(d[f"{tag}_pointing_hit"].mean()),
            "iou": float(d[f"{tag}_iou"].mean()),
            "dice": float(d[f"{tag}_dice"].mean()),
            "focus_ratio": float(d[f"{tag}_focus_ratio"].mean()),
            "frac_above_chance_focus": float((d[f"{tag}_focus_ratio"] > 1.0).mean()),
        }

    old, new = agg("old"), agg("new")
    rand = {"pointing_game": float(d.rand_pointing_hit.mean()),
            "iou": float(d.rand_iou.mean()), "dice": float(d.rand_dice.mean()),
            "focus_ratio": 1.0, "frac_above_chance_focus": None}

    # Paired bootstrap on the new-minus-old difference.
    diffs = {}
    for k, col in (("pointing_game", "pointing_hit"), ("iou", "iou"), ("dice", "dice")):
        delta = d[f"new_{col}"].astype(float).values - d[f"old_{col}"].astype(float).values
        boot = [delta[rng.randint(0, len(delta), len(delta))].mean() for _ in range(2000)]
        diffs[k] = {"mean_delta": float(delta.mean()),
                    "ci95_low": float(np.percentile(boot, 2.5)),
                    "ci95_high": float(np.percentile(boot, 97.5)),
                    "frac_bootstraps_favouring_new": float(np.mean(np.array(boot) > 0))}

    res = {
        "validation_design": (
            f"{len(d)} dermoscopic images with expert masks, disjoint from the "
            f"{len(used)} images used to select the configuration."),
        "n_images": int(len(d)),
        "selection_set_excluded": int(len(used)),
        "mean_lesion_area_fraction": float(d.lesion_area_fraction.mean()),
        "old_config": {"start_layer": 6, "fusion": "max", **old},
        "new_config": {"start_layer": 3, "fusion": "mean", **new},
        "random_baseline": rand,
        "paired_bootstrap_new_minus_old": diffs,
        "caveats": [
            "Masks are dermoscopic HAM10000; the deployed classifier trains on "
            "clinical photographs. The rollout comes from the frozen trunk, which "
            "is shared, so this measures backbone attention, not classifier accuracy.",
            "Attention rollout is not causal attribution. High attention does not "
            "establish that a region drove the prediction.",
            "Pointing Game's random baseline equals mean lesion area, so it is only "
            "meaningful read against that number.",
        ],
    }
    (OUT / "updated_localization_metrics.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")

    # ---- figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    labels = ["Pointing Game", "IoU", "Dice"]
    keys = ["pointing_game", "iou", "dice"]
    xs = np.arange(3)
    axes[0].bar(xs - 0.26, [old[k] for k in keys], 0.25, label="old (max/b6)", color="#a62a22")
    axes[0].bar(xs, [new[k] for k in keys], 0.25, label="new (mean/b3)", color="#0b6e7a")
    axes[0].bar(xs + 0.26, [rand[k] for k in keys], 0.25, label="random", color="#9aa4ad")
    axes[0].set_xticks(xs); axes[0].set_xticklabels(labels)
    axes[0].set_title(f"Localization vs expert masks (n={len(d)})")
    axes[0].legend(fontsize=8); axes[0].set_ylim(0, max(0.5, new["dice"] * 1.25))

    axes[1].hist(d.old_focus_ratio, bins=30, alpha=.6, label="old", color="#a62a22")
    axes[1].hist(d.new_focus_ratio, bins=30, alpha=.6, label="new", color="#0b6e7a")
    axes[1].axvline(1.0, color="k", ls="--", lw=1, label="chance")
    axes[1].set_xlabel("Focus ratio"); axes[1].set_ylabel("Images")
    axes[1].set_title("Focus ratio — the metric that misled"); axes[1].legend(fontsize=8)

    dd = d.new_iou.values - d.old_iou.values
    axes[2].hist(dd, bins=30, color="#0b6e7a", alpha=.85)
    axes[2].axvline(0, color="k", ls="--", lw=1)
    axes[2].set_xlabel("IoU change (new − old), per image")
    axes[2].set_ylabel("Images")
    axes[2].set_title(f"Per-image change · {np.mean(dd > 0):.0%} improved")
    fig.tight_layout(); fig.savefig(OUT / "rollout_comparison.png", dpi=140)
    plt.close(fig)

    print(f"\n{'=' * 74}")
    print(f"  VALIDATION on {len(d)} HELD-OUT masked images "
          f"({len(used)} selection images excluded)")
    print(f"{'-' * 74}")
    print(f"  {'metric':22s} {'old':>9s} {'new':>9s} {'random':>9s} {'delta':>10s}")
    for k, lbl in zip(keys, labels):
        print(f"  {lbl:22s} {old[k]:9.3f} {new[k]:9.3f} {rand[k]:9.3f} "
              f"{new[k] - old[k]:+10.3f}")
    print(f"  {'focus ratio':22s} {old['focus_ratio']:9.3f} {new['focus_ratio']:9.3f} "
          f"{1.0:9.3f} {new['focus_ratio'] - old['focus_ratio']:+10.3f}")
    print(f"  {'above-chance focus':22s} {old['frac_above_chance_focus']:8.1%} "
          f"{new['frac_above_chance_focus']:8.1%}")
    print(f"  {'mean lesion area':22s} {res['mean_lesion_area_fraction']:9.3f}"
          f"   (= random Pointing baseline)")
    print(f"{'-' * 74}")
    for k in keys:
        b = diffs[k]
        print(f"  {k:22s} delta {b['mean_delta']:+.3f}  "
              f"95% CI [{b['ci95_low']:+.3f}, {b['ci95_high']:+.3f}]  "
              f"favours new in {b['frac_bootstraps_favouring_new']:.1%}")
    print(f"{'=' * 74}")
    print(f"  wrote updated_localization_metrics.json, rollout_comparison.png,")
    print(f"        rollout_validation_per_image.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
