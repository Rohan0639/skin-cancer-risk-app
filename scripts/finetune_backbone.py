"""Fine-tune the last K blocks of PanDerm on our data, instead of freezing it.

Everything so far trained a small head on *frozen* PanDerm embeddings. The
backbone has therefore never seen a single image from our dataset: it supplies
generic dermatology features and the head does what it can with them. That is
the published "linear probe" setting, and it is why six-class top-1 sits at
67% while macro ROC-AUC is 0.937 -- the features separate the classes, but
nothing has ever adjusted them to *these* classes.

Unfreezing the whole 86M-parameter trunk on 3,104 images would overfit, so
this trains only the last K transformer blocks plus the final norm, at a much
lower learning rate than the head (discriminative fine-tuning). Augmentation
is on, because the trunk can now memorise.

Run::

    python scripts/finetune_backbone.py --blocks 4 --epochs 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backbone import build_backbone  # noqa: E402
from src.config import Config, load_config  # noqa: E402
from src.dataset import (  # noqa: E402
    TARGET_COLUMN,
    build_metadata_matrix,
    diagnosis_classes,
    group_aware_split,
    load_unified_csv,
)
from src.model.fusion import FusionHead  # noqa: E402
from src.model.train import compute_class_weights, macro_f1  # noqa: E402
from src.utils import (  # noqa: E402
    ensure_dir,
    get_logger,
    load_image_rgb,
    resolve_device,
    set_seed,
)

logger = get_logger("finetune")


class LesionDataset(Dataset):
    """Images + metadata + label, with augmentation on the training split."""

    def __init__(self, df, cfg, transform, classes):
        self.paths = [Config.resolve(p) for p in df["image_path"]]
        self.meta = build_metadata_matrix(df, cfg).astype(np.float32)
        c2i = {c: i for i, c in enumerate(classes)}
        self.y = np.array([c2i[v] for v in df[TARGET_COLUMN]], dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.fromarray(load_image_rgb(self.paths[i]))
        return self.transform(img), torch.from_numpy(self.meta[i]), int(self.y[i])


class EndToEnd(nn.Module):
    """Backbone trunk + fusion head, trained together."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, img, meta):
        return self.head(self.backbone(img), meta)


def _block_sequence(model):
    """Return the backbone's blocks in depth order, shallowest first.

    A ViT exposes ``.blocks`` directly. ConvNeXt nests its blocks inside
    ``.stages``, so those are flattened into one ordered list -- which lets the
    same budget-matched unfreezing apply to either architecture.
    """
    blocks = getattr(model, "blocks", None)
    if blocks is not None:
        return list(blocks)
    stages = getattr(model, "stages", None)
    if stages is not None:
        out = []
        for stage in stages:
            inner = getattr(stage, "blocks", None)
            out.extend(list(inner) if inner is not None else [stage])
        return out
    raise RuntimeError("Backbone exposes neither .blocks nor .stages to unfreeze.")


def unfreeze_last_blocks(backbone, k: int, budget: float = 0.0) -> int:
    """Freeze everything, then re-enable the deepest blocks and the final norm.

    With ``budget`` (in millions of parameters) the block count is chosen to hit
    that many trainable parameters instead of using ``k``. Comparing two
    architectures fairly means matching what each is allowed to learn, and
    "4 blocks" means very different things in a ViT and a ConvNeXt.
    """
    for p in backbone.parameters():
        p.requires_grad = False
    model = backbone.model
    blocks = _block_sequence(model)

    if budget > 0:
        total, k = 0.0, 0
        for blk in reversed(blocks):
            total += sum(p.numel() for p in blk.parameters()) / 1e6
            k += 1
            if total >= budget:
                break
    k = max(0, min(k, len(blocks)))

    for blk in blocks[len(blocks) - k:]:
        for p in blk.parameters():
            p.requires_grad = True
    for attr in ("norm", "fc_norm", "norm_pre", "head_norm"):
        mod = getattr(model, attr, None)
        if isinstance(mod, nn.Module):
            for p in mod.parameters():
                p.requires_grad = True

    n = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    logger.info(
        "Unfroze last %d/%d blocks -> %.1fM trainable backbone params",
        k, len(blocks), n / 1e6,
    )
    return n


@torch.no_grad()
def evaluate(model, loader, device, n_classes, amp=False):
    """Top-1, top-3, balanced accuracy and macro-F1 over a loader."""
    model.eval()
    probs, ys = [], []
    for img, meta, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(img.to(device), meta.to(device))
        probs.append(torch.softmax(out.float(), 1).cpu().numpy())
        ys.append(y.numpy())
    P, y = np.concatenate(probs), np.concatenate(ys)
    pred = P.argmax(1)
    order = np.argsort(-P, 1)
    return {
        "top1": float((pred == y).mean()),
        "top3": float(np.mean([y[i] in order[i, :3] for i in range(len(y))])),
        "macro_f1": float(macro_f1(y, pred, n_classes)),
        "balanced": float(np.mean([
            (pred[y == c] == c).mean() for c in range(n_classes) if (y == c).sum()
        ])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blocks", type=int, default=4,
                    help="how many final transformer blocks to unfreeze")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader worker processes (4 is sensible on a GPU)")
    ap.add_argument("--backend", choices=("panderm", "fallback", "auto"), default=None,
                    help="override backbone.backend, to compare two backbones")
    ap.add_argument("--match-params", type=float, default=0.0,
                    help="unfreeze enough blocks to reach this many M trainable "
                         "params, instead of --blocks (for a fair cross-model test)")
    ap.add_argument("--tag", default="",
                    help="suffix for output files, so runs do not overwrite")
    ap.add_argument("--no-amp", action="store_true",
                    help="disable mixed precision (CUDA only; it is on by default)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(0)

    cfg = load_config()
    if args.backend:
        cfg._data["backbone"]["backend"] = args.backend
    device = resolve_device(cfg.get("project.device", "auto"))
    classes = diagnosis_classes(cfg)
    n_classes = len(classes)

    df = load_unified_csv(cfg)
    tr, va, te = group_aware_split(df, cfg)
    logger.info("split train=%d val=%d test=%d", len(tr), len(va), len(te))

    backbone = build_backbone(cfg).to(device)
    n_trainable = unfreeze_last_blocks(backbone, args.blocks, args.match_params)

    head = FusionHead(
        feature_dim=backbone.feature_dim,
        metadata_dim=build_metadata_matrix(tr, cfg).shape[1],
        num_classes=n_classes,
        image_proj_dim=int(cfg.get("model.image_proj_dim")),
        metadata_embed_dim=32,
        hidden_dims=list(cfg.get("model.hidden_dims")),
        dropout=float(cfg.get("model.dropout")),
        use_metadata=True,
    ).to(device)
    model = EndToEnd(backbone, head).to(device)

    train_tf = backbone.build_transform(train=True)
    eval_tf = backbone.build_transform(train=False)

    pin = device.type == "cuda"

    def loader(frame, tf, shuffle):
        return DataLoader(
            LesionDataset(frame, cfg, tf, classes),
            batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.workers, pin_memory=pin,
            persistent_workers=args.workers > 0,
        )

    dl_tr = loader(tr, train_tf, True)
    dl_va = loader(va, eval_tf, False)
    dl_te = loader(te, eval_tf, False)

    crit = nn.CrossEntropyLoss(
        weight=compute_class_weights(tr, classes).to(device),
        label_smoothing=float(cfg.get("train.label_smoothing", 0.0)),
    )
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in backbone.parameters() if p.requires_grad],
             "lr": args.backbone_lr},
            {"params": head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=float(cfg.get("train.weight_decay", 0.01)),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = cfg.path("paths.checkpoint_dir")
    ensure_dir(out_dir)
    tag = args.tag or ("" if not args.backend else f"_{args.backend}")
    ckpt = out_dir / f"panderm_finetuned{tag}.pt"
    history, best, stale = [], -1.0, 0

    # Mixed precision roughly halves VRAM and speeds up the matmul-heavy
    # transformer; it is a CUDA-only win, so it stays off on CPU.
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    trainable = [p for p in model.parameters() if p.requires_grad]

    if device.type == "cuda":
        logger.info("GPU: %s (%.1f GB VRAM), mixed precision %s",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9,
                    "on" if use_amp else "off")
    logger.info(
        "Training %.1fM backbone + %.1fM head params on %s",
        n_trainable / 1e6,
        sum(p.numel() for p in head.parameters()) / 1e6,
        device,
    )

    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for bi, (img, meta, y) in enumerate(dl_tr, 1):
            img, meta, y = img.to(device), meta.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp):
                loss = crit(model(img, meta), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)  # unscale before clipping, or the norm is wrong
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * y.size(0)
            seen += y.size(0)
            if bi % 20 == 0:
                logger.info("  ep%d %d/%d loss %.4f (%.1f img/s)", ep, seen, len(tr),
                            tot / seen, seen / max(time.time() - t0, 1e-6))
        sched.step()

        vm = evaluate(model, dl_va, device, n_classes, amp=use_amp)
        history.append({"epoch": ep, "train_loss": tot / max(seen, 1), **vm})
        logger.info(
            "epoch %d: train loss %.4f | val top1 %.1f%% macroF1 %.4f (%.1f min)",
            ep, tot / max(seen, 1), vm["top1"] * 100, vm["macro_f1"],
            (time.time() - t0) / 60,
        )

        if vm["macro_f1"] > best:
            best, stale = vm["macro_f1"], 0
            torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                        "epoch": ep, "val_macro_f1": best, "blocks": args.blocks}, ckpt)
            logger.info("  saved new best (val macroF1 %.4f)", best)
        else:
            stale += 1
            if stale >= args.patience:
                logger.info("early stop: no val gain for %d epochs", args.patience)
                break

    sd = torch.load(ckpt, map_location=device, weights_only=False)
    backbone.load_state_dict(sd["backbone"])
    head.load_state_dict(sd["head"])
    test = evaluate(model, dl_te, device, n_classes, amp=use_amp)

    # Measured on this same test split with the frozen backbone.
    base = {"top1": 0.673, "top3": 0.974, "balanced": 0.663, "macro_f1": 0.652}
    print()
    print("=" * 68)
    print(f"  FINE-TUNED {backbone.name}: {n_trainable/1e6:.1f}M params "
          f"(best epoch {sd['epoch']})")
    print("-" * 68)
    print(f"  {'metric':14s} {'frozen':>10s} {'fine-tuned':>12s} {'change':>10s}")
    for k in ("top1", "top3", "balanced", "macro_f1"):
        print(f"  {k:14s} {base[k]:9.3f} {test[k]:12.3f} {test[k] - base[k]:+10.3f}")
    print("=" * 68)

    res = {"frozen_baseline": base, "finetuned": test, "best_epoch": sd["epoch"],
           "blocks_unfrozen": args.blocks, "backbone": backbone.name,
           "trainable_params_m": round(n_trainable / 1e6, 2), "history": history}
    Path(f"outputs/finetune_results{tag}.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(f"  wrote outputs/finetune_results{tag}.json and {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
