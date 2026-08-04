"""Train and evaluate against the OpenEDS 2019 segmentation benchmark.

Published target: RITnet, **95.3% mIoU** with 248,900 parameters (0.98 MB),
winner of the 2019 challenge (Chaudhary et al., arXiv:1910.00694). The
challenge scored mIoU + min(1/S, 1) with S in MB, so a smaller model at equal
accuracy scores higher.

Test labels are held out by the challenge, so the reproducible comparison is
the 2,403-image validation split.

    python -m benchmarks.bench_openeds --epochs 60
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from visionscreen.ml.openeds import (
    RITNET_MIOU,
    RITNET_PARAMS,
    OpenEDSDataset,
    evaluate_miou,
)
from visionscreen.ml.segnet_dense import CombinedLoss, DenseEyeNet

ZIP = Path("data/openeds/seg.zip")
CKPT = Path("models/openeds_dense.pt")
RESULTS = Path("results/openeds.json")


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run(epochs: int = 60, batch: int = 16, width: int = 28, growth: int = 10,
        size=(192, 320), lr: float = 2e-3, device: str | None = None,
        limit: int | None = None, workers: int = 4) -> dict:
    device = pick_device(device)
    train_ds = OpenEDSDataset(ZIP, "train", size=size, augment=True, limit=limit)
    val_ds = OpenEDSDataset(ZIP, "validation", size=size, augment=False,
                            limit=(limit // 4 if limit else None))
    print(f"train {len(train_ds)}  val {len(val_ds)}  device {device}", flush=True)

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=workers, persistent_workers=workers > 0,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch, num_workers=workers,
                        persistent_workers=workers > 0)

    model = DenseEyeNet(4, width=width, growth=growth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = n_params * 4 / 1e6
    print(f"model {n_params:,} params = {size_mb:.2f} MB "
          f"(RITnet {RITNET_PARAMS:,} = 0.98 MB)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(len(train_dl), 1), pct_start=0.25)
    criterion = CombinedLoss()

    best = {"mIoU": 0.0}
    history = []
    for epoch in range(epochs):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            running += loss.item()
        metrics = evaluate_miou(model, val_dl, device)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = round(running / max(len(train_dl), 1), 4)
        metrics["secs"] = round(time.time() - t0, 1)
        history.append(metrics)
        gap = metrics["mIoU"] - RITNET_MIOU
        print(f"epoch {epoch+1:3d}/{epochs}  loss {metrics['train_loss']:.4f}  "
              f"mIoU {metrics['mIoU']:.4f}  ({gap:+.4f} vs RITnet)  "
              f"pupil {metrics['pupil_iou']:.4f}  {metrics['secs']}s", flush=True)

        if metrics["mIoU"] > best["mIoU"]:
            best = dict(metrics)
            CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "metrics": best,
                        "width": width, "growth": growth, "size": size}, CKPT)

    return {
        "best": best,
        "history": history,
        "params": n_params,
        "size_mb": round(size_mb, 3),
        "ritnet": {"mIoU": RITNET_MIOU, "params": RITNET_PARAMS, "size_mb": 0.98},
        "beats_ritnet_miou": best["mIoU"] >= RITNET_MIOU,
        "smaller_than_ritnet": n_params < RITNET_PARAMS,
        "challenge_score": round(best["mIoU"] + min(1.0 / size_mb, 1.0), 4),
        "ritnet_challenge_score": 0.976,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--width", type=int, default=28)
    ap.add_argument("--growth", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    r = run(epochs=args.epochs, batch=args.batch, width=args.width,
            growth=args.growth, lr=args.lr, device=args.device,
            limit=args.limit, workers=args.workers)
    Path("results").mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(r, indent=2))

    b = r["best"]
    print("\n| model | mIoU | params | size | challenge score |")
    print("|---|---|---|---|---|")
    print(f"| RITnet (2019 winner) | {RITNET_MIOU:.4f} | {RITNET_PARAMS:,} | 0.98 MB | 0.976 |")
    print(f"| DenseEyeNet (ours)   | {b['mIoU']:.4f} | {r['params']:,} | "
          f"{r['size_mb']:.2f} MB | {r['challenge_score']:.4f} |")
    print(f"\nper class: " + ", ".join(
        f"{k.replace('_iou','')} {v}" for k, v in b.items() if k.endswith("_iou")))
    verdict = "MATCHED/BEATEN" if r["beats_ritnet_miou"] else "not yet reached"
    print(f"\nRITnet 95.3% mIoU: {verdict} ({b['mIoU']*100:.2f}%)")


if __name__ == "__main__":
    main()
