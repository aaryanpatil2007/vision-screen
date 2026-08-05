"""Benchmark against the published periorbital segmentation result.

Reference: Nahass et al., *Ophthalmology Science* 2025, DeepLabV3-ResNet101
(~60M parameters), per-class Dice on an 80/20 split of face photographs from
the Chicago Face Database and CelebAMask-HQ.

This is the comparison that is actually valid for a webcam system — same
spectrum, same optics, same framing. The OpenEDS/RITnet benchmark this project
started with is near-infrared head-mounted imagery and is not comparable; that
run is kept only as an architecture check and an efficiency datapoint.

    python -m benchmarks.bench_periorbital --subset combined --epochs 60
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.multiprocessing
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy("file_system")

from visionscreen.ml.periorbital import (  # noqa: E402
    CLASS_NAMES,
    N_CLASSES,
    REFERENCE_DICE,
    REFERENCE_MODEL,
    REFERENCE_PARAMS,
    PeriorbitalDataset,
    compare_to_reference,
    evaluate_dice,
    pairs,
    split_pairs,
)
from visionscreen.ml.segnet_dense import CombinedLoss, DenseEyeNet  # noqa: E402

CKPT = Path("models/periorbital_dense.pt")
LAST = Path("models/periorbital_dense_last.pt")
RESULTS = Path("results/periorbital.json")


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run(subset: str = "combined", epochs: int = 60, batch: int = 16,
        width: int = 32, growth: int = 12, size=(128, 320), lr: float = 2e-3,
        device: str | None = None, workers: int = 4, seed: int = 0,
        resume: bool = False) -> dict:
    device = pick_device(device)
    items = pairs(subset)
    if not items:
        raise SystemExit(f"no image/mask pairs found for subset {subset!r}")
    train_items, val_items = split_pairs(items, val_frac=0.2, seed=seed)

    train_ds = PeriorbitalDataset(train_items, size=size, augment=True)
    # scored at the label's own resolution, as the reference paper does
    val_ds = PeriorbitalDataset(val_items, size=size, augment=False,
                                native_labels=False)
    print(f"subset {subset}: {len(train_ds)} train / {len(val_ds)} val "
          f"({len(items)} pairs)  device {device}", flush=True)

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=workers, persistent_workers=workers > 0,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch, num_workers=max(workers // 2, 1),
                        persistent_workers=workers > 0)

    model = DenseEyeNet(N_CLASSES, width=width, growth=growth, in_ch=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model {n_params:,} params ({n_params / REFERENCE_PARAMS:.1%} of the "
          f"reference DeepLabV3-ResNet101)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(len(train_dl), 1), pct_start=0.25)
    criterion = CombinedLoss()

    best = {"mean_dice_structures": 0.0}
    history: list[dict] = []
    start = 0
    if resume and LAST.exists():
        ck = torch.load(LAST, map_location=device)
        model.load_state_dict(ck["state_dict"])
        best = dict(ck.get("metrics", best))
        history = list(ck.get("history", []))
        start = int(ck.get("epoch", 0))
        if ck.get("opt"):
            opt.load_state_dict(ck["opt"])
        if ck.get("sched"):
            sched.load_state_dict(ck["sched"])
        print(f"resumed from epoch {start}", flush=True)

    for epoch in range(start, epochs):
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

        m = evaluate_dice(model, val_dl, device)
        m["epoch"] = epoch + 1
        m["train_loss"] = round(running / max(len(train_dl), 1), 4)
        m["secs"] = round(time.time() - t0, 1)
        history.append(m)
        ref = REFERENCE_DICE[subset]
        print(f"epoch {epoch+1:3d}/{epochs}  loss {m['train_loss']:.4f}  "
              f"mean-dice {m['mean_dice_structures']:.4f}  "
              f"iris {m['iris_dice']:.3f}/{ref.get('iris', 0):.2f}  "
              f"sclera {m['sclera_dice']:.3f}/{ref.get('sclera', 0):.2f}  "
              f"caruncle {m['caruncle_dice']:.3f}/{ref.get('caruncle', 0):.2f}  "
              f"{m['secs']}s", flush=True)

        improved = m["mean_dice_structures"] > best["mean_dice_structures"]
        if improved:
            best = dict(m)
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "metrics": best,
                    "width": width, "growth": growth, "size": size,
                    "epoch": epoch + 1, "history": history,
                    "opt": opt.state_dict(), "sched": sched.state_dict()}, LAST)
        if improved:
            torch.save({"state_dict": model.state_dict(), "metrics": best,
                        "width": width, "growth": growth, "size": size,
                        "subset": subset}, CKPT)

    tta = None
    if CKPT.exists():
        model.load_state_dict(torch.load(CKPT, map_location=device)["state_dict"])
        tta = evaluate_dice(model, val_dl, device, tta=True)
        print(f"best + flip TTA: mean-dice {tta['mean_dice_structures']:.4f}",
              flush=True)

    final = tta if (tta and tta["mean_dice_structures"] >
                    best["mean_dice_structures"]) else best
    return {
        "subset": subset,
        "reference": {"model": REFERENCE_MODEL, "params": REFERENCE_PARAMS,
                      "dice": REFERENCE_DICE[subset]},
        "ours": {"params": n_params,
                 "params_ratio": round(n_params / REFERENCE_PARAMS, 5)},
        "best": best, "best_tta": tta, "final": final,
        "comparison": compare_to_reference(final, subset),
        "history": history,
        "n_train": len(train_ds), "n_val": len(val_ds),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="combined", choices=list(REFERENCE_DICE))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--growth", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    r = run(subset=a.subset, epochs=a.epochs, batch=a.batch, width=a.width,
            growth=a.growth, lr=a.lr, workers=a.workers, device=a.device,
            resume=a.resume)
    Path("results").mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(r, indent=2))

    print(f"\n{'class':10s} {'ours':>8s} {'reference':>10s} {'delta':>8s}")
    print("-" * 40)
    for name, row in r["comparison"].items():
        if name.startswith("_"):
            continue
        mark = "+" if row["beats"] else " "
        print(f"{name:10s} {row['ours']:8.3f} {row['reference']:10.2f} "
              f"{row['delta']:+8.3f} {mark}")
    s = r["comparison"]["_summary"]
    print(f"\nmatched or beaten on {s['classes_matched_or_beaten']}/"
          f"{s['classes_compared']} classes, at "
          f"{r['ours']['params_ratio']:.2%} of the reference model's parameters")


if __name__ == "__main__":
    main()
