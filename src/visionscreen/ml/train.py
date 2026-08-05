from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from visionscreen.ml.dataset import RealEyeDataset, SyntheticEyeDataset, collate_resize
from visionscreen.ml.model import CLASS_NAMES, N_CLASSES, EyeSegNet

DEFAULT_CKPT = Path("models/eyesegnet.pt")


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def class_ious(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        p, t = pred == c, target == c
        union = (p | t).sum().item()
        out[f"{name}_iou"] = (p & t).sum().item() / union if union else float("nan")
    return out


def evaluate(model, n: int, seed: int, out_size, device: str = "cpu",
             real_root: Path | None = None) -> dict:
    ds = (
        RealEyeDataset(real_root, out_size)
        if real_root is not None
        else SyntheticEyeDataset(n=n, seed=seed, out_size=out_size)
    )
    if len(ds) == 0:
        return {}
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_resize)
    model = model.to(device).eval()
    inter = np.zeros(N_CLASSES)
    union = np.zeros(N_CLASSES)
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            pred = model(imgs).argmax(1)
            for c in range(N_CLASSES):
                p, t = pred == c, masks == c
                inter[c] += (p & t).sum().item()
                union[c] += (p | t).sum().item()
    metrics = {
        f"{name}_iou": (inter[c] / union[c]) if union[c] else float("nan")
        for c, name in enumerate(CLASS_NAMES)
    }
    valid = [v for v in metrics.values() if not np.isnan(v)]
    metrics["mean_iou"] = float(np.mean(valid)) if valid else float("nan")
    metrics["n_samples"] = len(ds)
    return metrics


def train_model(
    n_train: int = 4000,
    n_val: int = 400,
    epochs: int = 12,
    batch_size: int = 32,
    out_size: tuple[int, int] = (64, 96),
    seed: int = 0,
    device: str | None = None,
    lr: float = 3e-3,
    real_root: Path | None = None,
    log_every: int = 1,
    real_dataset: torch.utils.data.Dataset | None = None,
):
    """real_dataset takes precedence over real_root — pass an explicit TRAIN
    subset when a held-out real split exists, or the benchmark leaks."""
    device = pick_device(device)
    # `seed` reached the dataset but never torch, so weight initialisation,
    # dropout and shuffling were all unseeded: the same call trained a
    # different network every time. Invisible while metrics are averaged over
    # many samples, very visible on any single borderline case — it is what
    # made a reflex-detection test pass alone and fail under load.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_ds: torch.utils.data.Dataset = SyntheticEyeDataset(n_train, seed, out_size)
    real = real_dataset if real_dataset is not None else (
        RealEyeDataset(real_root, out_size) if real_root is not None else None
    )
    if real is not None:
        if len(real):
            # oversample scarce real data so it isn't drowned by synthetic
            reps = max(1, n_train // (4 * len(real)))
            train_ds = ConcatDataset([train_ds] + [real] * reps)
            print(f"mixing {len(real)} real crops (x{reps}) with {n_train} synthetic")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_resize
    )
    model = EyeSegNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    # rare classes (pupil, reflex) need upweighting or the net predicts background
    weights = torch.tensor([1.0, 1.0, 2.0, 4.0, 8.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    history = {"train_loss": [], "val_mean_iou": [], "val_pupil_iou": []}
    for epoch in range(epochs):
        model.train()
        losses = []
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        history["train_loss"].append(float(np.mean(losses)))
        val = evaluate(model, n_val, seed + 10_000, out_size, device)
        history["val_mean_iou"].append(val.get("mean_iou", float("nan")))
        history["val_pupil_iou"].append(val.get("pupil_iou", float("nan")))
        if log_every and epoch % log_every == 0:
            print(
                f"epoch {epoch+1}/{epochs} loss={history['train_loss'][-1]:.4f} "
                f"val_mIoU={history['val_mean_iou'][-1]:.4f} "
                f"pupil={history['val_pupil_iou'][-1]:.4f}"
            )
    return model, history


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--real-root", default=None)
    ap.add_argument("--out", default=str(DEFAULT_CKPT))
    args = ap.parse_args()

    real_root = Path(args.real_root) if args.real_root else None
    model, history = train_model(
        n_train=args.n_train, epochs=args.epochs, batch_size=args.batch_size,
        device=args.device, real_root=real_root,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "history": history}, out)
    final = evaluate(model, 500, 4242, (64, 96), pick_device(args.device))
    Path("results").mkdir(exist_ok=True)
    Path("results/segmentation.json").write_text(
        json.dumps({"history": history, "final_synthetic": final}, indent=2)
    )
    print("saved", out)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
