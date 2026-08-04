"""Segmentation benchmark: synthetic accuracy AND the sim-to-real gap.

Reports three columns per model:
  * synthetic held-out  — does the net fit its training distribution?
  * REAL held-out       — does it transfer to genuine webcam eyes?
  * the gap between them, which is the honest headline number.

Real ground truth is weak (geometric+photometric priors, not human labels), so
real IoU is a *proxy* for agreement with a defensible reference, not clinical
truth — stated plainly rather than papered over.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from visionscreen.ml.dataset import RealEyeDataset, SyntheticEyeDataset
from visionscreen.ml.model import EyeSegNet
from visionscreen.ml.train import evaluate, pick_device, train_model

OUT_SIZE = (64, 96)
REAL_ROOT = Path("data/corpus/real")
RESULTS = Path("results/segmentation_bench.json")


def real_split(root: Path = REAL_ROOT, holdout: float = 0.25, seed: int = 0):
    """Deterministic split of the real corpus; the same split is used everywhere."""
    ds = RealEyeDataset(root, OUT_SIZE)
    n = len(ds)
    if n == 0:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * holdout)
    return Subset(ds, idx[cut:].tolist()), Subset(ds, idx[:cut].tolist())


def _eval_subset(model, subset, device: str) -> dict:
    from torch.utils.data import DataLoader

    from visionscreen.ml.dataset import collate_resize
    from visionscreen.ml.model import CLASS_NAMES, N_CLASSES

    if subset is None or len(subset) == 0:
        return {}
    loader = DataLoader(subset, batch_size=16, collate_fn=collate_resize)
    model = model.to(device).eval()
    inter, union = np.zeros(N_CLASSES), np.zeros(N_CLASSES)
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            pred = model(imgs).argmax(1)
            for c in range(N_CLASSES):
                p, t = pred == c, masks == c
                inter[c] += (p & t).sum().item()
                union[c] += (p | t).sum().item()
    m = {f"{n_}_iou": (inter[c] / union[c]) if union[c] else float("nan")
         for c, n_ in enumerate(CLASS_NAMES)}
    valid = [v for v in m.values() if not np.isnan(v)]
    m["mean_iou"] = float(np.mean(valid)) if valid else float("nan")
    m["n_samples"] = len(subset)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def run(n_train: int = 6000, epochs: int = 12, device: str | None = None) -> dict:
    device = pick_device(device)
    train_real, test_real = real_split()
    out: dict = {"device": device, "epochs": epochs, "n_synth_train": n_train}

    # --- A: synthetic only (the sim-to-real baseline) ---
    model_a, hist_a = train_model(
        n_train=n_train, n_val=256, epochs=epochs, batch_size=48,
        out_size=OUT_SIZE, seed=1, device=device, log_every=0,
    )
    out["synthetic_only"] = {
        "synthetic_holdout": evaluate(model_a, 400, 7777, OUT_SIZE, device),
        "real_holdout": _eval_subset(model_a, test_real, device),
    }

    # --- B: synthetic + real (domain adaptation via weak labels) ---
    model_b, hist_b = train_model(
        n_train=n_train, n_val=256, epochs=epochs, batch_size=48,
        out_size=OUT_SIZE, seed=1, device=device, log_every=0,
        real_dataset=train_real,   # TRAIN split only — never the held-out real test
    )
    out["synthetic_plus_real"] = {
        "synthetic_holdout": evaluate(model_b, 400, 7777, OUT_SIZE, device),
        "real_holdout": _eval_subset(model_b, test_real, device),
    }

    a = out["synthetic_only"]["real_holdout"].get("mean_iou", float("nan"))
    b = out["synthetic_plus_real"]["real_holdout"].get("mean_iou", float("nan"))
    sa = out["synthetic_only"]["synthetic_holdout"].get("mean_iou", float("nan"))
    out["sim_to_real_gap_synthetic_only"] = round(sa - a, 4) if a == a else None
    out["real_iou_improvement"] = round(b - a, 4) if (a == a and b == b) else None
    out["real_train_n"] = len(train_real) if train_real else 0
    out["real_test_n"] = len(test_real) if test_real else 0
    return out, model_b


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-best", default="models/eyesegnet.pt")
    args = ap.parse_args()

    result, best = run(args.n_train, args.epochs, args.device)
    Path("results").mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=2))

    if args.save_best:
        Path(args.save_best).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best.state_dict(), "bench": result}, args.save_best)

    print("| model | synthetic mIoU | REAL mIoU | real pupil IoU |")
    print("|---|---|---|---|")
    for key, label in (("synthetic_only", "synthetic only"),
                       ("synthetic_plus_real", "synthetic + real (weak)")):
        s = result[key]["synthetic_holdout"].get("mean_iou")
        r = result[key]["real_holdout"].get("mean_iou")
        rp = result[key]["real_holdout"].get("pupil_iou")
        print(f"| {label} | {s} | {r} | {rp} |")
    print(f"\nsim-to-real gap (synthetic-only): {result['sim_to_real_gap_synthetic_only']}")
    print(f"real mIoU improvement from weak real data: {result['real_iou_improvement']}")
    print(f"real corpus: {result['real_train_n']} train / {result['real_test_n']} test")


if __name__ == "__main__":
    main()
