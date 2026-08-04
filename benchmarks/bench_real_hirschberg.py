"""Hirschberg accuracy on REAL eye images with known ground truth.

The synthetic benchmark (bench_module2) reported 0.60 PD error and was wrong
about real photographs, which produced readings up to 163 PD. The difference is
appearance: real skin, iris pigment, lids, lashes, sensor noise and ambient
reflections. The difference the other way is ground truth: real photographs
carry no known deviation.

This benchmark has both. It takes real eye crops from the weakly-labeled
corpus, suppresses any pre-existing corneal highlight, injects a controlled
glint at a **known** decentration, and runs the production detection +
inversion. Errors here are attributable to real image conditions rather than to
the simulator's own assumptions.

It also runs a NULL condition: the same real crops with no glint injected at
all, where any confident "measurement" is by construction an artifact. That is
the condition that caught the 163 PD failure.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from visionscreen.ml.infer import EyeSegmenter
from visionscreen.modules.alignment import (
    HIRSCHBERG_PD_PER_MM,
    MAX_PLAUSIBLE_PD,
    reflex_decentration_mm,
)
from visionscreen.perception.iris import detect_corneal_reflex
from visionscreen.synth.eyes2d import HVID_MM
from visionscreen.synth.glint import add_glint, suppress_existing_speculars

CORPUS = Path("data/corpus/real")
RESULTS = Path("results/real_hirschberg.json")
IRIS_CLASS, PUPIL_CLASS = 2, 3

# deviations to inject, in prism dioptres (converted to mm via the ratio)
DEVIATIONS_PD = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0]


def _iris_from_mask(mask: np.ndarray) -> tuple[tuple[float, float], float] | None:
    """Iris centre and diameter from the weak-label mask (our reference)."""
    disk = (mask == IRIS_CLASS) | (mask == PUPIL_CLASS) | (mask == 4)
    ys, xs = np.nonzero(disk)
    if len(xs) < 40:
        return None
    area = len(xs)
    return (float(xs.mean()), float(ys.mean())), float(2 * np.sqrt(area / np.pi))


def _measure(crop: np.ndarray, center: tuple[float, float], diameter: float,
             segmenter: EyeSegmenter) -> tuple[float, float] | None:
    """Production detection path: learned segmenter, classical fallback."""
    reflex = None
    if segmenter.available:
        seg = segmenter.segment(crop)
        if seg is not None and seg.reflex_center is not None:
            reflex = seg.reflex_center
    if reflex is None:
        reflex = detect_corneal_reflex(crop, center_xy=center, radius_px=diameter / 2)
    if reflex is None:
        return None
    return reflex_decentration_mm(reflex, center, diameter)


def run(n_crops: int = 300, seed: int = 11) -> dict:
    img_dir, mask_dir = CORPUS / "images", CORPUS / "masks"
    if not img_dir.is_dir():
        raise SystemExit("build the real corpus first")

    paths = sorted(img_dir.glob("*.png"))
    rng = np.random.default_rng(seed)
    if len(paths) > n_crops:
        idx = rng.choice(len(paths), n_crops, replace=False)
        paths = [paths[i] for i in idx]

    segmenter = EyeSegmenter()
    per_dev: dict[float, list[float]] = {d: [] for d in DEVIATIONS_PD}
    detected: dict[float, int] = {d: 0 for d in DEVIATIONS_PD}
    attempts = 0
    null_measurements: list[float] = []
    null_confident = 0

    for p in paths:
        mask_p = mask_dir / p.name
        if not mask_p.exists():
            continue
        crop = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if crop is None or mask is None:
            continue
        iris = _iris_from_mask(mask)
        if iris is None:
            continue
        center, diameter = iris
        if diameter < 12:
            continue
        attempts += 1

        clean = suppress_existing_speculars(crop, center, diameter)

        # --- NULL condition: no glint at all ---
        null = _measure(clean, center, diameter, segmenter)
        if null is not None:
            pd = float(np.hypot(*null)) * HIRSCHBERG_PD_PER_MM
            null_measurements.append(pd)
            if pd <= MAX_PLAUSIBLE_PD:
                null_confident += 1

        # --- injected glint at known decentration ---
        for dev_pd in DEVIATIONS_PD:
            dev_mm = dev_pd / HIRSCHBERG_PD_PER_MM
            angle = rng.uniform(0, 2 * np.pi)
            offset = (dev_mm * np.cos(angle), dev_mm * np.sin(angle))
            withglint = add_glint(clean, center, diameter, offset)
            got = _measure(withglint, center, diameter, segmenter)
            if got is None:
                continue
            detected[dev_pd] += 1
            est_pd = float(np.hypot(*got)) * HIRSCHBERG_PD_PER_MM
            per_dev[dev_pd].append(abs(est_pd - dev_pd))

    rows = []
    for dev in DEVIATIONS_PD:
        errs = per_dev[dev]
        rows.append({
            "deviation_pd": dev,
            "n": len(errs),
            "detection_rate": round(detected[dev] / attempts, 3) if attempts else 0.0,
            "mean_abs_error_pd": round(float(np.mean(errs)), 2) if errs else None,
            "p90_abs_error_pd": round(float(np.percentile(errs, 90)), 2) if errs else None,
        })

    all_errs = [e for d in DEVIATIONS_PD for e in per_dev[d]]
    return {
        "crops_used": attempts,
        "rows": rows,
        "mean_abs_error_pd": round(float(np.mean(all_errs)), 2) if all_errs else None,
        "null_condition": {
            "n_measured": len(null_measurements),
            "measured_rate": round(len(null_measurements) / attempts, 3) if attempts else 0,
            "false_confident": null_confident,
            "false_confident_rate": round(null_confident / attempts, 3) if attempts else 0,
            "median_pd": round(float(np.median(null_measurements)), 1)
                         if null_measurements else None,
        },
    }


def run_multiframe(n_trials: int = 120, frames_per_trial: int = 40,
                   deviation_pd: float = 15.0, seed: int = 3) -> dict:
    """The decisive question: does frame aggregation rescue the accuracy?

    A single real crop gives ~7.5 PD error. The product does not score single
    frames — it takes the median decentration across a whole capture segment.
    If the per-frame error is largely independent noise, the median over N
    frames should shrink it toward the synthetic figure; if it is a systematic
    bias, aggregation will not help at all. This distinguishes the two.
    """
    img_dir, mask_dir = CORPUS / "images", CORPUS / "masks"
    paths = sorted(img_dir.glob("*.png"))
    rng = np.random.default_rng(seed)
    segmenter = EyeSegmenter()

    usable: list[tuple[np.ndarray, tuple[float, float], float]] = []
    for p in paths:
        mask_p = mask_dir / p.name
        if not mask_p.exists():
            continue
        crop = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if crop is None or mask is None:
            continue
        iris = _iris_from_mask(mask)
        if iris is None or iris[1] < 12:
            continue
        usable.append((suppress_existing_speculars(crop, iris[0], iris[1]),
                       iris[0], iris[1]))
        if len(usable) >= 600:
            break
    if len(usable) < frames_per_trial:
        return {"error": "not enough usable crops"}

    dev_mm = deviation_pd / HIRSCHBERG_PD_PER_MM
    single_errs, agg_errs = [], []
    for _ in range(n_trials):
        picks = rng.choice(len(usable), frames_per_trial, replace=False)
        per_frame = []
        for i in picks:
            crop, center, diameter = usable[i]
            angle = rng.uniform(0, 2 * np.pi)
            offset = (dev_mm * np.cos(angle), dev_mm * np.sin(angle))
            got = _measure(add_glint(crop, center, diameter, offset),
                           center, diameter, segmenter)
            if got is None:
                continue
            per_frame.append(float(np.hypot(*got)) * HIRSCHBERG_PD_PER_MM)
        if not per_frame:
            continue
        single_errs.append(abs(per_frame[0] - deviation_pd))
        agg_errs.append(abs(float(np.median(per_frame)) - deviation_pd))

    return {
        "deviation_pd": deviation_pd,
        "frames_per_trial": frames_per_trial,
        "trials": len(agg_errs),
        "single_frame_mean_abs_error_pd": round(float(np.mean(single_errs)), 2),
        "aggregated_mean_abs_error_pd": round(float(np.mean(agg_errs)), 2),
        "aggregated_p90_abs_error_pd": round(float(np.percentile(agg_errs, 90)), 2),
    }


def main() -> None:
    result = run()
    Path("results").mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=2))

    print(f"Hirschberg on REAL eye crops with injected known glint "
          f"({result['crops_used']} crops)\n")
    print("| true deviation | detection | mean abs err | p90 err |")
    print("|---|---|---|---|")
    for r in result["rows"]:
        print(f"| {r['deviation_pd']:.0f} PD | {r['detection_rate']:.2f} | "
              f"{r['mean_abs_error_pd']} PD | {r['p90_abs_error_pd']} PD |")
    print(f"\noverall mean absolute error: {result['mean_abs_error_pd']} PD")
    n = result["null_condition"]
    print(f"\nNULL condition (no glint injected — any confident reading is an artifact):")
    print(f"  produced a measurement on {n['measured_rate']:.0%} of crops, "
          f"median {n['median_pd']} PD")
    print(f"  passed the plausibility ceiling on {n['false_confident_rate']:.0%} "
          f"({n['false_confident']}/{result['crops_used']})")

    mf = run_multiframe()
    result["multiframe"] = mf
    RESULTS.write_text(json.dumps(result, indent=2))
    if "error" not in mf:
        print(f"\nFrame aggregation ({mf['frames_per_trial']} frames, "
              f"{mf['trials']} trials at {mf['deviation_pd']:.0f} PD):")
        print(f"  single frame:  {mf['single_frame_mean_abs_error_pd']} PD")
        print(f"  median of {mf['frames_per_trial']}: "
              f"{mf['aggregated_mean_abs_error_pd']} PD "
              f"(p90 {mf['aggregated_p90_abs_error_pd']} PD)")


if __name__ == "__main__":
    main()
