from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from visionscreen.modules.alignment import (
    HIRSCHBERG_PD_PER_MM,
    reflex_decentration_mm,
)
from visionscreen.perception.iris import detect_corneal_reflex
from visionscreen.synth.eyes2d import render_eye

ASYMMETRIES_MM = [round(0.25 * i, 2) for i in range(9)]  # 0.0 .. 2.0


def _measure_offset_mm(offset_mm: float, noise_sigma: float, rng) -> float | None:
    """Render one eye with a known reflex offset, run the real perception path,
    return the measured decentration magnitude in mm (None if reflex missed)."""
    img, truth = render_eye(
        width_px=200, iris_diameter_px=80.0,
        reflex_offset_mm=(offset_mm, 0.0),
        noise_sigma=noise_sigma, rng=rng,
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reflex = detect_corneal_reflex(
        gray,
        center_xy=truth["iris_center_px"],
        radius_px=truth["iris_diameter_px"] / 2,
    )
    if reflex is None:
        return None
    dx, dy = reflex_decentration_mm(
        reflex, truth["iris_center_px"], truth["iris_diameter_px"]
    )
    return float(np.hypot(dx, dy))


def run_benchmark(seeds: int = 20, noise_sigma: float = 8.0) -> dict:
    rows = []
    detected = 0
    attempts = 0
    for asym in ASYMMETRIES_MM:
        errors = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            # deviated eye carries the asymmetry; fellow eye centered
            measured = _measure_offset_mm(asym, noise_sigma, rng)
            attempts += 1
            if measured is None:
                continue
            detected += 1
            true_pd = asym * HIRSCHBERG_PD_PER_MM
            est_pd = measured * HIRSCHBERG_PD_PER_MM
            errors.append(abs(est_pd - true_pd))
        rows.append({
            "asymmetry_mm": asym,
            "true_pd": round(asym * HIRSCHBERG_PD_PER_MM, 1),
            "mean_abs_error_pd": round(float(np.mean(errors)), 2) if errors else None,
            "n": len(errors),
        })
    all_errors = [r["mean_abs_error_pd"] for r in rows if r["mean_abs_error_pd"] is not None]
    return {
        "detection_rate": round(detected / attempts, 3) if attempts else 0.0,
        "mean_abs_error_pd": round(float(np.mean(all_errors)), 2),
        "max_abs_error_pd": round(float(np.max(all_errors)), 2),
        "rows": rows,
    }


def main() -> None:
    result = run_benchmark()
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "module2.json").write_text(json.dumps(result, indent=2))
    print("| metric | value |\n|---|---|")
    print(f"| detection rate | {result['detection_rate']} |")
    print(f"| mean abs error (PD) | {result['mean_abs_error_pd']} |")
    print(f"| max abs error (PD) | {result['max_abs_error_pd']} |")


if __name__ == "__main__":
    main()
