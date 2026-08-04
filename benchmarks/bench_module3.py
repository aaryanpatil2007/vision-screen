from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visionscreen.modules.photoref import measure_reflex
from visionscreen.synth.photoref import render_reflex

PX_PER_M = 8000.0
KW = dict(e_m=0.005, d_m=0.5, px_per_m=PX_PER_M)
# spheres beyond the ±1.25 D dead zone of this geometry
SPHERES = [round(s, 2) for s in np.arange(-4.0, -1.49, 0.25)] + [
    round(s, 2) for s in np.arange(1.5, 4.01, 0.25)
]
CYLINDERS = [0.0, 1.0, 2.0]


def _sweep(noise_sigma: float, seeds: int) -> dict:
    sphere_errors: list[float] = []
    axis_errors: list[float] = []
    attempts = detected = 0
    for S in SPHERES:
        for C in CYLINDERS:
            # keep every meridian outside the dead zone (see Plan 3)
            if abs(S) - C < 1.5 and S < 0:
                continue
            if S > 0 and S < 1.5:
                continue
            for seed in range(seeds):
                rng = np.random.default_rng(seed)
                axis = float(rng.uniform(0, 180)) if C > 0 else 0.0
                img, truth = render_reflex(
                    32, S=S, C=C, axis_deg=axis, noise_sigma=noise_sigma, rng=rng,
                    **KW,
                )
                attempts += 1
                est = measure_reflex(img, truth["center_px"], truth["pupil_radius_px"], **KW)
                if est is None:
                    continue
                detected += 1
                S_est, C_est, axis_est = est
                # compare spherical equivalents (S + C/2) — stable across the
                # sphere/cylinder split ambiguity at meridian extremes
                se_true, se_est = S + C / 2.0, S_est + C_est / 2.0
                sphere_errors.append(abs(se_est - se_true))
                if C > 0:
                    axis_errors.append(
                        min(abs(axis_est - axis), 180 - abs(axis_est - axis))
                    )
    return {
        "detection_rate": round(detected / attempts, 3) if attempts else 0.0,
        "mean_abs_sphere_error_d": round(float(np.mean(sphere_errors)), 3),
        "max_abs_sphere_error_d": round(float(np.max(sphere_errors)), 3),
        "mean_axis_error_deg": round(float(np.mean(axis_errors)), 1) if axis_errors else None,
        "n": attempts,
    }


def run_benchmark(seeds: int = 3) -> dict:
    return {"clean": _sweep(0.0, seeds), "noisy": _sweep(25.0, seeds)}


def main() -> None:
    result = run_benchmark()
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "module3.json").write_text(json.dumps(result, indent=2))
    print("| condition | detection | mean abs SE error (D) | max (D) | axis err (deg) |")
    print("|---|---|---|---|---|")
    for cond in ("clean", "noisy"):
        r = result[cond]
        print(
            f"| {cond} | {r['detection_rate']} | {r['mean_abs_sphere_error_d']} "
            f"| {r['max_abs_sphere_error_d']} | {r['mean_axis_error_deg']} |"
        )


if __name__ == "__main__":
    main()
