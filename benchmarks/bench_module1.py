from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visionscreen.modules.acuity import Staircase, score_trials

DIRS = ["up", "down", "left", "right"]


def simulate_observer(true_logmar: float, lapse: float, rng: np.random.Generator) -> list[dict]:
    """Ideal-ish observer: sees letters larger than threshold, guesses below it,
    with a lapse rate of random errors."""
    s = Staircase()
    trials: list[dict] = []
    while not s.done:
        level = s.current()
        shown = DIRS[rng.integers(4)]
        sees = level >= true_logmar and rng.random() > lapse
        answered = shown if sees else DIRS[rng.integers(4)]
        trials.append({"logmar": round(level, 2), "shown": shown, "answered": answered})
        s.record(answered == shown)
    return trials


def run_benchmark(n_observers: int = 50, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_observers):
        true_logmar = float(rng.uniform(-0.1, 1.0))
        trials = simulate_observer(true_logmar, lapse=0.05, rng=rng)
        finding = score_trials(trials)
        est = finding.metrics.get("logmar_raw_tumbling_e")
        rows.append({"true": round(true_logmar, 2), "est": est,
                     "tier": finding.tier,
                     "abs_error": None if est is None else abs(est - true_logmar)})
    errors = [r["abs_error"] for r in rows if r["abs_error"] is not None]
    return {
        "n": n_observers,
        "mean_abs_error": round(float(np.mean(errors)), 3),
        "max_abs_error": round(float(np.max(errors)), 3),
        "rows": rows,
    }


def main() -> None:
    result = run_benchmark()
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "module1.json").write_text(json.dumps(result, indent=2))
    print("| metric | value |\n|---|---|")
    print(f"| observers | {result['n']} |")
    print(f"| mean abs error (logMAR) | {result['mean_abs_error']} |")
    print(f"| max abs error (logMAR) | {result['max_abs_error']} |")


if __name__ == "__main__":
    main()
