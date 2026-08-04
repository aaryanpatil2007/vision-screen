from __future__ import annotations

import math

from visionscreen.report import Finding

STEP_DOWN, STEP_UP = 0.1, 0.2
MAX_TRIALS, MAX_REVERSALS = 30, 6


def letter_height_px(logmar: float, distance_cm: float, px_per_cm: float) -> float:
    arcmin = 5.0 * (10.0 ** logmar)
    height_cm = 2.0 * distance_cm * math.tan(math.radians(arcmin / 60.0 / 2.0))
    return height_cm * px_per_cm


class Staircase:
    def __init__(self, start_logmar: float = 1.0, floor: float = -0.3, ceiling: float = 1.3):
        self._level = start_logmar
        self._floor, self._ceiling = floor, ceiling
        self._last_correct: bool | None = None
        self._reversals: list[float] = []
        self._trials = 0

    def current(self) -> float:
        return self._level

    @property
    def done(self) -> bool:
        return len(self._reversals) >= MAX_REVERSALS or self._trials >= MAX_TRIALS

    def record(self, correct: bool) -> None:
        if self.done:
            return
        self._trials += 1
        if self._last_correct is not None and correct != self._last_correct:
            self._reversals.append(self._level)
        self._last_correct = correct
        delta = -STEP_DOWN if correct else STEP_UP
        self._level = min(self._ceiling, max(self._floor, self._level + delta))

    def threshold(self) -> float | None:
        if not self.done or not self._reversals:
            return None
        tail = self._reversals[-4:]
        return sum(tail) / len(tail)


def score_trials(trials: list[dict]) -> Finding:
    n = len(trials)
    if n < 8:
        return Finding(
            module="acuity",
            summary=f"Not enough acuity trials completed ({n}) to estimate.",
            tier="inconclusive",
            retakes=["Complete the full letter test — answer every letter shown."],
        )
    # replay the recorded trials to recover the threshold
    correct_at: list[tuple[float, bool]] = [
        (t["logmar"], t["shown"] == t["answered"]) for t in trials
    ]
    # smallest level with majority-correct performance over >= 2 trials
    # (a single lucky guess at a tiny letter must not set the threshold)
    levels = sorted({lm for lm, _ in correct_at})
    threshold = levels[-1]
    for lv in levels:
        results = [ok for lm, ok in correct_at if lm == lv]
        if len(results) >= 2 and sum(results) / len(results) >= 0.5:
            threshold = lv
            break
    tier = "measured" if n >= 15 else "weak-signal"
    return Finding(
        module="acuity",
        summary=f"Estimated acuity {threshold:.2f} logMAR",
        tier=tier,
        metrics={"logmar": round(threshold, 2), "trials": n},
    )
