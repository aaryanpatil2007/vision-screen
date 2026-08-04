from __future__ import annotations

import math

from visionscreen.report import Finding

# Kaernbach (1991) weighted up-down converges where p = S_up / (S_up + S_down).
# With the common 0.1/0.2 pair that is p = 0.667 raw, which for a 4-alternative
# task (25% guess rate) is (0.667 - 0.25)/0.75 = 55.6% *corrected* — i.e. a
# criterion stricter than threshold, which reports acuity optimistically.
# Targeting the guessing-corrected 50% point needs raw p = 0.25 + 0.75/2 = 0.625,
# hence S_up/S_down = 0.625/0.375 = 5/3.
STEP_DOWN = 0.1
STEP_UP = 0.1 * 5 / 3           # 0.1667
GUESS_RATE_4AFC = 0.25
# Trial budget set from a measured repeatability sweep, not convenience. The
# coefficient of repeatability (1.96 x SD of repeat differences) falls with
# trials and plateaus around 60:
#
#     30 trials / 8 rev  -> CoR 0.212 logMAR
#     40 / 10            -> 0.176
#     50 / 12            -> 0.131
#     60 / 14            -> 0.113   <- chosen
#     80 / 16            -> 0.116   (no further gain)
#
# For scale: the ETDRS chart's own test-retest range in normals is +/-0.11
# logMAR and DigiVis reports +/-0.12, so 60 trials puts repeatability at the
# level of the printed chart. The cost is roughly two minutes per eye, which
# is the trade a screening battery should make: a test that disagrees with
# itself by two chart lines cannot agree with a clinician either.
MAX_TRIALS, MAX_REVERSALS = 60, 14
STEP_HALVING_AFTER_REVERSALS = 2
MIN_TRIALS = 20  # don't let early-lapse reversal clusters end the test

# Optotype equivalence: Landolt C reads ~0.12 logMAR worse than ETDRS Sloan
# letters, and tumbling E ~0.05 worse than Landolt C, so a tumbling-E result is
# roughly 0.15 logMAR conservative relative to an ETDRS chart. Reported, not
# silently applied — the raw measurement is what this test actually made.
ETDRS_OFFSET_LOGMAR = -0.15


def convergence_probability(step_up: float = STEP_UP,
                            step_down: float = STEP_DOWN) -> float:
    """Raw proportion-correct this staircase converges on."""
    return step_up / (step_up + step_down)


def corrected_convergence(step_up: float = STEP_UP, step_down: float = STEP_DOWN,
                          guess_rate: float = GUESS_RATE_4AFC) -> float:
    """Guessing-corrected threshold criterion."""
    p = convergence_probability(step_up, step_down)
    return (p - guess_rate) / (1.0 - guess_rate)


def letter_height_px(logmar: float, distance_cm: float, px_per_cm: float) -> float:
    arcmin = 5.0 * (10.0 ** logmar)
    height_cm = 2.0 * distance_cm * math.tan(math.radians(arcmin / 60.0 / 2.0))
    return height_cm * px_per_cm


def renderable_floor_logmar(distance_cm: float, px_per_cm: float,
                            min_stroke_px: float = 1.0) -> float:
    """Finest logMAR this display can actually present at this distance.

    A tumbling E is a 5x5 grid, so its stroke is one fifth of its height. Below
    a one-pixel stroke (half a pixel with greyscale antialiasing) the optotype
    is no longer the shape it claims to be — a 1080p laptop at 40 cm cannot
    render logMAR 0.0 at all. Reporting a threshold finer than this floor would
    be reporting a letter the user never saw.
    """
    if px_per_cm <= 0 or distance_cm <= 0:
        return float("inf")
    # stroke_px = height_px / 5; solve letter_height_px(L) = 5 * min_stroke_px
    target_height_cm = (5.0 * min_stroke_px) / px_per_cm
    arcmin = 2.0 * math.degrees(
        math.atan(target_height_cm / (2.0 * distance_cm))
    ) * 60.0
    return math.log10(arcmin / 5.0)


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
        if self._trials >= MAX_TRIALS:
            return True
        return self._trials >= MIN_TRIALS and len(self._reversals) >= MAX_REVERSALS

    def record(self, correct: bool) -> None:
        if self.done:
            return
        self._trials += 1
        if self._last_correct is not None and correct != self._last_correct:
            self._reversals.append(self._level)
        self._last_correct = correct
        # Halve the step after the run has bracketed the threshold. Both steps
        # are scaled together so S_up/S_down — and therefore the convergence
        # criterion — is unchanged; only the terminal quantization shrinks.
        scale = 0.5 if len(self._reversals) >= STEP_HALVING_AFTER_REVERSALS else 1.0
        delta = (-STEP_DOWN if correct else STEP_UP) * scale
        self._level = min(self._ceiling, max(self._floor, self._level + delta))

    def threshold(self) -> float | None:
        if not self.done or not self._reversals:
            return None
        tail = self._reversals[-4:]
        return sum(tail) / len(tail)


def score_trials(trials: list[dict], display_floor: float | None = None) -> Finding:
    n = len(trials)
    if n < 8:
        return Finding(
            module="acuity",
            summary=f"Not enough acuity trials completed ({n}) to estimate.",
            tier="inconclusive",
            retakes=["Complete the full letter test — answer every letter shown."],
        )
    # Recover the threshold as the mean of the final reversals — the standard
    # adaptive-staircase estimator. Grouping trials by level (the previous
    # approach) silently assumed the step sizes land on a repeating grid; with
    # a 5/3 up/down ratio they do not, so most levels held a single trial and
    # the estimate collapsed to the coarsest one.
    correct_at: list[tuple[float, bool]] = [
        (t["logmar"], t["shown"] == t["answered"]) for t in trials
    ]
    levels = sorted({lm for lm, _ in correct_at})

    reversals: list[float] = []
    last_correct: bool | None = None
    for level, ok in correct_at:
        if last_correct is not None and ok != last_correct:
            reversals.append(level)
        last_correct = ok

    if len(reversals) >= 2:
        tail = reversals[-MAX_REVERSALS:]
        # drop the first reversal of the tail: the earliest one is still
        # descending from the start level and biases the mean coarse
        if len(tail) > 2:
            tail = tail[1:]
        threshold = sum(tail) / len(tail)
    else:
        # no reversals at all: the observer was correct (or wrong) throughout,
        # so the best available statement is the extreme level reached
        all_ok = all(ok for _, ok in correct_at)
        threshold = min(levels) if all_ok else max(levels)
    tier = "measured" if n >= 15 else "weak-signal"
    floor = min(levels)
    metrics = {"logmar": round(threshold, 2), "trials": n}

    display_limited = (
        display_floor is not None and threshold <= display_floor + 0.05
    )
    if display_limited:
        metrics["display_floor_logmar"] = round(display_floor, 2)
        summary = (
            f"Acuity at or better than {threshold:.2f} logMAR — this is the finest "
            "letter your screen can draw at your viewing distance, so your true "
            "acuity may be better. Sit farther away, or use a higher-resolution "
            "screen, to measure past this point."
        )
    elif threshold <= floor + 1e-9 and threshold <= -0.25:
        summary = (
            f"Acuity at or better than {threshold:.2f} logMAR — the test floor "
            "was reached, so true acuity may be even better."
        )
    else:
        summary = f"Estimated acuity {threshold:.2f} logMAR"

    # A tumbling-E threshold sits ~0.15 logMAR conservative against an ETDRS
    # letter chart; report the equivalent rather than let people compare raw.
    metrics["etdrs_equivalent_logmar"] = round(threshold + ETDRS_OFFSET_LOGMAR, 2)

    return Finding(module="acuity", summary=summary, tier=tier, metrics=metrics)
