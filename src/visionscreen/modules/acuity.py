from __future__ import annotations

import math
from dataclasses import dataclass

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

# Sloan letters are the ETDRS optotype itself, so using them removes the
# tumbling-E equivalence term rather than correcting for it — the single
# largest contributor to the acuity error budget (48% of variance, and a
# -0.15 logMAR systematic offset). The task becomes 10-alternative, so the
# guess rate falls to 0.10 and the staircase ratio must change to keep
# converging on the guessing-corrected 50% point:
#     raw p = 0.10 + 0.90/2 = 0.55  ->  S_up/S_down = 0.55/0.45 = 11/9
GUESS_RATE_10AFC = 0.10
STEP_UP_SLOAN = 0.1 * 11 / 9    # 0.1222
SLOAN_LETTERS = ("C", "D", "H", "K", "N", "O", "R", "S", "V", "Z")


@dataclass(frozen=True)
class Optotype:
    """A stimulus family, its guess rate, and its offset from the ETDRS chart."""
    name: str
    guess_rate: float
    step_up: float
    chart_offset_logmar: float   # add to convert a threshold to chart scale


TUMBLING_E = Optotype("tumbling_e", GUESS_RATE_4AFC, STEP_UP, -0.15)
SLOAN = Optotype("sloan", GUESS_RATE_10AFC, STEP_UP_SLOAN, 0.0)
OPTOTYPES = {o.name: o for o in (TUMBLING_E, SLOAN)}
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
    def __init__(self, start_logmar: float = 1.0, floor: float = -0.3,
                 ceiling: float = 1.3, optotype: Optotype = TUMBLING_E):
        self._level = start_logmar
        self._floor, self._ceiling = floor, ceiling
        self.optotype = optotype
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
        step_up = self.optotype.step_up
        delta = (-STEP_DOWN if correct else step_up) * scale
        self._level = min(self._ceiling, max(self._floor, self._level + delta))

    def threshold(self) -> float | None:
        if not self.done or not self._reversals:
            return None
        tail = self._reversals[-4:]
        return sum(tail) / len(tail)


def _above_chance(correct: int, n: int, chance: float,
                  alpha: float = 0.05) -> bool:
    """One-sided exact binomial test that performance beats guessing.

    Exact rather than normal-approximated because n is small (tens of trials)
    and chance can be 0.10, where the normal approximation is poor precisely in
    the tail that matters.
    """
    if n <= 0:
        return False
    from math import comb

    tail = sum(comb(n, k) * chance ** k * (1 - chance) ** (n - k)
               for k in range(correct, n + 1))
    return tail < alpha


def score_trials(trials: list[dict], display_floor: float | None = None,
                 optotype: Optotype | str = TUMBLING_E) -> Finding:
    if isinstance(optotype, str):
        optotype = OPTOTYPES[optotype]
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

    # --- did they see the largest letter at all? ---
    # A subject who never beats chance at the coarsest optotype has vision
    # below what this test can present. Reporting the ceiling as a threshold
    # would hand a confident "20/400" to someone who saw nothing, which is
    # both wrong and the opposite of useful for the person most in need of
    # attention.
    coarsest = max(lm for lm, _ in correct_at)
    at_coarsest = [ok for lm, ok in correct_at if lm >= coarsest - 1e-9]
    chance = optotype.guess_rate
    # Requiring the staircase to have *parked* at the coarsest level is what
    # separates "cannot see the biggest letter" from "answers were noise". A
    # subject who truly sees nothing drives the level to the ceiling and stays
    # there, so most trials land on one level; random input instead scatters
    # trials across many levels while scoring chance on each. Without this
    # share test, a scattered run that happened to miss its few coarsest trials
    # would be reported as a confident "worse than 20/400".
    coarsest_share = len(at_coarsest) / max(len(correct_at), 1)
    if (len(at_coarsest) >= 3 and coarsest_share >= 0.25
            and (sum(at_coarsest) / len(at_coarsest)) <= chance + 1e-9):
        return Finding(
            module="acuity",
            summary=(
                f"The largest letter this test can show ({coarsest:.2f} logMAR, "
                f"about {snellen_hint(coarsest)}) was not read reliably, so acuity "
                "is below the range this screening can measure. That is a result, "
                "not a failure — it should be taken to an optometrist rather than "
                "retaken here."
            ),
            tier="measured",
            metrics={
                "flags": ["acuity below measurable range"],
                "worse_than_logmar": round(coarsest, 2),
                "worse_than_snellen": snellen_hint(coarsest),
                "trials": n,
                "optotype": optotype.name,
            },
        )

    # --- is this a measurement at all? ---
    # A staircase converges because the answers carry information. If the
    # responses are indistinguishable from guessing, the level random-walks
    # around wherever it started and the reversal mean returns that starting
    # value with false precision. That is what a broken input, a misunderstood
    # instruction or an unreadable display produces — and from the numbers
    # alone it is indistinguishable from a real threshold unless checked.
    #
    # Reporting "20/200, measured" from noise is the worst failure this module
    # can produce, because it is confidently wrong about someone who may be fine.
    n_correct = sum(1 for _, ok in correct_at if ok)
    if not _above_chance(n_correct, n, optotype.guess_rate):
        return Finding(
            module="acuity",
            summary=(
                f"Only {n_correct} of {n} letters were answered correctly — no "
                f"better than guessing at {optotype.guess_rate:.0%} chance. No "
                "acuity can be read from that. Usually it means the letters "
                "were not visible, the buttons were not doing what was "
                "expected, or the test was rushed; it does not by itself mean "
                "anything about your vision."
            ),
            tier="inconclusive",
            metrics={"trials": n, "correct": n_correct,
                     "chance_rate": optotype.guess_rate,
                     "rejected_reason": "responses no better than chance"},
            retakes=[
                "Retake the letter test, answering every letter deliberately — "
                "guess only when you genuinely cannot tell.",
                "If the letters looked clear but the answer buttons felt wrong, "
                "that is a bug worth reporting rather than a result.",
            ],
        )


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

    # Report the CHART-EQUIVALENT value as the headline. A raw tumbling-E
    # threshold sits ~0.15 logMAR conservative against ETDRS Sloan letters, so
    # publishing it as "your acuity" would carry a systematic 1.5-line bias
    # against the chart everyone actually compares to — the largest single term
    # in the error budget, and correctable exactly because it is known.
    raw = threshold
    chart_equivalent = threshold + optotype.chart_offset_logmar
    metrics["logmar"] = round(chart_equivalent, 2)
    metrics["logmar_raw_tumbling_e"] = round(raw, 2)
    metrics["optotype"] = optotype.name
    metrics["optotype_correction_logmar"] = optotype.chart_offset_logmar

    summary = summary.replace(f"{raw:.2f} logMAR", f"{chart_equivalent:.2f} logMAR")
    if not display_limited and not (raw <= floor + 1e-9 and raw <= -0.25):
        scale_note = (
            "measured directly with ETDRS letters"
            if optotype.chart_offset_logmar == 0.0
            else "adjusted to letter-chart scale"
        )
        summary = (
            f"Estimated acuity {chart_equivalent:.2f} logMAR "
            f"({snellen_hint(chart_equivalent)}), {scale_note}."
        )

    return Finding(module="acuity", summary=summary, tier=tier, metrics=metrics)


def snellen_hint(logmar: float) -> str:
    return f"20/{round(20 * (10 ** logmar))}"
