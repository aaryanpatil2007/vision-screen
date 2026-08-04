"""Letter contrast sensitivity, Pelli-Robson style.

Pelli, Robson & Wilkins (1988): letters at a fixed large size (~3 cycles/deg
equivalent) descend in contrast in 0.15 log-unit steps, three letters per
triplet, and the score is the faintest triplet largely read correctly.
Normal adults reach ~1.75-2.00 log CS; below ~1.5 is clinically reduced and
correlates with cataract, amblyopia and early retinal disease that ordinary
acuity testing can miss entirely.

Screen caveat (documented, not hidden): an uncalibrated LCD is not photometric
and gamma varies, so absolute log CS carries a systematic error. What survives
is the *relative* measure and gross reduction — which is what a screening tool
needs to flag.
"""
from __future__ import annotations

import statistics

from visionscreen.report import Finding

PELLI_ROBSON_STEP = 0.15
LETTERS_PER_TRIPLET = 3
NORMAL_LOG_CS = 1.75
REDUCED_LOG_CS = 1.50
MIN_TRIALS = 6
SRGB_GAMMA = 2.2


def triplet_levels(start_log_cs: float = 0.0, n: int = 16) -> list[float]:
    """Log contrast-sensitivity level of each successive triplet."""
    return [round(start_log_cs + PELLI_ROBSON_STEP * i, 4) for i in range(n)]


def log_cs_to_weber(log_cs: float) -> float:
    """Log contrast sensitivity -> Weber contrast (1/sensitivity)."""
    return 10.0 ** (-log_cs)


def contrast_to_luminance_pair(log_cs: float, background: int = 255) -> tuple[int, int]:
    """8-bit sRGB code values for a letter at the requested contrast.

    Contrast is defined in LINEAR luminance, then gamma-encoded — doing this
    in code-value space (the common shortcut) would misstate the contrast by
    the display gamma.
    """
    weber = log_cs_to_weber(log_cs)
    bg_lin = (background / 255.0) ** SRGB_GAMMA
    fg_lin = max(0.0, bg_lin * (1.0 - weber))
    fg = int(round(255.0 * (fg_lin ** (1.0 / SRGB_GAMMA))))
    return max(0, min(255, fg)), background


def score_contrast(trials: list[dict], valid_fraction: float) -> Finding:
    """trials: [{"log_cs": float, "correct": bool}] in presentation order."""
    n = len(trials)
    if n < MIN_TRIALS:
        return Finding(
            module="contrast",
            summary="Not enough contrast trials to estimate sensitivity.",
            tier="inconclusive",
            retakes=["Complete the contrast letters test — answer every letter."],
        )

    by_level: dict[float, list[bool]] = {}
    for t in trials:
        by_level.setdefault(round(float(t["log_cs"]), 2), []).append(bool(t["correct"]))

    # threshold = faintest (highest log CS) level still mostly correct
    threshold = min(by_level)
    for level in sorted(by_level):
        results = by_level[level]
        if sum(results) / len(results) >= 0.5:
            threshold = level
        else:
            break

    flags: list[str] = []
    if threshold < REDUCED_LOG_CS:
        flags.append("reduced contrast sensitivity")
    elif threshold < NORMAL_LOG_CS:
        flags.append("borderline contrast sensitivity")

    tier = "measured" if (valid_fraction >= 0.7 and n >= 9) else "weak-signal"
    summary = (
        f"Contrast sensitivity {threshold:.2f} log CS"
        + (f" — {', '.join(flags)}." if flags else " — within normal range.")
        + " Screen-based estimate; absolute value depends on display calibration."
    )
    return Finding(
        module="contrast",
        summary=summary,
        tier=tier,
        metrics={
            "log_cs": round(threshold, 2),
            "weber_contrast_pct": round(100 * log_cs_to_weber(threshold), 2),
            "flags": flags,
            "trials": n,
        },
    )
