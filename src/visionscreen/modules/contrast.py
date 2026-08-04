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

import math
import statistics

from visionscreen.report import Finding

PELLI_ROBSON_STEP = 0.15
LETTERS_PER_TRIPLET = 3
NORMAL_LOG_CS = 1.75
REDUCED_LOG_CS = 1.50
MIN_TRIALS = 6
SRGB_GAMMA = 2.2


def display_ceiling_log_cs(background: int = 255, bits: int = 8) -> float:
    """Faintest logCS an uncalibrated display of this bit depth can present.

    One code step is the smallest contrast available. On 8-bit sRGB against a
    white background that is 255 -> 254, which is only logCS ~2.07 — so the
    bottom two Pelli-Robson triplets (2.10 and 2.25) are NOT reproducible and
    would render as byte-identical copies of the 1.95 stimulus. A subject
    "passing" them is guessing at an unchanged image.
    """
    max_code = (1 << bits) - 1
    bg_lin = (background / max_code) ** SRGB_GAMMA
    fg_lin = ((background - 1) / max_code) ** SRGB_GAMMA
    weber = (bg_lin - fg_lin) / bg_lin
    return float(-math.log10(weber)) if weber > 0 else float("inf")


def triplet_levels(start_log_cs: float = 0.0, n: int = 16,
                   ceiling: float | None = None) -> list[float]:
    """Log contrast-sensitivity level of each successive triplet.

    Truncated at the display ceiling so the ladder never contains two rungs
    that render to the same pixel value.
    """
    levels = [round(start_log_cs + PELLI_ROBSON_STEP * i, 4) for i in range(n)]
    if ceiling is None:
        return levels
    return [lv for lv in levels if lv <= ceiling + 1e-9]


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


def score_contrast(trials: list[dict], valid_fraction: float,
                   ceiling: float | None = None) -> Finding:
    """trials: [{"log_cs": float, "correct": bool}] in presentation order."""
    n = len(trials)
    if ceiling is None:
        ceiling = display_ceiling_log_cs()
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

    # Pelli-Robson scoring: the faintest TRIPLET with at least 2 of 3 letters
    # correct. Scanning for the last passing level (rather than stopping at the
    # first failure) keeps a single lapse from truncating the estimate — the
    # dominant error source when one letter is shown per level.
    levels = sorted(by_level)
    threshold = levels[0]
    for level in levels:
        results = by_level[level]
        passed = sum(results) / len(results) >= (2 / 3 if len(results) >= 3 else 0.5)
        if passed:
            threshold = level
    # credit partial performance on the next (failed) triplet, as the chart does
    nxt = [lv for lv in levels if lv > threshold]
    if nxt:
        part = by_level[nxt[0]]
        if part and 0 < sum(part) / len(part) < (2 / 3 if len(part) >= 3 else 0.5):
            threshold += PELLI_ROBSON_STEP * (sum(part) / len(part))

    flags: list[str] = []
    if threshold < REDUCED_LOG_CS:
        flags.append("reduced contrast sensitivity")
    elif threshold < NORMAL_LOG_CS:
        flags.append("borderline contrast sensitivity")

    tier = "measured" if (valid_fraction >= 0.7 and n >= 9) else "weak-signal"
    # Because the ladder is truncated at the display ceiling, "hit the ceiling"
    # means passing the highest rung that was actually presentable — the
    # threshold can never exceed it.
    at_ceiling = (
        threshold >= max(levels) - 1e-9
        and max(levels) >= ceiling - PELLI_ROBSON_STEP
    )
    ceiling = min(ceiling, max(levels)) if at_ceiling else ceiling
    if at_ceiling:
        # Everything beyond one code step is the same image; do not pretend
        # to have measured past it.
        flags = [f for f in flags if f != "borderline contrast sensitivity"]
        summary = (
            f"Contrast sensitivity at or better than {ceiling:.2f} log CS — the "
            "faintest letter this screen can draw. True sensitivity may be higher; "
            "measuring past this needs a 10-bit display."
        )
    else:
        summary = (
            f"Contrast sensitivity {threshold:.2f} log CS"
            + (f" — {', '.join(flags)}." if flags else " — within normal range.")
            + " Screen-based estimate; absolute value depends on display calibration."
        )

    metrics = {
        "log_cs": round(min(threshold, ceiling), 2),
        "weber_contrast_pct": round(100 * log_cs_to_weber(min(threshold, ceiling)), 2),
        "flags": flags,
        "trials": n,
    }
    if at_ceiling:
        metrics["display_ceiling_log_cs"] = round(ceiling, 2)
    return Finding(module="contrast", summary=summary, tier=tier, metrics=metrics)
