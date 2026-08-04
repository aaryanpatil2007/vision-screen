from __future__ import annotations

import statistics

from visionscreen.report import Finding

SQUINT_EAR = 0.18
SQUINT_FRACTION_FLAG = 0.20
LEAN_RATIO_FLAG = 1.15
TILT_DEG_FLAG = 8.0


def _tier(valid_fraction: float) -> str:
    if valid_fraction >= 0.7:
        return "measured"
    if valid_fraction >= 0.4:
        return "weak-signal"
    return "inconclusive"


def analyze_series(
    ears: list[float],
    interocular: list[float],
    rolls: list[float],
    valid_fraction: float,
) -> Finding:
    tier = _tier(valid_fraction)
    if tier == "inconclusive" or not ears:
        return Finding(
            module="behavioral",
            summary="Too few usable frames to assess viewing behavior.",
            tier="inconclusive",
            retakes=["Re-record with your face steady, centered, and well lit."],
        )

    flags: list[str] = []
    squint_fraction = sum(e < SQUINT_EAR for e in ears) / len(ears)
    if squint_fraction > SQUINT_FRACTION_FLAG:
        flags.append("frequent squinting")

    k = max(1, len(interocular) // 5)
    lean_ratio = statistics.median(interocular[-k:]) / statistics.median(interocular[:k])
    if lean_ratio > LEAN_RATIO_FLAG:
        flags.append("leaning toward screen")

    if statistics.median(abs(r) for r in rolls) > TILT_DEG_FLAG:
        flags.append("sustained head tilt")

    summary = (
        "No behavioral signs of visual strain observed."
        if not flags
        else "Behavioral signs observed: " + ", ".join(flags) + "."
    )
    return Finding(
        module="behavioral",
        summary=summary,
        tier=tier,
        metrics={
            "flags": flags,
            "squint_fraction": round(squint_fraction, 3),
            "lean_ratio": round(lean_ratio, 3),
        },
    )
