"""Pupil light reflex and relative afferent pupillary defect (RAPD) screening.

The screen flashes; the webcam measures each pupil's diameter over time. The
clinically meaningful quantities are constriction amplitude (percent of
baseline), latency to onset (~200-250 ms normally), and recovery. A RAPD --
the classic swinging-flashlight finding -- shows up here as a marked
inter-eye asymmetry in constriction to the same stimulus, and is one of the
few objective signs of optic-nerve disease obtainable without instruments.

Screen flashes are far dimmer than a clinical transilluminator, so absolute
amplitudes are not comparable to clinic values; asymmetry between the two
eyes under the identical stimulus is the defensible signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visionscreen.report import Finding

MIN_SAMPLES = 20
NO_RESPONSE_PCT = 5.0
NORMAL_CONSTRICTION_PCT = 12.0
RAPD_RATIO = 0.45          # weaker/stronger amplitude ratio below this = asymmetric
BASELINE_WINDOW_S = 0.4


@dataclass(frozen=True)
class PupilTrace:
    ts: list[float]
    diameter_mm: list[float]
    flash_ts: float


def constriction_metrics(trace: PupilTrace) -> dict | None:
    if len(trace.ts) < MIN_SAMPLES:
        return None
    ts = np.asarray(trace.ts, float)
    d = np.asarray(trace.diameter_mm, float)

    pre = d[(ts >= trace.flash_ts - BASELINE_WINDOW_S) & (ts <= trace.flash_ts)]
    baseline = float(np.median(pre)) if pre.size >= 3 else float(np.median(d[ts <= trace.flash_ts]))
    if not np.isfinite(baseline) or baseline <= 0:
        return None

    post_mask = ts > trace.flash_ts
    if post_mask.sum() < 5:
        return None
    post_ts, post_d = ts[post_mask], d[post_mask]
    min_idx = int(np.argmin(post_d))
    min_d = float(post_d[min_idx])
    constriction_pct = 100.0 * (baseline - min_d) / baseline

    # latency: first sample dropping 10% of the way to the trough
    latency = float("nan")
    if constriction_pct > NO_RESPONSE_PCT:
        target = baseline - 0.1 * (baseline - min_d)
        below = np.nonzero(post_d <= target)[0]
        if below.size:
            latency = float(post_ts[below[0]] - trace.flash_ts)

    return {
        "baseline_mm": round(baseline, 3),
        "min_mm": round(min_d, 3),
        "constriction_pct": round(constriction_pct, 2),
        "latency_s": round(latency, 3) if np.isfinite(latency) else None,
        "samples": int(len(ts)),
    }


def score_pupillometry(
    left: PupilTrace | None,
    right: PupilTrace | None,
    valid_fraction: float,
) -> Finding:
    ml = constriction_metrics(left) if left else None
    mr = constriction_metrics(right) if right else None

    if valid_fraction < 0.4 or (ml is None and mr is None):
        return Finding(
            module="pupillometry",
            summary="Pupil light response could not be measured.",
            tier="inconclusive",
            retakes=[
                "Repeat the flash test in a dim room, looking at the camera and holding still.",
            ],
        )

    flags: list[str] = []
    amps = [m["constriction_pct"] for m in (ml, mr) if m is not None]
    if max(amps) < NO_RESPONSE_PCT:
        flags.append("no measurable light response")
    elif ml and mr:
        lo, hi = sorted([ml["constriction_pct"], mr["constriction_pct"]])
        if hi > NO_RESPONSE_PCT and (lo / hi) < RAPD_RATIO:
            flags.append("asymmetric pupil response")

    weak = [m for m in (ml, mr) if m and NO_RESPONSE_PCT <= m["constriction_pct"] < NORMAL_CONSTRICTION_PCT]
    if weak and "no measurable light response" not in flags:
        flags.append("reduced constriction amplitude")

    if "asymmetric pupil response" in flags:
        summary = (
            "The two pupils responded unequally to the same light — a pattern that can "
            "indicate a relative afferent pupillary defect and should be evaluated."
        )
    elif "no measurable light response" in flags:
        summary = (
            "No pupil constriction was detected. Screen flashes are weak, so this is "
            "most often a capture limitation rather than a clinical finding — repeat in a dark room."
        )
    elif flags:
        summary = "Pupil constriction was present but smaller than typical for this stimulus."
    else:
        summary = "Both pupils constricted promptly and symmetrically to light."

    tier = "measured" if (valid_fraction >= 0.7 and ml and mr) else "weak-signal"
    metrics = {"flags": flags}
    if ml:
        metrics["left"] = ml
    if mr:
        metrics["right"] = mr
    return Finding(module="pupillometry", summary=summary, tier=tier, metrics=metrics)
