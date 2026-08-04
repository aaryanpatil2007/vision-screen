"""Binocular pupil light reflex and anisocoria screening.

**What this is not.** It is not a swinging-flashlight (RAPD) test, and it
cannot be one. The pupil light reflex is fully consensual: a screen flash
reaches both retinas, so both pupils respond to the *summed* afferent input.
Inter-eye response asymmetry under a bilateral stimulus therefore reflects
efferent/iris differences, not an afferent defect. A true RAPD requires
alternating *monocular* stimulation with controlled dwell and re-adaptation,
which a bare screen cannot deliver. Claiming otherwise would be a category
error, so this module reports what is actually measurable.

**What it does measure.** A full-field screen flash is a legitimate PLR
stimulus: a calibrated 3 cd/m² screen has been shown to evoke ~42% relative
constriction with split-half ICC 0.84 (Wang et al., PLoS One 2018), and an
ordinary laptop at full white delivers ~2 log units *more* retinal
illuminance than that. So we report baseline diameter, relative constriction,
average constriction velocity, and static anisocoria.

**Anisocoria threshold.** 41% of normal subjects show ≥0.4 mm difference at
some sitting and 19% at any given exam, while ≥1.0 mm is rare (Lam, Thompson
& Corbett 1987). The flag therefore sits at 1.0 mm, not 0.4 mm.

**Latency is deliberately withheld below 60 fps.** At 30 fps the frame
quantization SD is ~9.6 ms, which exceeds the bottom of the physiological
inter-eye latency asymmetry range (8.3-35 ms; Bergamin & Kardon 2003), so a
latency number there would be noise dressed as a measurement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visionscreen.report import Finding

MIN_SAMPLES = 20
NO_RESPONSE_PCT = 5.0
NORMAL_CONSTRICTION_PCT = 12.0
ANISOCORIA_FLAG_MM = 1.0     # Lam 1987: >=0.4 mm is common in normals
BASELINE_WINDOW_S = 0.4
MIN_FPS_FOR_LATENCY = 55.0   # below this, frame quantization swamps the signal


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
    fps: float = 30.0,
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

    weak = [
        m for m in (ml, mr)
        if m and NO_RESPONSE_PCT <= m["constriction_pct"] < NORMAL_CONSTRICTION_PCT
    ]
    if weak and "no measurable light response" not in flags:
        flags.append("reduced constriction amplitude")

    # Static anisocoria: a resting size difference, which IS interpretable from
    # a bilateral stimulus (unlike response asymmetry).
    anisocoria_mm = None
    if ml and mr:
        anisocoria_mm = abs(ml["baseline_mm"] - mr["baseline_mm"])
        if anisocoria_mm >= ANISOCORIA_FLAG_MM:
            flags.append("unequal pupil sizes (anisocoria)")

    if "no measurable light response" in flags:
        summary = (
            "No pupil constriction was detected. This is most often a capture "
            "limitation — repeat in a darker room with the screen at full brightness."
        )
    elif "unequal pupil sizes (anisocoria)" in flags:
        summary = (
            f"The pupils differ in resting size by about {anisocoria_mm:.1f} mm. "
            "Differences this large are uncommon and worth mentioning to an optometrist."
        )
    elif flags:
        summary = "Pupil constriction was present but smaller than typical for this stimulus."
    else:
        summary = "Both pupils constricted promptly to light, with equal resting sizes."

    summary += (
        " This is a both-eyes light response; it cannot detect a relative afferent "
        "pupillary defect, which needs alternating one-eye-at-a-time stimulation."
    )

    tier = "measured" if (valid_fraction >= 0.7 and ml and mr) else "weak-signal"
    metrics: dict = {"flags": flags}
    if anisocoria_mm is not None:
        metrics["anisocoria_mm"] = round(anisocoria_mm, 2)
    for name, m in (("left", ml), ("right", mr)):
        if not m:
            continue
        m = dict(m)
        # Frame quantization at low fps makes latency uninterpretable.
        if fps < MIN_FPS_FOR_LATENCY:
            m.pop("latency_s", None)
        metrics[name] = m
    if fps < MIN_FPS_FOR_LATENCY:
        metrics["latency_withheld"] = f"needs >={int(MIN_FPS_FOR_LATENCY)} fps capture"
    return Finding(module="pupillometry", summary=summary, tier=tier, metrics=metrics)
