"""Ocular motility: smooth-pursuit gain and saccade detection.

Clinically, the extraocular exam asks whether the eyes follow a target
smoothly, together, and to full excursion. Two quantities capture most of the
screening value:

* **Pursuit gain** — eye velocity divided by target velocity. Healthy adults
  track slow targets at gain ~0.9-1.0; substantially reduced gain produces
  catch-up saccades and is seen in neurological disease, medication effects,
  and inattention.
* **Catch-up saccades** — fast steps interrupting smooth tracking, detected
  from velocity peaks.

Gain is estimated by regressing eye position on the time-shifted target,
which is robust to the amplitude normalization of webcam-derived gaze.
"""
from __future__ import annotations

import numpy as np

from visionscreen.report import Finding

MIN_SAMPLES = 30
LOW_GAIN = 0.60
ASYMMETRY_RATIO = 0.55
SACCADE_VELOCITY_SD = 4.0     # velocity peak, in SDs above median absolute velocity
MAX_LAG_S = 0.5


def _velocity(sig: list[float], ts: list[float]) -> np.ndarray:
    s, t = np.asarray(sig, float), np.asarray(ts, float)
    dt = np.gradient(t)
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    return np.gradient(s) / dt


def pursuit_gain(eye: list[float], target: list[float], ts: list[float]) -> dict:
    n = min(len(eye), len(target), len(ts))
    e = np.asarray(eye[:n], float)
    g = np.asarray(target[:n], float)
    t = np.asarray(ts[:n], float)
    if n < MIN_SAMPLES or g.std() < 1e-6:
        return {"gain": float("nan"), "lag_s": float("nan"), "r": float("nan")}

    fps = 1.0 / max(np.median(np.diff(t)), 1e-6)
    max_shift = int(MAX_LAG_S * fps)
    best = {"gain": float("nan"), "lag_s": 0.0, "r": -2.0}
    for shift in range(0, max_shift + 1):
        a = e[shift:] if shift else e
        b = g[: n - shift] if shift else g
        if len(a) < MIN_SAMPLES or b.std() < 1e-9 or a.std() < 1e-9:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if r > best["r"]:
            slope = float(np.polyfit(b, a, 1)[0])
            best = {"gain": abs(slope), "lag_s": shift / fps, "r": r}
    return best


def detect_saccades(eye: list[float], ts: list[float]) -> list[dict]:
    n = min(len(eye), len(ts))
    if n < MIN_SAMPLES:
        return []
    v = _velocity(eye[:n], ts[:n])
    av = np.abs(v)
    med = float(np.median(av))
    mad = float(np.median(np.abs(av - med))) or 1e-6
    thresh = med + SACCADE_VELOCITY_SD * 1.4826 * mad
    above = av > max(thresh, 1e-6)

    events, i = [], 0
    e = np.asarray(eye[:n], float)
    t = np.asarray(ts[:n], float)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and above[j + 1]:
            j += 1
        lo = max(0, i - 1)
        hi = min(n - 1, j + 1)
        amp = abs(e[hi] - e[lo])
        if amp > 1e-3:
            events.append({
                "t": float(t[lo]),
                "amplitude": float(amp),
                "peak_velocity": float(av[i:j + 1].max()),
                "duration_s": float(t[hi] - t[lo]),
            })
        i = j + 1
    return events


def score_motility(
    gaze_left: list[float],
    gaze_right: list[float],
    target: list[float],
    ts: list[float],
    valid_fraction: float,
) -> Finding:
    n = min(len(gaze_left), len(gaze_right), len(target), len(ts))
    if n < MIN_SAMPLES or valid_fraction < 0.4:
        return Finding(
            module="motility",
            summary="Eye-movement recording was too short or unusable to assess.",
            tier="inconclusive",
            retakes=["Repeat the moving-target test, following the dot with your eyes only."],
        )

    gl = pursuit_gain(gaze_left, target, ts)
    gr = pursuit_gain(gaze_right, target, ts)
    sl = detect_saccades(gaze_left, ts)
    sr = detect_saccades(gaze_right, ts)

    gains = [g["gain"] for g in (gl, gr) if np.isfinite(g["gain"])]
    if not gains:
        return Finding(
            module="motility",
            summary="Eye position could not be tracked well enough to measure pursuit.",
            tier="inconclusive",
            retakes=["Repeat the moving-target test with better lighting on your face."],
        )

    flags: list[str] = []
    mean_gain = float(np.mean(gains))
    if mean_gain < LOW_GAIN:
        flags.append("reduced smooth-pursuit gain")
    if len(gains) == 2:
        lo, hi = sorted(gains)
        if hi > 1e-6 and lo / hi < ASYMMETRY_RATIO:
            flags.append("asymmetric eye movement")

    saccade_rate = (len(sl) + len(sr)) / 2 / max(ts[n - 1] - ts[0], 1e-6)
    if saccade_rate > 2.0 and mean_gain < 0.85:
        flags.append("frequent catch-up saccades")

    if flags:
        summary = "Eye-movement findings: " + ", ".join(flags) + "."
    else:
        summary = (
            f"Both eyes followed the moving target smoothly "
            f"(pursuit gain {mean_gain:.2f}, symmetric)."
        )
    tier = "measured" if valid_fraction >= 0.7 else "weak-signal"
    return Finding(
        module="motility",
        summary=summary,
        tier=tier,
        metrics={
            "flags": flags,
            "gain_left": round(gl["gain"], 3) if np.isfinite(gl["gain"]) else None,
            "gain_right": round(gr["gain"], 3) if np.isfinite(gr["gain"]) else None,
            "lag_left_s": round(gl["lag_s"], 3) if np.isfinite(gl["lag_s"]) else None,
            "saccades_per_s": round(saccade_rate, 2),
        },
    )
