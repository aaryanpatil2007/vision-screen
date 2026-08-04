from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from visionscreen.modules.acuity import score_trials
from visionscreen.modules.alignment import (
    AlignmentFrame,
    pursuit_conjugacy,
    reflex_decentration_mm,
    score_alignment,
)
from visionscreen.modules.behavioral import analyze_series
from visionscreen.modules.photoref import measure_reflex, score_photoref
from visionscreen.perception.eyes import eye_aspect_ratio, head_roll_deg, interocular_px
from visionscreen.perception.iris import (
    detect_corneal_reflex,
    eye_crop,
    iris_center,
    iris_diameter_px,
)
from visionscreen.perception.landmarks import LandmarkExtractor
from visionscreen.protocol import SegmentMeta, SessionMeta
from visionscreen.quality.gates import check_frame
from visionscreen.report import Finding
from visionscreen.synth.eyes2d import HVID_MM

_EYE_CORNERS = {"left": (33, 133), "right": (362, 263)}
PHOTOREF_BRIGHTNESS = (5.0, 90.0)  # dim room required for the red reflex
PUPIL_TO_IRIS_DIAMETER = 0.35  # dim-light pupil ≈ 4 mm on an 11.7 mm iris


def _gaze_x(landmarks: np.ndarray, side: str) -> float | None:
    a, b = _EYE_CORNERS[side]
    ax, bx = landmarks[a, 0], landmarks[b, 0]
    lo, hi = sorted((ax, bx))
    if hi - lo < 1e-6:
        return None
    return float((iris_center(landmarks, side)[0] - lo) / (hi - lo))


def _eye_decentration(frame, landmarks, side: str) -> tuple[float, float] | None:
    h, w = frame.shape[:2]
    crop, (ox, oy) = eye_crop(frame, landmarks, side)
    if crop.size == 0:
        return None
    center_px = tuple(iris_center(landmarks, side) * (w, h))
    diameter = iris_diameter_px(landmarks, side, w, h)
    if diameter < 1e-6:
        return None
    reflex = detect_corneal_reflex(
        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
        center_xy=(center_px[0] - ox, center_px[1] - oy),
        radius_px=diameter / 2,
    )
    if reflex is None:
        return None
    reflex_px = (reflex[0] + ox, reflex[1] + oy)
    return reflex_decentration_mm(reflex_px, center_px, diameter)


def _photoref_frame(frame, landmarks, e_m: float, d_m: float) -> tuple[float, float, float] | None:
    """Measure one eye pair's reflex; returns the better-conditioned eye's estimate."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = []
    for side in ("left", "right"):
        center = tuple(iris_center(landmarks, side) * (w, h))
        iris_d = iris_diameter_px(landmarks, side, w, h)
        if iris_d < 8:
            continue
        px_per_m = iris_d / (HVID_MM / 1000.0)
        pupil_r = PUPIL_TO_IRIS_DIAMETER * iris_d
        est = measure_reflex(gray, center, pupil_r, e_m=e_m, d_m=d_m, px_per_m=px_per_m)
        if est is not None:
            results.append(est)
    if not results:
        return None
    return results[0] if len(results) == 1 else tuple(
        float(np.median([r[i] for r in results])) for i in range(3)
    )


def _dot_positions(segment: SegmentMeta, frame_ts: list[float]) -> list[float]:
    dots = [(ev.ts, ev.payload.get("x", 0.5)) for ev in segment.events if ev.kind == "dot"]
    if not dots:
        return []
    times = np.array([t for t, _ in dots])
    xs = np.array([x for _, x in dots])
    return [float(xs[int(np.argmin(np.abs(times - t)))]) for t in frame_ts]


def analyze_session(video_path: Path, meta: SessionMeta) -> list[Finding]:
    ears: list[float] = []
    interocular: list[float] = []
    rolls: list[float] = []
    total = 0

    align_seg = meta.segment("alignment")
    align_frames: list[AlignmentFrame] = []
    gaze_l: list[float] = []
    gaze_r: list[float] = []
    align_ts: list[float] = []
    align_total = 0

    pr_seg = meta.segment("photoref")
    pr_estimates: list[tuple[float, float, float]] = []
    pr_dead = 0
    pr_usable = 0
    pr_total = 0
    pr_cfg = {}
    if pr_seg is not None:
        for ev in pr_seg.events:
            if ev.kind == "photoref_config":
                pr_cfg = ev.payload
    pr_e = float(pr_cfg.get("e_m", 0.005))
    pr_d = float(pr_cfg.get("d_m", meta.distance_cm / 100.0))

    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    try:
        with LandmarkExtractor() as extractor:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ts = idx / meta.fps if meta.fps else 0.0
                idx += 1
                face = extractor.extract(frame)
                gate_ok = check_frame(frame, face).passed

                in_align = (
                    align_seg is not None
                    and align_seg.start_ts <= ts <= align_seg.end_ts
                )
                if in_align:
                    align_total += 1

                in_pr = (
                    pr_seg is not None and pr_seg.start_ts <= ts <= pr_seg.end_ts
                )
                if in_pr:
                    pr_total += 1
                    # photoref wants a DIM frame; run its own gate variant
                    if face.ok and check_frame(
                        frame, face, brightness_range=PHOTOREF_BRIGHTNESS
                    ).passed:
                        pr_usable += 1
                        est = _photoref_frame(frame, face.landmarks, pr_e, pr_d)
                        if est is None:
                            pr_dead += 1
                        else:
                            pr_estimates.append(est)
                    continue  # dim frames must not pollute the behavioral series

                total += 1
                if not gate_ok:
                    continue
                lm = face.landmarks
                ears.append(
                    (eye_aspect_ratio(lm, "left") + eye_aspect_ratio(lm, "right")) / 2
                )
                h, w = frame.shape[:2]
                interocular.append(interocular_px(lm, w, h))
                rolls.append(head_roll_deg(lm))

                if in_align:
                    dec_l = _eye_decentration(frame, lm, "left")
                    dec_r = _eye_decentration(frame, lm, "right")
                    gl, gr = _gaze_x(lm, "left"), _gaze_x(lm, "right")
                    if dec_l and dec_r and gl is not None and gr is not None:
                        align_frames.append(AlignmentFrame(dec_l, dec_r))
                        gaze_l.append(gl)
                        gaze_r.append(gr)
                        align_ts.append(ts)
    finally:
        cap.release()

    valid_fraction = (len(ears) / total) if total else 0.0
    behavioral = analyze_series(ears, interocular, rolls, valid_fraction)

    seg = meta.segment("acuity")
    trials = [ev.payload for ev in seg.events if ev.kind == "trial"] if seg else []
    acuity = score_trials(trials)

    if align_seg is None:
        alignment = Finding(
            module="alignment",
            summary="Alignment test was not performed.",
            tier="inconclusive",
            retakes=["Run the dot-following test segment."],
        )
    else:
        align_valid = (len(align_frames) / align_total) if align_total else 0.0
        pursuit = None
        dot_xs = _dot_positions(align_seg, align_ts)
        if dot_xs:
            pursuit = pursuit_conjugacy(gaze_l, gaze_r, dot_xs)
        alignment = score_alignment(align_frames, pursuit, align_valid)

    if pr_seg is None:
        photoref = Finding(
            module="photorefraction",
            summary="Photorefraction test was not performed.",
            tier="inconclusive",
            retakes=["Run the dim-room flash test segment."],
        )
    else:
        pr_valid = (pr_usable / pr_total) if pr_total else 0.0
        photoref = score_photoref(pr_estimates, pr_dead, pr_valid)

    return [acuity, behavioral, photoref, alignment]
