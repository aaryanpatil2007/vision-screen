"""Session analysis: video + protocol metadata -> screening findings.

One pass over the video extracts every per-frame perception signal, tagged by
which protocol segment it falls in; each module then scores its own segment.
Perception uses the learned segmenter when a checkpoint is available and falls
back to the classical detectors otherwise, so the pipeline degrades instead of
failing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from visionscreen.ml.infer import EyeSegmenter
from visionscreen.modules.acuity import score_trials
from visionscreen.modules.alignment import (
    AlignmentFrame,
    pursuit_conjugacy,
    reflex_decentration_mm,
    score_alignment,
)
from visionscreen.modules.amsler import score_amsler
from visionscreen.modules.astigmatic import score_astigmatic_dial
from visionscreen.modules.behavioral import analyze_series
from visionscreen.modules.colorvision import score_color_vision
from visionscreen.modules.contrast import score_contrast
from visionscreen.modules.motility import score_motility
from visionscreen.modules.photoref import measure_reflex, score_photoref
from visionscreen.modules.pupillometry import PupilTrace, score_pupillometry
from visionscreen.modules.stereo import score_stereo
from visionscreen.modules.suppression import score_suppression
from visionscreen.perception.distance import (
    acuity_bias_logmar,
    distance_from_iris,
    estimate_focal_px_from_iris,
    score_distance_stability,
)
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
PHOTOREF_BRIGHTNESS = (5.0, 90.0)
PUPIL_TO_IRIS_DIAMETER = 0.35
ACUITY_SEGMENTS = ("acuity_both", "acuity_right", "acuity_left", "acuity")


@dataclass
class FrameSignals:
    """Everything one usable video frame contributes."""
    ts: float
    ear: float
    interocular: float
    iris_px: float
    roll: float
    gaze: dict[str, float] = field(default_factory=dict)
    decentration: dict[str, tuple[float, float]] = field(default_factory=dict)
    pupil_mm: dict[str, float] = field(default_factory=dict)


def _gaze_x(landmarks: np.ndarray, side: str) -> float | None:
    a, b = _EYE_CORNERS[side]
    lo, hi = sorted((landmarks[a, 0], landmarks[b, 0]))
    if hi - lo < 1e-6:
        return None
    return float((iris_center(landmarks, side)[0] - lo) / (hi - lo))


def _eye_measurements(frame, landmarks, side, segmenter: EyeSegmenter | None):
    """Reflex decentration (mm) and pupil diameter (mm) for one eye.

    Learned segmentation first (robust on real webcam frames); classical
    threshold detection as fallback so a missing checkpoint never breaks the run.
    """
    h, w = frame.shape[:2]
    crop_bgr, (ox, oy) = eye_crop(frame, landmarks, side)
    if crop_bgr.size == 0:
        return None, None
    center_px = tuple(iris_center(landmarks, side) * (w, h))
    iris_d = iris_diameter_px(landmarks, side, w, h)
    if iris_d < 1e-6:
        return None, None
    px_per_mm = iris_d / HVID_MM
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    seg = segmenter.segment(gray) if (segmenter and segmenter.available) else None

    pupil_mm = None
    reflex_local = None
    ref_center = center_px
    if seg is not None:
        # The net is the only source of a pupil boundary; the classical path
        # has no equivalent.
        pupil_mm = (2 * seg.pupil_radius_px / px_per_mm) if seg.pupil_radius_px else None
        if seg.iris_center is not None:
            ref_center = (seg.iris_center[0] + ox, seg.iris_center[1] + oy)
        reflex_local = seg.reflex_center

    # Reflex: prefer the net, but fall back to threshold detection when it
    # finds none. Measured on reflex-bearing real crops, classical recall is
    # ~99% against ~77% for the net (the specular highlight is a strong,
    # simple photometric signal), so trusting the net alone silently discards
    # about a fifth of the alignment data.
    if reflex_local is None:
        reflex_local = detect_corneal_reflex(
            gray,
            center_xy=(ref_center[0] - ox, ref_center[1] - oy),
            radius_px=iris_d / 2,
        )
    if reflex_local is None:
        return None, pupil_mm

    reflex_px = (reflex_local[0] + ox, reflex_local[1] + oy)
    return reflex_decentration_mm(reflex_px, ref_center, iris_d), pupil_mm


def _photoref_frame(frame, landmarks, e_m, d_m):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = []
    for side in ("left", "right"):
        center = tuple(iris_center(landmarks, side) * (w, h))
        iris_d = iris_diameter_px(landmarks, side, w, h)
        if iris_d < 8:
            continue
        px_per_m = iris_d / (HVID_MM / 1000.0)
        est = measure_reflex(gray, center, PUPIL_TO_IRIS_DIAMETER * iris_d,
                             e_m=e_m, d_m=d_m, px_per_m=px_per_m)
        if est is not None:
            results.append(est)
    if not results:
        return None
    if len(results) == 1:
        return results[0]
    return tuple(float(np.median([r[i] for r in results])) for i in range(3))


def _resample(segment: SegmentMeta, kind: str, key: str, frame_ts: list[float]) -> list[float]:
    pts = [(ev.ts, ev.payload.get(key)) for ev in segment.events if ev.kind == kind]
    pts = [(t, v) for t, v in pts if v is not None]
    if not pts or not frame_ts:
        return []
    times = np.array([t for t, _ in pts])
    vals = np.array([v for _, v in pts], dtype=float)
    return [float(vals[int(np.argmin(np.abs(times - t)))]) for t in frame_ts]


def _in(seg: SegmentMeta | None, ts: float) -> bool:
    return seg is not None and seg.start_ts <= ts <= (seg.end_ts or float("inf"))


def analyze_session(video_path: Path, meta: SessionMeta,
                    segmenter: EyeSegmenter | None = None) -> list[Finding]:
    if segmenter is None:
        segmenter = EyeSegmenter()

    seg_photoref = meta.segment("photoref")
    seg_motility = meta.segment("motility") or meta.segment("alignment")
    seg_pupil = meta.segment("pupil")

    frames: list[FrameSignals] = []
    counts = {"total": 0, "gated": 0}
    per_segment_total: dict[str, int] = {}
    pr_estimates: list[tuple[float, float, float]] = []
    pr_dead = pr_usable = pr_total = 0

    pr_cfg = {}
    if seg_photoref is not None:
        for ev in seg_photoref.events:
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

                for s in meta.segments:
                    if _in(s, ts):
                        per_segment_total[s.test_id] = per_segment_total.get(s.test_id, 0) + 1

                # photorefraction has its own (dark-room) gate
                if _in(seg_photoref, ts):
                    pr_total += 1
                    if face.ok and check_frame(
                        frame, face, brightness_range=PHOTOREF_BRIGHTNESS
                    ).passed:
                        pr_usable += 1
                        est = _photoref_frame(frame, face.landmarks, pr_e, pr_d)
                        if est is None:
                            pr_dead += 1
                        else:
                            pr_estimates.append(est)
                    continue

                counts["total"] += 1
                if not check_frame(frame, face).passed:
                    continue
                counts["gated"] += 1

                lm = face.landmarks
                h, w = frame.shape[:2]
                fs = FrameSignals(
                    ts=ts,
                    ear=(eye_aspect_ratio(lm, "left") + eye_aspect_ratio(lm, "right")) / 2,
                    interocular=interocular_px(lm, w, h),
                    # max of the two irides: the eye turned toward the camera is
                    # the least foreshortened, so this is the yaw-robust estimate
                    iris_px=max(iris_diameter_px(lm, "left", w, h),
                                iris_diameter_px(lm, "right", w, h)),
                    roll=head_roll_deg(lm),
                )
                for side in ("left", "right"):
                    g = _gaze_x(lm, side)
                    if g is not None:
                        fs.gaze[side] = g
                need_eyes = _in(seg_motility, ts) or _in(seg_pupil, ts)
                if need_eyes:
                    for side in ("left", "right"):
                        dec, pupil_mm = _eye_measurements(frame, lm, side, segmenter)
                        if dec is not None:
                            fs.decentration[side] = dec
                        if pupil_mm is not None:
                            fs.pupil_mm[side] = pupil_mm
                frames.append(fs)
    finally:
        cap.release()

    valid_fraction = counts["gated"] / counts["total"] if counts["total"] else 0.0
    findings: list[Finding] = []

    # ---- objective viewing distance ----
    # Calibrate the pinhole focal length from the MEDIAN early-session span
    # against the stated distance, then track distance per frame. This cannot
    # detect a constant offset in the user's estimate (that needs a physical
    # reference), but it does catch drift and within-session movement, which
    # are the errors that vary across the battery.
    nominal_mm = meta.distance_cm * 10.0
    distances_mm: list[float] = []
    if frames:
        warmup = [f.iris_px for f in frames[: max(5, len(frames) // 10)] if f.iris_px > 0]
        focal = estimate_focal_px_from_iris(float(np.median(warmup)), nominal_mm) if warmup else None
        if focal:
            distances_mm = [
                d for f in frames
                if (d := distance_from_iris(f.iris_px, focal)) is not None
            ]
    if len(distances_mm) >= 10:
        findings.append(score_distance_stability(distances_mm, nominal_mm))

    # ---- acuity (binocular + each eye) ----
    ts_to_distance = dict(zip((f.ts for f in frames), distances_mm)) if distances_mm else {}
    for test_id, label in (
        ("acuity_both", "both eyes"),
        ("acuity_right", "right eye"),
        ("acuity_left", "left eye"),
        ("acuity", "acuity"),
    ):
        s = meta.segment(test_id)
        if s is None:
            continue
        trials = [ev.payload for ev in s.events if ev.kind == "trial"]
        f = score_trials(trials)
        if test_id != "acuity":
            f.module = f"Acuity ({label})"

        # Correct for where the user actually sat during THIS test. Optotypes
        # were sized for the stated distance; testing nearer or farther shifts
        # the threshold by log10(assumed/actual).
        seg_d = [d for t, d in ts_to_distance.items() if _in(s, t)]
        if seg_d and f.metrics.get("logmar") is not None:
            actual = float(np.median(seg_d))
            bias = acuity_bias_logmar(actual, nominal_mm)
            if abs(bias) >= 0.05:
                raw = f.metrics["logmar"]
                f.metrics["logmar_uncorrected"] = raw
                f.metrics["logmar"] = round(raw + bias, 2)
                f.metrics["distance_correction_logmar"] = round(bias, 3)
                f.metrics["measured_distance_cm"] = round(actual / 10, 1)
                f.summary = (
                    f"Estimated acuity {f.metrics['logmar']:.2f} logMAR "
                    f"(corrected for a measured viewing distance of "
                    f"{actual/10:.0f} cm)."
                )
        findings.append(f)
    if not any(f.module.lower().startswith("acuity") for f in findings):
        findings.append(Finding(
            module="acuity", summary="Acuity test was not performed.",
            tier="inconclusive", retakes=["Run the letter test."]))

    # ---- behavioral ----
    findings.append(analyze_series(
        [f.ear for f in frames],
        [f.interocular for f in frames],
        [f.roll for f in frames],
        valid_fraction,
    ))

    # ---- contrast ----
    s = meta.segment("contrast")
    if s is not None:
        trials = [{"log_cs": ev.payload["log_cs"], "correct": ev.payload["correct"]}
                  for ev in s.events if ev.kind == "trial"]
        findings.append(score_contrast(trials, valid_fraction))

    # ---- astigmatism ----
    s = meta.segment("astigmatism")
    if s is not None:
        dials = [ev.payload for ev in s.events if ev.kind == "dial"]
        angles = [d["dark_meridian_deg"] for d in dials if d.get("dark_meridian_deg") is not None]
        no_pref = bool(dials) and all(d.get("no_preference") for d in dials)
        findings.append(score_astigmatic_dial(angles, valid_fraction, no_pref))

    # ---- color vision ----
    s = meta.segment("color_vision")
    if s is not None:
        answers = {ev.payload["id"]: ev.payload.get("answered")
                   for ev in s.events if ev.kind == "plate"}
        findings.append(score_color_vision(answers, valid_fraction))

    # ---- Amsler ----
    s = meta.segment("amsler")
    if s is not None:
        marks = [ev.payload for ev in s.events if ev.kind == "amsler"]
        distortions = [m for e in marks for m in e.get("marks", [])]
        findings.append(score_amsler(distortions, [], len(marks), valid_fraction))

    # ---- stereo ----
    s = meta.segment("stereo")
    if s is not None:
        trials = [{"arcsec": ev.payload["arcsec"], "correct": ev.payload["correct"]}
                  for ev in s.events if ev.kind == "trial"]
        catch = [{"correct": ev.payload["correct"]}
                 for ev in s.events if ev.kind == "catch"]
        floor = next((ev.payload.get("display_floor_arcsec") for ev in s.events
                      if ev.kind == "stereo_config"), None)
        findings.append(score_stereo(trials, catch, valid_fraction,
                                     display_floor_arcsec=floor))

    # ---- binocular fusion / suppression ----
    s = meta.segment("suppression")
    if s is not None:
        responses = {ev.payload["distance"]: ev.payload["response"]
                     for ev in s.events if ev.kind == "worth"}
        findings.append(score_suppression(responses, valid_fraction))

    # ---- motility + alignment (share the pursuit segment) ----
    if seg_motility is not None:
        mframes = [f for f in frames if _in(seg_motility, f.ts)]
        mts = [f.ts for f in mframes]
        target = _resample(seg_motility, "dot", "x", mts)
        gl = [f.gaze.get("left") for f in mframes]
        gr = [f.gaze.get("right") for f in mframes]
        ok = [i for i, (a, b) in enumerate(zip(gl, gr)) if a is not None and b is not None]
        gl2 = [gl[i] for i in ok]
        gr2 = [gr[i] for i in ok]
        ts2 = [mts[i] for i in ok]
        tgt2 = [target[i] for i in ok] if target else []
        seg_total = per_segment_total.get(seg_motility.test_id, 0)
        mvalid = len(ok) / seg_total if seg_total else 0.0

        if tgt2:
            findings.append(score_motility(gl2, gr2, tgt2, ts2, mvalid))
            pursuit = pursuit_conjugacy(gl2, gr2, tgt2)
        else:
            pursuit = None

        align_frames = [
            AlignmentFrame(f.decentration["left"], f.decentration["right"])
            for f in mframes if "left" in f.decentration and "right" in f.decentration
        ]
        avalid = (len(align_frames) / seg_total) if (align_frames and seg_total) else mvalid
        findings.append(score_alignment(align_frames, pursuit, avalid))

    # ---- pupillometry ----
    if seg_pupil is not None:
        flashes = [ev.ts for ev in seg_pupil.events if ev.kind == "flash_on"]
        pframes = [f for f in frames if _in(seg_pupil, f.ts)]
        seg_total = per_segment_total.get("pupil", 0)
        traces = {}
        if flashes and pframes:
            for side in ("left", "right"):
                ts_l = [f.ts for f in pframes if side in f.pupil_mm]
                d_l = [f.pupil_mm[side] for f in pframes if side in f.pupil_mm]
                if len(ts_l) >= 20:
                    traces[side] = PupilTrace(ts=ts_l, diameter_mm=d_l, flash_ts=flashes[0])
        pvalid = (len(pframes) / seg_total) if seg_total else 0.0
        findings.append(score_pupillometry(
            traces.get("left"), traces.get("right"), pvalid, fps=meta.fps))

    # ---- photorefraction ----
    if seg_photoref is not None:
        pr_valid = (pr_usable / pr_total) if pr_total else 0.0
        findings.append(score_photoref(pr_estimates, pr_dead, pr_valid))

    return findings
