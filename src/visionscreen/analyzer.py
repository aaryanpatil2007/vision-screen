from __future__ import annotations

from pathlib import Path

import cv2

from visionscreen.modules.acuity import score_trials
from visionscreen.modules.behavioral import analyze_series
from visionscreen.perception.eyes import eye_aspect_ratio, head_roll_deg, interocular_px
from visionscreen.perception.landmarks import LandmarkExtractor
from visionscreen.protocol import SessionMeta
from visionscreen.quality.gates import check_frame
from visionscreen.report import Finding


def analyze_session(video_path: Path, meta: SessionMeta) -> list[Finding]:
    ears: list[float] = []
    interocular: list[float] = []
    rolls: list[float] = []
    total = 0

    cap = cv2.VideoCapture(str(video_path))
    try:
        with LandmarkExtractor() as extractor:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                total += 1
                face = extractor.extract(frame)
                if not check_frame(frame, face).passed:
                    continue
                lm = face.landmarks
                ears.append(
                    (eye_aspect_ratio(lm, "left") + eye_aspect_ratio(lm, "right")) / 2
                )
                h, w = frame.shape[:2]
                interocular.append(interocular_px(lm, w, h))
                rolls.append(head_roll_deg(lm))
    finally:
        cap.release()

    valid_fraction = (len(ears) / total) if total else 0.0
    behavioral = analyze_series(ears, interocular, rolls, valid_fraction)

    seg = meta.segment("acuity")
    trials = (
        [ev.payload for ev in seg.events if ev.kind == "trial"] if seg else []
    )
    acuity = score_trials(trials)

    return [acuity, behavioral]
