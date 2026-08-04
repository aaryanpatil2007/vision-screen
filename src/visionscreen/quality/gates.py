from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from visionscreen.perception.eyes import interocular_px
from visionscreen.perception.landmarks import FaceFrame

NO_FACE = "No face detected — face the camera."
TOO_SMALL = "Move closer — your eyes are too small in frame."
TOO_DARK = "Lighting too dark — add light."
TOO_BRIGHT = "Lighting too bright — reduce glare."


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)


def check_frame(
    frame_bgr: np.ndarray,
    face: FaceFrame,
    min_eye_px: float = 60.0,
    brightness_range: tuple[float, float] = (30.0, 225.0),
) -> GateResult:
    failures: list[str] = []
    mean_brightness = float(frame_bgr.mean())
    if mean_brightness < brightness_range[0]:
        failures.append(TOO_DARK)
    elif mean_brightness > brightness_range[1]:
        failures.append(TOO_BRIGHT)
    if not face.ok:
        failures.append(NO_FACE)
    else:
        h, w = frame_bgr.shape[:2]
        if interocular_px(face.landmarks, w, h) < min_eye_px:
            failures.append(TOO_SMALL)
    return GateResult(passed=not failures, failures=failures)
