from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

N_LANDMARKS = 478  # FaceLandmarker model output (includes iris points)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def _model_path() -> Path:
    cache_dir = Path(
        os.environ.get("VISIONSCREEN_MODEL_DIR", Path.home() / ".cache" / "visionscreen")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "face_landmarker.task"
    if not path.exists():
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


@dataclass
class FaceFrame:
    landmarks: np.ndarray  # (478, 3) float32, normalized image coords
    ok: bool


class LandmarkExtractor:
    def __init__(self) -> None:
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_model_path())),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def extract(self, frame_bgr: np.ndarray) -> FaceFrame:
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect(image)
        if not res.face_landmarks:
            return FaceFrame(np.zeros((N_LANDMARKS, 3), np.float32), ok=False)
        pts = res.face_landmarks[0]
        arr = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)
        return FaceFrame(arr, ok=True)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "LandmarkExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
