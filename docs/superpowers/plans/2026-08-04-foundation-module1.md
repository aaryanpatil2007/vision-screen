# Vision Screening — Foundation + Module 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pipeline foundation (protocol schema, perception layer, quality gates, report) plus Module 1 (behavioral + acuity screening), ending with a browser demo and a benchmark harness that emits a results table.

**Architecture:** Physics-spine pipeline per the spec (`docs/superpowers/specs/2026-08-04-vision-screening-pipeline-design.md`). Perception = MediaPipe FaceMesh landmarks → geometric eye metrics. Module 1 scores an on-screen tumbling-E staircase test and behavioral signals (squint, lean-in, head tilt). Every module output carries a confidence tier; quality gates block garbage estimates. This plan is Plan 1 of 3 — Module 2 (alignment) and Module 3 (photorefraction + synthetic renderer) build on these interfaces in later plans.

**Tech Stack:** Python 3.11, MediaPipe (FaceMesh, `refine_landmarks=True` for iris), OpenCV, NumPy, FastAPI + vanilla JS front-end, pytest. (PyTorch enters in Plan 2/3, not here.)

## Global Constraints

- Python 3.11; dependencies pinned in `pyproject.toml`: `mediapipe>=0.10`, `opencv-python>=4.9`, `numpy>=1.26`, `fastapi>=0.110`, `uvicorn>=0.29`, `python-multipart>=0.0.9`; dev: `pytest>=8`, `httpx>=0.27`, `scikit-image>=0.23` (test face image only).
- Confidence tiers are exactly the strings `"measured"`, `"weak-signal"`, `"inconclusive"` (type `Tier` in `report.py`). A module that cannot measure returns `"inconclusive"` — never a fabricated estimate.
- Every rendered report MUST contain the exact copy: `This is a screening signal only, not a diagnosis. See an optometrist for a clinical evaluation.`
- Quality-gate failures produce retake instructions, never estimates.
- All work happens in `~/vision-screen`; commit after every task.
- Run tests with `python -m pytest` from the repo root.

## File Structure

```
vision-screen/
  pyproject.toml
  src/visionscreen/
    __init__.py
    protocol.py            # session/segment/event schema + JSON round-trip
    perception/
      __init__.py
      landmarks.py         # MediaPipe FaceMesh wrapper → FaceFrame
      eyes.py              # EAR, interocular px, head roll (pure geometry)
    quality/
      __init__.py
      gates.py             # per-frame capture-quality gates
    modules/
      __init__.py
      acuity.py            # tumbling-E sizing + staircase + scoring
      behavioral.py        # squint / lean-in / tilt flags from time series
    report.py              # Finding + Tier + HTML rendering
    analyzer.py            # video + SessionMeta → list[Finding]
  webapp/
    app.py                 # FastAPI: serves UI, POST /analyze
    static/index.html      # guided protocol UI (tumbling-E, MediaRecorder)
  benchmarks/
    bench_module1.py       # simulated observers → results table
  tests/                   # mirrors src layout
```

---

### Task 1: Project scaffold + protocol schema

**Files:**
- Create: `pyproject.toml`, `src/visionscreen/__init__.py`, `src/visionscreen/protocol.py`, `tests/__init__.py` (empty — later tasks import helpers across test modules, e.g. `from tests.test_eyes import synth_landmarks`)
- Test: `tests/test_protocol.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `ScreenEvent(ts: float, kind: str, payload: dict)`, `SegmentMeta(test_id: str, start_ts: float, end_ts: float, events: list[ScreenEvent])`, `SessionMeta(session_id: str, px_per_cm: float, distance_cm: float, fps: float, segments: list[SegmentMeta])` with `SessionMeta.to_json() -> str` and `SessionMeta.from_json(s: str) -> SessionMeta`. All dataclasses.

- [ ] **Step 1: Write pyproject and failing test**

`pyproject.toml`:

```toml
[project]
name = "visionscreen"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "mediapipe>=0.10",
    "opencv-python>=4.9",
    "numpy>=1.26",
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "python-multipart>=0.0.9",
]

[project.optional-dependencies]
dev = ["pytest>=8", "httpx>=0.27", "scikit-image>=0.23"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`tests/test_protocol.py`:

```python
from visionscreen.protocol import ScreenEvent, SegmentMeta, SessionMeta


def make_session() -> SessionMeta:
    ev = ScreenEvent(ts=1.5, kind="trial", payload={"logmar": 0.5, "shown": "up", "answered": "up"})
    seg = SegmentMeta(test_id="acuity", start_ts=0.0, end_ts=30.0, events=[ev])
    return SessionMeta(session_id="s1", px_per_cm=37.8, distance_cm=50.0, fps=30.0, segments=[seg])


def test_json_round_trip():
    s = make_session()
    restored = SessionMeta.from_json(s.to_json())
    assert restored == s
    assert restored.segments[0].events[0].payload["shown"] == "up"


def test_segment_lookup():
    s = make_session()
    assert s.segment("acuity").test_id == "acuity"
    assert s.segment("missing") is None
```

- [ ] **Step 2: Create venv, install, verify test fails**

```bash
cd ~/vision-screen
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/test_protocol.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'visionscreen.protocol'` (create `src/visionscreen/__init__.py` empty first if the editable install complains).

- [ ] **Step 3: Implement `src/visionscreen/protocol.py`**

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ScreenEvent:
    ts: float
    kind: str
    payload: dict


@dataclass(frozen=True)
class SegmentMeta:
    test_id: str
    start_ts: float
    end_ts: float
    events: list[ScreenEvent] = field(default_factory=list)


@dataclass(frozen=True)
class SessionMeta:
    session_id: str
    px_per_cm: float
    distance_cm: float
    fps: float
    segments: list[SegmentMeta] = field(default_factory=list)

    def segment(self, test_id: str) -> SegmentMeta | None:
        return next((s for s in self.segments if s.test_id == test_id), None)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> SessionMeta:
        d = json.loads(s)
        segments = [
            SegmentMeta(
                test_id=seg["test_id"],
                start_ts=seg["start_ts"],
                end_ts=seg["end_ts"],
                events=[ScreenEvent(**ev) for ev in seg["events"]],
            )
            for seg in d.pop("segments")
        ]
        return cls(segments=segments, **d)
```

- [ ] **Step 4: Run test, verify pass**

Run: `python -m pytest tests/test_protocol.py -v` — Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src tests && git commit -m "feat: project scaffold + session protocol schema"
```

---

### Task 2: Landmark extractor (MediaPipe wrapper)

**Files:**
- Create: `src/visionscreen/perception/__init__.py`, `src/visionscreen/perception/landmarks.py`
- Test: `tests/test_landmarks.py`

**Interfaces:**
- Consumes: nothing
- Produces: `FaceFrame(landmarks: np.ndarray, ok: bool)` where `landmarks` is float32 `(478, 3)` in normalized image coords; `LandmarkExtractor` with `extract(frame_bgr: np.ndarray) -> FaceFrame` (`ok=False`, zero landmarks when no face) and context-manager `close()`.

- [ ] **Step 1: Write the failing test**

`tests/test_landmarks.py`:

```python
import numpy as np
import pytest

from visionscreen.perception.landmarks import FaceFrame, LandmarkExtractor


def test_no_face_returns_not_ok():
    with LandmarkExtractor() as ex:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        ff = ex.extract(blank)
    assert isinstance(ff, FaceFrame)
    assert ff.ok is False
    assert ff.landmarks.shape == (478, 3)


def test_real_face_detected():
    skimage = pytest.importorskip("skimage")
    from skimage import data

    face_rgb = data.astronaut()  # public-domain photo containing a real face
    face_bgr = face_rgb[:, :, ::-1].copy()
    with LandmarkExtractor() as ex:
        ff = ex.extract(face_bgr)
    assert ff.ok is True
    # normalized coords in [0, 1] for a centered face
    assert 0.0 < ff.landmarks[:, 0].mean() < 1.0
    assert 0.0 < ff.landmarks[:, 1].mean() < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_landmarks.py -v`
Expected: FAIL with `ModuleNotFoundError` on `visionscreen.perception.landmarks`.

- [ ] **Step 3: Implement `landmarks.py` (and empty `perception/__init__.py`)**

```python
from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

N_LANDMARKS = 478  # FaceMesh with refine_landmarks=True (includes iris points)


@dataclass
class FaceFrame:
    landmarks: np.ndarray  # (478, 3) float32, normalized image coords
    ok: bool


class LandmarkExtractor:
    def __init__(self) -> None:
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def extract(self, frame_bgr: np.ndarray) -> FaceFrame:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._mesh.process(rgb)
        if not res.multi_face_landmarks:
            return FaceFrame(np.zeros((N_LANDMARKS, 3), np.float32), ok=False)
        pts = res.multi_face_landmarks[0].landmark
        arr = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)
        return FaceFrame(arr, ok=True)

    def close(self) -> None:
        self._mesh.close()

    def __enter__(self) -> "LandmarkExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_landmarks.py -v`
Expected: 2 PASS. (If MediaPipe fails to detect the astronaut face, first check the BGR/RGB conversion — that is the usual culprit.)

- [ ] **Step 5: Commit**

```bash
git add src/visionscreen/perception tests/test_landmarks.py && git commit -m "feat: MediaPipe landmark extractor"
```

---

### Task 3: Eye geometry metrics

**Files:**
- Create: `src/visionscreen/perception/eyes.py`
- Test: `tests/test_eyes.py`

**Interfaces:**
- Consumes: `FaceFrame.landmarks` layout from Task 2 (FaceMesh indexing).
- Produces: `eye_aspect_ratio(landmarks, side: str) -> float` (side `"left"`/`"right"`), `interocular_px(landmarks, frame_w: int, frame_h: int) -> float`, `head_roll_deg(landmarks) -> float`. Pure functions on `(478,3)` arrays.

FaceMesh index cheatsheet used here (subject's left eye appears on image right): left eye horizontal corners 33/133, vertical pairs (159,145),(158,153); right eye corners 362/263, vertical pairs (386,374),(385,380). Outer corners for roll: 33 and 263.

- [ ] **Step 1: Write the failing test**

`tests/test_eyes.py`:

```python
import numpy as np
import pytest

from visionscreen.perception.eyes import eye_aspect_ratio, head_roll_deg, interocular_px

L = {"h": (33, 133), "v": [(159, 145), (158, 153)]}
R = {"h": (362, 263), "v": [(386, 374), (385, 380)]}


def synth_landmarks(open_h: float = 0.02, roll_rad: float = 0.0) -> np.ndarray:
    """478 landmarks with two synthetic eyes of controllable opening and roll.
    Roll rotates the whole face about the global center so measured roll == roll_rad."""
    lm = np.zeros((478, 3), np.float32)
    centers = {"L": np.array([0.35, 0.5]), "R": np.array([0.65, 0.5])}
    global_c = np.array([0.5, 0.5])
    rot = np.array(
        [[np.cos(roll_rad), -np.sin(roll_rad)], [np.sin(roll_rad), np.cos(roll_rad)]]
    )

    def put(idx, offset, center):
        p = center + np.array(offset, np.float32) - global_c
        lm[idx, :2] = global_c + rot @ p

    for key, spec in (("L", L), ("R", R)):
        c = centers[key]
        put(spec["h"][0], (-0.04, 0.0), c)
        put(spec["h"][1], (0.04, 0.0), c)
        for (top, bot), dx in zip(spec["v"], (-0.01, 0.01)):
            put(top, (dx, -open_h / 2), c)
            put(bot, (dx, open_h / 2), c)
    return lm


def test_ear_drops_when_eye_closes():
    open_ear = eye_aspect_ratio(synth_landmarks(open_h=0.02), "left")
    squint_ear = eye_aspect_ratio(synth_landmarks(open_h=0.006), "left")
    assert open_ear == pytest.approx(0.25, abs=0.02)
    assert squint_ear < open_ear / 2


def test_interocular_scales_with_frame_width():
    lm = synth_landmarks()
    # outer corners: 33 at x=0.31, 263 at x=0.69 → 0.38 of frame width
    assert interocular_px(lm, 640, 480) == pytest.approx(0.38 * 640, rel=0.05)


def test_head_roll_recovered():
    assert head_roll_deg(synth_landmarks()) == pytest.approx(0.0, abs=0.5)
    assert head_roll_deg(synth_landmarks(roll_rad=np.radians(10))) == pytest.approx(10.0, abs=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eyes.py -v` — Expected: FAIL, module not found.

- [ ] **Step 3: Implement `eyes.py`**

```python
from __future__ import annotations

import numpy as np

_EYES = {
    "left": {"h": (33, 133), "v": [(159, 145), (158, 153)]},
    "right": {"h": (362, 263), "v": [(386, 374), (385, 380)]},
}
_OUTER_CORNERS = (33, 263)


def eye_aspect_ratio(landmarks: np.ndarray, side: str) -> float:
    spec = _EYES[side]
    h = np.linalg.norm(landmarks[spec["h"][0], :2] - landmarks[spec["h"][1], :2])
    if h == 0:
        return 0.0
    v = np.mean(
        [np.linalg.norm(landmarks[t, :2] - landmarks[b, :2]) for t, b in spec["v"]]
    )
    return float(v / h)


def interocular_px(landmarks: np.ndarray, frame_w: int, frame_h: int) -> float:
    a = landmarks[_OUTER_CORNERS[0], :2] * (frame_w, frame_h)
    b = landmarks[_OUTER_CORNERS[1], :2] * (frame_w, frame_h)
    return float(np.linalg.norm(a - b))


def head_roll_deg(landmarks: np.ndarray) -> float:
    a, b = landmarks[_OUTER_CORNERS[0], :2], landmarks[_OUTER_CORNERS[1], :2]
    dx, dy = b[0] - a[0], b[1] - a[1]
    return float(np.degrees(np.arctan2(dy, dx)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eyes.py -v` — Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visionscreen/perception/eyes.py tests/test_eyes.py && git commit -m "feat: eye geometry metrics (EAR, interocular, roll)"
```

---

### Task 4: Capture-quality gates

**Files:**
- Create: `src/visionscreen/quality/__init__.py`, `src/visionscreen/quality/gates.py`
- Test: `tests/test_gates.py`

**Interfaces:**
- Consumes: `FaceFrame` (Task 2), `interocular_px` (Task 3).
- Produces: `GateResult(passed: bool, failures: list[str])`; `check_frame(frame_bgr, face: FaceFrame, min_eye_px: float = 60.0, brightness_range: tuple = (30.0, 225.0)) -> GateResult`. Failure strings are exact retake instructions (used verbatim in the report/UI): `"No face detected — face the camera."`, `"Move closer — your eyes are too small in frame."`, `"Lighting too dark — add light."`, `"Lighting too bright — reduce glare."`

- [ ] **Step 1: Write the failing test**

`tests/test_gates.py`:

```python
import numpy as np

from visionscreen.perception.landmarks import FaceFrame
from visionscreen.quality.gates import check_frame

from tests.test_eyes import synth_landmarks


def frame(brightness: int) -> np.ndarray:
    return np.full((480, 640, 3), brightness, dtype=np.uint8)


def test_no_face_fails():
    ff = FaceFrame(np.zeros((478, 3), np.float32), ok=False)
    r = check_frame(frame(128), ff)
    assert not r.passed
    assert "No face detected — face the camera." in r.failures


def test_good_frame_passes():
    ff = FaceFrame(synth_landmarks(), ok=True)  # interocular ≈ 192 px at 640 wide
    r = check_frame(frame(128), ff)
    assert r.passed and r.failures == []


def test_dark_frame_fails():
    ff = FaceFrame(synth_landmarks(), ok=True)
    r = check_frame(frame(10), ff)
    assert not r.passed
    assert "Lighting too dark — add light." in r.failures


def test_tiny_face_fails():
    lm = synth_landmarks()
    center = np.array([0.5, 0.5, 0.0], np.float32)
    shrunk = (lm - center) * 0.1 + center  # interocular ≈ 19 px
    r = check_frame(frame(128), FaceFrame(shrunk.astype(np.float32), ok=True))
    assert not r.passed
    assert "Move closer — your eyes are too small in frame." in r.failures
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gates.py -v` — Expected: FAIL, module not found.

- [ ] **Step 3: Implement `gates.py` (and empty `quality/__init__.py`)**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gates.py -v` — Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visionscreen/quality tests/test_gates.py && git commit -m "feat: capture-quality gates with retake instructions"
```

---

### Task 5: Report model + HTML rendering

**Files:**
- Create: `src/visionscreen/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Tier` (Literal `"measured" | "weak-signal" | "inconclusive"`), `Finding(module: str, summary: str, tier: Tier, metrics: dict, retakes: list[str])`, `DISCLAIMER` constant (exact copy from Global Constraints), `render_html(findings: list[Finding], session_id: str) -> str`.

- [ ] **Step 1: Write the failing test**

`tests/test_report.py`:

```python
from visionscreen.report import DISCLAIMER, Finding, render_html


def test_disclaimer_always_present():
    html = render_html([], session_id="s1")
    assert DISCLAIMER in html
    assert DISCLAIMER == (
        "This is a screening signal only, not a diagnosis. "
        "See an optometrist for a clinical evaluation."
    )


def test_measured_finding_shows_metrics():
    f = Finding(
        module="acuity", summary="Estimated acuity 0.30 logMAR",
        tier="measured", metrics={"logmar": 0.30}, retakes=[],
    )
    html = render_html([f], session_id="s1")
    assert "Estimated acuity 0.30 logMAR" in html and "measured" in html


def test_inconclusive_hides_metrics_shows_retakes():
    f = Finding(
        module="behavioral", summary="Could not assess",
        tier="inconclusive", metrics={"squint_fraction": 0.9},
        retakes=["Lighting too dark — add light."],
    )
    html = render_html([f], session_id="s1")
    assert "squint_fraction" not in html          # never report numbers we don't trust
    assert "Lighting too dark — add light." in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_report.py -v` — Expected: FAIL, module not found.

- [ ] **Step 3: Implement `report.py`**

```python
from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from typing import Literal

Tier = Literal["measured", "weak-signal", "inconclusive"]

DISCLAIMER = (
    "This is a screening signal only, not a diagnosis. "
    "See an optometrist for a clinical evaluation."
)


@dataclass
class Finding:
    module: str
    summary: str
    tier: Tier
    metrics: dict = field(default_factory=dict)
    retakes: list[str] = field(default_factory=list)


def _finding_html(f: Finding) -> str:
    parts = [f"<section class='finding tier-{f.tier}'>"]
    parts.append(f"<h2>{_html.escape(f.module)} <em>({f.tier})</em></h2>")
    parts.append(f"<p>{_html.escape(f.summary)}</p>")
    if f.tier != "inconclusive" and f.metrics:
        rows = "".join(
            f"<tr><td>{_html.escape(k)}</td><td>{_html.escape(str(v))}</td></tr>"
            for k, v in f.metrics.items()
        )
        parts.append(f"<table>{rows}</table>")
    if f.retakes:
        items = "".join(f"<li>{_html.escape(r)}</li>" for r in f.retakes)
        parts.append(f"<p>To improve this result, retake:</p><ul>{items}</ul>")
    parts.append("</section>")
    return "".join(parts)


def render_html(findings: list[Finding], session_id: str) -> str:
    body = "".join(_finding_html(f) for f in findings) or "<p>No results.</p>"
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Vision Screening Report</title></head><body>"
        f"<h1>Vision Screening Report — session {_html.escape(session_id)}</h1>"
        f"<p class='disclaimer'><strong>{DISCLAIMER}</strong></p>"
        f"{body}"
        "</body></html>"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_report.py -v` — Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visionscreen/report.py tests/test_report.py && git commit -m "feat: screening report model + HTML rendering"
```

---

### Task 6: Acuity engine (tumbling-E sizing, staircase, scoring)

**Files:**
- Create: `src/visionscreen/modules/__init__.py`, `src/visionscreen/modules/acuity.py`
- Test: `tests/test_acuity.py`

**Interfaces:**
- Consumes: `Finding`/`Tier` from Task 5.
- Produces:
  - `letter_height_px(logmar: float, distance_cm: float, px_per_cm: float) -> float` — a logMAR-L letter subtends `5 * 10**L` arcmin.
  - `Staircase(start_logmar=1.0, floor=-0.3, ceiling=1.3)` with `current() -> float`, `record(correct: bool) -> None`, `done: bool` (after 6 reversals or 30 trials), `threshold() -> float | None` (mean of last 4 reversal values; `None` until done). Step: down 0.1 on correct, up 0.2 on incorrect.
  - `score_trials(trials: list[dict]) -> Finding` — trials are `{"logmar": float, "shown": str, "answered": str}` (from `ScreenEvent.payload`, kinds `"trial"`). Tier `"measured"` if ≥ 15 trials, `"weak-signal"` if 8–14, else `"inconclusive"`.

- [ ] **Step 1: Write the failing test**

`tests/test_acuity.py`:

```python
import math

import pytest

from visionscreen.modules.acuity import Staircase, letter_height_px, score_trials


def test_letter_height_physics():
    # logMAR 0.0 at 50 cm: 5 arcmin → height = 2*50*tan(2.5') ≈ 0.0727 cm
    px = letter_height_px(0.0, distance_cm=50.0, px_per_cm=37.8)
    assert px == pytest.approx(2 * 50 * math.tan(math.radians(5 / 60 / 2)) * 37.8, rel=1e-3)
    # one logMAR unit = 10x the size
    assert letter_height_px(1.0, 50.0, 37.8) == pytest.approx(px * 10, rel=1e-3)


def simulate(true_logmar: float) -> Staircase:
    s = Staircase()
    while not s.done:
        s.record(s.current() >= true_logmar)  # ideal observer: correct iff letter big enough
    return s


def test_staircase_converges_to_true_threshold():
    s = simulate(0.4)
    assert s.threshold() == pytest.approx(0.4, abs=0.15)


def test_staircase_terminates_within_30_trials():
    s = Staircase()
    for _ in range(30):
        if s.done:
            break
        s.record(False)
    assert s.done


def make_trials(n: int, logmar: float = 0.3) -> list[dict]:
    return [{"logmar": logmar, "shown": "up", "answered": "up"} for _ in range(n)]


def test_score_trials_tiers():
    assert score_trials(make_trials(20)).tier == "measured"
    assert score_trials(make_trials(10)).tier == "weak-signal"
    f = score_trials(make_trials(3))
    assert f.tier == "inconclusive"
    assert f.metrics == {} or "logmar" not in f.metrics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_acuity.py -v` — Expected: FAIL, module not found.

- [ ] **Step 3: Implement `acuity.py` (and empty `modules/__init__.py`)**

```python
from __future__ import annotations

import math

from visionscreen.report import Finding

STEP_DOWN, STEP_UP = 0.1, 0.2
MAX_TRIALS, MAX_REVERSALS = 30, 6


def letter_height_px(logmar: float, distance_cm: float, px_per_cm: float) -> float:
    arcmin = 5.0 * (10.0 ** logmar)
    height_cm = 2.0 * distance_cm * math.tan(math.radians(arcmin / 60.0 / 2.0))
    return height_cm * px_per_cm


class Staircase:
    def __init__(self, start_logmar: float = 1.0, floor: float = -0.3, ceiling: float = 1.3):
        self._level = start_logmar
        self._floor, self._ceiling = floor, ceiling
        self._last_correct: bool | None = None
        self._reversals: list[float] = []
        self._trials = 0

    def current(self) -> float:
        return self._level

    @property
    def done(self) -> bool:
        return len(self._reversals) >= MAX_REVERSALS or self._trials >= MAX_TRIALS

    def record(self, correct: bool) -> None:
        if self.done:
            return
        self._trials += 1
        if self._last_correct is not None and correct != self._last_correct:
            self._reversals.append(self._level)
        self._last_correct = correct
        delta = -STEP_DOWN if correct else STEP_UP
        self._level = min(self._ceiling, max(self._floor, self._level + delta))

    def threshold(self) -> float | None:
        if not self.done or not self._reversals:
            return None
        tail = self._reversals[-4:]
        return sum(tail) / len(tail)


def score_trials(trials: list[dict]) -> Finding:
    n = len(trials)
    if n < 8:
        return Finding(
            module="acuity",
            summary=f"Not enough acuity trials completed ({n}) to estimate.",
            tier="inconclusive",
            retakes=["Complete the full letter test — answer every letter shown."],
        )
    # replay the recorded trials to recover the threshold
    correct_at: list[tuple[float, bool]] = [
        (t["logmar"], t["shown"] == t["answered"]) for t in trials
    ]
    # smallest level with majority-correct performance over >= 2 trials
    # (a single lucky guess at a tiny letter must not set the threshold)
    levels = sorted({lm for lm, _ in correct_at})
    threshold = levels[-1]
    for lv in levels:
        results = [ok for lm, ok in correct_at if lm == lv]
        if len(results) >= 2 and sum(results) / len(results) >= 0.5:
            threshold = lv
            break
    tier = "measured" if n >= 15 else "weak-signal"
    return Finding(
        module="acuity",
        summary=f"Estimated acuity {threshold:.2f} logMAR",
        tier=tier,
        metrics={"logmar": round(threshold, 2), "trials": n},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_acuity.py -v` — Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visionscreen/modules tests/test_acuity.py && git commit -m "feat: tumbling-E acuity engine (sizing, staircase, scoring)"
```

---

### Task 7: Behavioral module

**Files:**
- Create: `src/visionscreen/modules/behavioral.py`
- Test: `tests/test_behavioral.py`

**Interfaces:**
- Consumes: `Finding`/`Tier` (Task 5). Inputs are per-frame series produced by the analyzer (Task 8): `ears: list[float]` (mean of both eyes' EAR), `interocular: list[float]`, `rolls: list[float]`, `valid_fraction: float` (fraction of frames that passed gates).
- Produces: `analyze_series(ears, interocular, rolls, valid_fraction) -> Finding`. Thresholds: squint = EAR < 0.18; flag `"frequent squinting"` if > 20% of frames squint; `"leaning toward screen"` if median of last-20% interocular / median of first-20% > 1.15; `"sustained head tilt"` if median |roll| > 8°. Tier: `valid_fraction >= 0.7` → `"measured"`, `>= 0.4` → `"weak-signal"`, else `"inconclusive"`.

- [ ] **Step 1: Write the failing test**

`tests/test_behavioral.py`:

```python
from visionscreen.modules.behavioral import analyze_series


def steady(n=100, ear=0.28, iod=200.0, roll=0.0):
    return [ear] * n, [iod] * n, [roll] * n


def test_normal_series_no_flags():
    ears, iod, rolls = steady()
    f = analyze_series(ears, iod, rolls, valid_fraction=0.95)
    assert f.tier == "measured"
    assert f.metrics["flags"] == []


def test_squinting_flagged():
    ears = [0.28] * 60 + [0.10] * 40  # 40% of frames squinting
    _, iod, rolls = steady()
    f = analyze_series(ears, iod, rolls, valid_fraction=0.95)
    assert "frequent squinting" in f.metrics["flags"]


def test_lean_in_flagged():
    iod = [200.0] * 80 + [260.0] * 20  # face grows 30% near the end
    ears, _, rolls = steady()
    f = analyze_series(ears, iod, rolls, valid_fraction=0.95)
    assert "leaning toward screen" in f.metrics["flags"]


def test_low_valid_fraction_is_inconclusive():
    ears, iod, rolls = steady()
    f = analyze_series(ears, iod, rolls, valid_fraction=0.3)
    assert f.tier == "inconclusive"
    assert f.retakes  # must tell the user how to fix it
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_behavioral.py -v` — Expected: FAIL, module not found.

- [ ] **Step 3: Implement `behavioral.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_behavioral.py -v` — Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visionscreen/modules/behavioral.py tests/test_behavioral.py && git commit -m "feat: behavioral module (squint, lean-in, tilt flags)"
```

---

### Task 8: Session analyzer (video → findings)

**Files:**
- Create: `src/visionscreen/analyzer.py`
- Test: `tests/test_analyzer.py`

**Interfaces:**
- Consumes: everything above — `LandmarkExtractor`, `eye_aspect_ratio`/`interocular_px`/`head_roll_deg`, `check_frame`, `score_trials`, `analyze_series`, `SessionMeta`.
- Produces: `analyze_session(video_path: Path, meta: SessionMeta) -> list[Finding]`. Reads the video with `cv2.VideoCapture`, runs perception + gates per frame, builds the behavioral series from gate-passing frames, and scores acuity from `meta.segment("acuity")` trial events. Always returns one finding per module (inconclusive if a segment is absent).

- [ ] **Step 1: Write the failing test**

`tests/test_analyzer.py`:

```python
import cv2
import numpy as np
import pytest

from visionscreen.analyzer import analyze_session
from visionscreen.protocol import ScreenEvent, SegmentMeta, SessionMeta


@pytest.fixture(scope="module")
def face_video(tmp_path_factory):
    skimage = pytest.importorskip("skimage")
    from skimage import data

    path = tmp_path_factory.mktemp("vid") / "face.avi"
    frame = cv2.resize(data.astronaut()[:, :, ::-1].copy(), (640, 480))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480))
    for _ in range(20):
        writer.write(frame)
    writer.release()
    return path


def make_meta(n_trials: int = 16) -> SessionMeta:
    events = [
        ScreenEvent(ts=float(i), kind="trial",
                    payload={"logmar": 0.4, "shown": "up", "answered": "up"})
        for i in range(n_trials)
    ]
    return SessionMeta(
        session_id="t1", px_per_cm=37.8, distance_cm=50.0, fps=10.0,
        segments=[SegmentMeta(test_id="acuity", start_ts=0.0, end_ts=2.0, events=events)],
    )


def test_analyze_session_produces_both_findings(face_video):
    findings = analyze_session(face_video, make_meta())
    modules = {f.module: f for f in findings}
    assert set(modules) == {"acuity", "behavioral"}
    assert modules["acuity"].tier == "measured"
    assert modules["behavioral"].tier in ("measured", "weak-signal")


def test_missing_acuity_segment_is_inconclusive(face_video):
    meta = SessionMeta(session_id="t2", px_per_cm=37.8, distance_cm=50.0,
                       fps=10.0, segments=[])
    findings = analyze_session(face_video, meta)
    acuity = next(f for f in findings if f.module == "acuity")
    assert acuity.tier == "inconclusive"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_analyzer.py -v` — Expected: FAIL, module not found.

- [ ] **Step 3: Implement `analyzer.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_analyzer.py -v` — Expected: 2 PASS. (The astronaut frame passes the gates: face detected, interocular well above 60 px at 640-wide.)

- [ ] **Step 5: Commit**

```bash
git add src/visionscreen/analyzer.py tests/test_analyzer.py && git commit -m "feat: session analyzer wiring perception, gates, and modules"
```

---

### Task 9: Web capture app (guided protocol UI + analyze endpoint)

**Files:**
- Create: `webapp/app.py`, `webapp/static/index.html`
- Test: `tests/test_webapp.py`

**Interfaces:**
- Consumes: `analyze_session` (Task 8), `render_html` (Task 5), `SessionMeta.from_json` (Task 1), `letter_height_px` (Task 6, used client-side via a `/config` endpoint).
- Produces: FastAPI app `webapp.app:app` with `GET /` (static UI), `GET /config?logmar=&distance_cm=&px_per_cm=` → `{"letter_px": float}`, `POST /analyze` (multipart: `video` file + `meta` JSON string) → HTML report.

- [ ] **Step 1: Write the failing test**

`tests/test_webapp.py`:

```python
from fastapi.testclient import TestClient

from webapp.app import app

from tests.test_analyzer import make_meta  # reuse the 16-trial session meta


def test_index_served():
    c = TestClient(app)
    r = c.get("/")
    assert r.status_code == 200
    assert "Vision Screening" in r.text


def test_config_letter_size():
    c = TestClient(app)
    r = c.get("/config", params={"logmar": 0.0, "distance_cm": 50, "px_per_cm": 37.8})
    assert r.status_code == 200
    assert 2.0 < r.json()["letter_px"] < 4.0  # ≈2.75 px at these params


def test_analyze_returns_report(face_video_bytes):
    c = TestClient(app)
    r = c.post(
        "/analyze",
        files={"video": ("session.avi", face_video_bytes, "video/x-msvideo")},
        data={"meta": make_meta().to_json()},
    )
    assert r.status_code == 200
    assert "screening signal only" in r.text
    assert "acuity" in r.text
```

Add to `tests/conftest.py` (create it):

```python
import cv2
import pytest


@pytest.fixture(scope="session")
def face_video_bytes(tmp_path_factory):
    skimage = pytest.importorskip("skimage")
    from skimage import data

    path = tmp_path_factory.mktemp("vid") / "face.avi"
    frame = cv2.resize(data.astronaut()[:, :, ::-1].copy(), (640, 480))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480))
    for _ in range(20):
        writer.write(frame)
    writer.release()
    return path.read_bytes()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_webapp.py -v` — Expected: FAIL, `No module named 'webapp'`.

- [ ] **Step 3: Implement `webapp/app.py`**

```python
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from visionscreen.analyzer import analyze_session
from visionscreen.modules.acuity import letter_height_px
from visionscreen.protocol import SessionMeta
from visionscreen.report import render_html

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Vision Screening")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/config")
def config(logmar: float, distance_cm: float, px_per_cm: float) -> dict:
    return {"letter_px": letter_height_px(logmar, distance_cm, px_per_cm)}


@app.post("/analyze")
async def analyze(video: UploadFile, meta: str = Form(...)) -> HTMLResponse:
    session = SessionMeta.from_json(meta)
    with tempfile.NamedTemporaryFile(suffix=Path(video.filename or "v.webm").suffix) as tmp:
        tmp.write(await video.read())
        tmp.flush()
        findings = analyze_session(Path(tmp.name), session)
    return HTMLResponse(render_html(findings, session.session_id))
```

`webapp/static/index.html` (guided protocol: calibrate → record acuity test with webcam → upload). Keep it dependency-free vanilla JS:

```html
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Vision Screening</title>
<style>
  body { font-family: system-ui; max-width: 640px; margin: 2rem auto; text-align: center; }
  #letter { font-weight: bold; line-height: 1; user-select: none; }
  .rot-up { display: inline-block; transform: rotate(-90deg); }
  .rot-down { display: inline-block; transform: rotate(90deg); }
  .rot-left { display: inline-block; transform: rotate(180deg); }
  .rot-right { display: inline-block; }
  #video { width: 160px; position: fixed; right: 1rem; bottom: 1rem; }
</style></head>
<body>
<h1>Vision Screening</h1>
<p>Sit about 50 cm from the screen. When the letter E appears, press the arrow key
matching the direction its prongs point. 20 letters total.</p>
<button id="start">Start test</button>
<div id="letter"></div>
<video id="video" autoplay muted></video>
<script>
const DIRS = ["up", "down", "left", "right"];
const DISTANCE_CM = 50, PX_PER_CM = window.devicePixelRatio * 96 / 2.54;
let events = [], recorder, chunks = [], startTs = 0;
let logmar = 1.0, trial = 0, shown = null;
const N_TRIALS = 20;

async function begin() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  document.getElementById("video").srcObject = stream;
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = e => chunks.push(e.data);
  recorder.start();
  startTs = performance.now();
  document.getElementById("start").hidden = true;
  nextTrial();
}

async function nextTrial() {
  if (trial >= N_TRIALS) return finish();
  shown = DIRS[Math.floor(Math.random() * DIRS.length)];
  const r = await fetch(`/config?logmar=${logmar}&distance_cm=${DISTANCE_CM}&px_per_cm=${PX_PER_CM}`);
  const { letter_px } = await r.json();
  const el = document.getElementById("letter");
  el.style.fontSize = `${Math.max(letter_px, 4)}px`;
  el.innerHTML = `<span class="rot-${shown}">E</span>`;
}

document.addEventListener("keydown", e => {
  const key = e.key.replace("Arrow", "").toLowerCase();
  if (!shown || !DIRS.includes(key)) return;
  const correct = key === shown;
  events.push({ ts: (performance.now() - startTs) / 1000, kind: "trial",
                payload: { logmar, shown, answered: key } });
  logmar = Math.min(1.3, Math.max(-0.3, logmar + (correct ? -0.1 : 0.2)));
  trial += 1; shown = null;
  nextTrial();
});

async function finish() {
  recorder.stop();
  await new Promise(res => recorder.onstop = res);
  const endTs = (performance.now() - startTs) / 1000;
  const meta = JSON.stringify({
    session_id: crypto.randomUUID(), px_per_cm: PX_PER_CM,
    distance_cm: DISTANCE_CM, fps: 30,
    segments: [{ test_id: "acuity", start_ts: 0, end_ts: endTs, events }],
  });
  const form = new FormData();
  form.append("video", new Blob(chunks, { type: "video/webm" }), "session.webm");
  form.append("meta", meta);
  const resp = await fetch("/analyze", { method: "POST", body: form });
  document.open(); document.write(await resp.text()); document.close();
}

document.getElementById("start").addEventListener("click", begin);
</script>
</body>
</html>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_webapp.py -v` — Expected: 3 PASS.
Then manual smoke check: `uvicorn webapp.app:app --port 8123` and open `http://localhost:8123` — the test should run start-to-finish in a browser and land on a report page.

- [ ] **Step 5: Commit**

```bash
git add webapp tests/test_webapp.py tests/conftest.py && git commit -m "feat: guided capture web app with analyze endpoint"
```

---

### Task 10: Module 1 benchmark harness

**Files:**
- Create: `benchmarks/__init__.py` (empty), `benchmarks/bench_module1.py`
- Test: `tests/test_bench_module1.py`

**Interfaces:**
- Consumes: `Staircase`, `score_trials` (Task 6).
- Produces: `simulate_observer(true_logmar: float, lapse: float, rng) -> list[dict]` (runs a staircase with a noisy simulated observer, returns trial dicts), `run_benchmark(n_observers: int = 50, seed: int = 7) -> dict` returning `{"mean_abs_error": float, "max_abs_error": float, "n": int, "rows": [...]}`, and a `python -m benchmarks.bench_module1` entry that writes `results/module1.json` and prints a markdown table. This is the first column of the spec's benchmark section (end-to-end screening accuracy on synthetic "patients"); Plans 2–3 extend the same pattern.

- [ ] **Step 1: Write the failing test**

`tests/test_bench_module1.py`:

```python
import numpy as np

from benchmarks.bench_module1 import run_benchmark, simulate_observer


def test_simulated_observer_produces_trials():
    rng = np.random.default_rng(0)
    trials = simulate_observer(true_logmar=0.4, lapse=0.05, rng=rng)
    assert len(trials) >= 8
    assert {"logmar", "shown", "answered"} <= set(trials[0])


def test_benchmark_recovers_acuity():
    result = run_benchmark(n_observers=50, seed=7)
    assert result["n"] == 50
    assert result["mean_abs_error"] < 0.15  # logMAR — the headline Module 1 number
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench_module1.py -v` — Expected: FAIL, module not found.

- [ ] **Step 3: Implement `bench_module1.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visionscreen.modules.acuity import Staircase, score_trials

DIRS = ["up", "down", "left", "right"]


def simulate_observer(true_logmar: float, lapse: float, rng: np.random.Generator) -> list[dict]:
    """Ideal-ish observer: sees letters larger than threshold, guesses below it,
    with a lapse rate of random errors."""
    s = Staircase()
    trials: list[dict] = []
    while not s.done:
        level = s.current()
        shown = DIRS[rng.integers(4)]
        sees = level >= true_logmar and rng.random() > lapse
        answered = shown if sees else DIRS[rng.integers(4)]
        trials.append({"logmar": round(level, 2), "shown": shown, "answered": answered})
        s.record(answered == shown)
    return trials


def run_benchmark(n_observers: int = 50, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_observers):
        true_logmar = float(rng.uniform(-0.1, 1.0))
        trials = simulate_observer(true_logmar, lapse=0.05, rng=rng)
        finding = score_trials(trials)
        est = finding.metrics.get("logmar")
        rows.append({"true": round(true_logmar, 2), "est": est,
                     "tier": finding.tier,
                     "abs_error": None if est is None else abs(est - true_logmar)})
    errors = [r["abs_error"] for r in rows if r["abs_error"] is not None]
    return {
        "n": n_observers,
        "mean_abs_error": round(float(np.mean(errors)), 3),
        "max_abs_error": round(float(np.max(errors)), 3),
        "rows": rows,
    }


def main() -> None:
    result = run_benchmark()
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "module1.json").write_text(json.dumps(result, indent=2))
    print("| metric | value |\n|---|---|")
    print(f"| observers | {result['n']} |")
    print(f"| mean abs error (logMAR) | {result['mean_abs_error']} |")
    print(f"| max abs error (logMAR) | {result['max_abs_error']} |")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and the harness**

Run: `python -m pytest tests/test_bench_module1.py -v` — Expected: 2 PASS.
Run: `python -m benchmarks.bench_module1` — Expected: markdown table printed, `results/module1.json` written. Add `results/` to `.gitignore`.

- [ ] **Step 5: Run the FULL suite, then commit**

Run: `python -m pytest -v` — Expected: all tests from Tasks 1–10 PASS.

```bash
echo "results/" >> .gitignore
echo ".venv/" >> .gitignore
git add benchmarks tests/test_bench_module1.py .gitignore
git commit -m "feat: Module 1 benchmark harness (simulated observers)"
```

---

## Out of Scope for This Plan (→ Plans 2 and 3)

- Module 2 (Hirschberg alignment, cover test, gaze pursuit) — Plan 2, adds pursuit protocol segment + reflex localization to the perception layer.
- Module 3 (photorefraction), the synthetic eye renderer, learned perception nets (PyTorch), and public-dataset/scraped-data acquisition with provenance logging — Plan 3.
- These plug into the interfaces defined here: new modules return `Finding`s consumed by `analyzer.py` and `render_html`; new benchmark scripts follow the `benchmarks/bench_*.py` pattern.
