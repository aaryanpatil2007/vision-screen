import cv2
import numpy as np
import pytest

from visionscreen.analyzer import analyze_session
from visionscreen.protocol import ScreenEvent, SegmentMeta, SessionMeta


@pytest.fixture(scope="module")
def face_video(tmp_path_factory):
    pytest.importorskip("skimage")
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


def test_analyze_session_produces_all_findings(face_video):
    findings = analyze_session(face_video, make_meta())
    modules = {f.module: f for f in findings}
    assert set(modules) == {"acuity", "behavioral", "alignment", "photorefraction"}
    assert modules["acuity"].tier == "measured"
    assert modules["behavioral"].tier in ("measured", "weak-signal")
    # no alignment segment in this meta → honestly inconclusive, with instructions
    assert modules["alignment"].tier == "inconclusive"
    assert modules["alignment"].retakes


def test_alignment_segment_processed(face_video):
    dots = [
        ScreenEvent(ts=i * 0.1, kind="dot", payload={"x": 0.5}) for i in range(20)
    ]
    meta = SessionMeta(
        session_id="t3", px_per_cm=37.8, distance_cm=50.0, fps=10.0,
        segments=[SegmentMeta(test_id="alignment", start_ts=0.0, end_ts=2.0, events=dots)],
    )
    findings = analyze_session(face_video, meta)
    alignment = next(f for f in findings if f.module == "alignment")
    # tier depends on whether a corneal reflex is detectable in the test image;
    # the contract is: a real answer or an honest inconclusive with retakes
    assert alignment.tier in ("measured", "weak-signal", "inconclusive")
    if alignment.tier == "inconclusive":
        assert alignment.retakes
    else:
        assert "deviation_pd" in alignment.metrics


def test_photoref_on_bright_video_is_honest(face_video):
    # astronaut video is a normally lit scene; the dim-room gate must reject it
    # and the module must answer inconclusive-with-retakes, never a number
    meta = SessionMeta(
        session_id="t4", px_per_cm=37.8, distance_cm=50.0, fps=10.0,
        segments=[SegmentMeta(test_id="photoref", start_ts=0.0, end_ts=2.0, events=[])],
    )
    findings = analyze_session(face_video, meta)
    pr = next(f for f in findings if f.module == "photorefraction")
    assert pr.tier == "inconclusive"
    assert pr.retakes
    assert "sphere_d" not in pr.metrics


def test_missing_acuity_segment_is_inconclusive(face_video):
    meta = SessionMeta(session_id="t2", px_per_cm=37.8, distance_cm=50.0,
                       fps=10.0, segments=[])
    findings = analyze_session(face_video, meta)
    acuity = next(f for f in findings if f.module == "acuity")
    assert acuity.tier == "inconclusive"
