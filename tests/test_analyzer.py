import cv2
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
    for _ in range(24):
        writer.write(frame)
    writer.release()
    return path


def acuity_events(n=16, logmar=0.4):
    return [
        ScreenEvent(ts=float(i) * 0.05, kind="trial",
                    payload={"logmar": logmar, "shown": "up", "answered": "up"})
        for i in range(n)
    ]


def make_meta(segments) -> SessionMeta:
    return SessionMeta(session_id="t", px_per_cm=37.8, distance_cm=50.0,
                       fps=10.0, segments=segments)


def modules(findings):
    return {f.module: f for f in findings}


def test_only_attempted_tests_are_reported(face_video):
    """A test that was not part of the session must not appear in the report."""
    meta = make_meta([SegmentMeta("acuity_both", 0.0, 2.0, acuity_events())])
    m = modules(analyze_session(face_video, meta))
    assert "acuity (both eyes)" in m
    assert "behavioral" in m           # always available from the video itself
    assert not any(k.startswith("photorefraction") for k in m)
    assert "amsler" not in m


def test_monocular_acuity_reported_separately(face_video):
    meta = make_meta([
        SegmentMeta("acuity_both", 0.0, 0.8, acuity_events(16, 0.1)),
        SegmentMeta("acuity_right", 0.8, 1.6, acuity_events(16, 0.3)),
        SegmentMeta("acuity_left", 1.6, 2.3, acuity_events(16, 0.5)),
    ])
    m = modules(analyze_session(face_video, meta))
    assert m["acuity (both eyes)"].metrics["logmar"] == 0.1
    assert m["acuity (right eye)"].metrics["logmar"] == 0.3
    assert m["acuity (left eye)"].metrics["logmar"] == 0.5


def test_contrast_segment_scored(face_video):
    events = [
        ScreenEvent(ts=i * 0.05, kind="trial",
                   payload={"log_cs": round(0.15 * i, 2), "shown": "C",
                            "answered": "C", "correct": 0.15 * i <= 1.5})
        for i in range(12)
    ]
    meta = make_meta([SegmentMeta("contrast", 0.0, 2.0, events)])
    m = modules(analyze_session(face_video, meta))
    assert "contrast" in m
    assert m["contrast"].metrics["log_cs"] == pytest.approx(1.5, abs=0.16)


def test_color_and_amsler_segments_scored(face_video):
    color = [ScreenEvent(ts=0.1 * i, kind="plate",
                        payload={"id": f"p{i}", "type": "general", "shown": 8,
                                 "answered": 8})
             for i in range(1, 7)]
    amsler = [ScreenEvent(ts=1.0, kind="amsler",
                         payload={"eye": e, "marks": [], "reported_normal": True})
              for e in ("right", "left")]
    meta = make_meta([
        SegmentMeta("color_vision", 0.0, 0.9, color),
        SegmentMeta("amsler", 0.9, 2.0, amsler),
    ])
    m = modules(analyze_session(face_video, meta))
    assert m["color_vision"].tier == "weak-signal"   # never 'measured' on a screen
    assert m["amsler"].metrics["flags"] == []


def test_astigmatism_no_preference(face_video):
    events = [ScreenEvent(ts=0.2 * i, kind="dial",
                         payload={"eye": e, "dark_meridian_deg": None,
                                  "no_preference": True})
              for i, e in enumerate(("right", "left"))]
    meta = make_meta([SegmentMeta("astigmatism", 0.0, 2.0, events)])
    m = modules(analyze_session(face_video, meta))
    assert m["astigmatism"].metrics["flags"] == []


def test_motility_and_alignment_from_pursuit(face_video):
    dots = [ScreenEvent(ts=i * 0.1, kind="dot", payload={"x": 0.5 + 0.3 * (i % 4) / 4})
            for i in range(24)]
    meta = make_meta([SegmentMeta("motility", 0.0, 2.4, dots)])
    m = modules(analyze_session(face_video, meta))
    assert "motility" in m and "alignment" in m
    for f in (m["motility"], m["alignment"]):
        assert f.tier in ("measured", "weak-signal", "inconclusive")
        if f.tier == "inconclusive":
            assert f.retakes


def test_photoref_on_bright_video_is_honest(face_video):
    meta = make_meta([SegmentMeta("photoref", 0.0, 2.0, [])])
    m = modules(analyze_session(face_video, meta))
    pr = m["photorefraction"]
    assert pr.tier == "inconclusive"
    assert pr.retakes
    assert "sphere_d" not in pr.metrics


def test_pupillometry_segment_present(face_video):
    events = [ScreenEvent(ts=0.5, kind="flash_on", payload={"index": 0}),
              ScreenEvent(ts=1.0, kind="flash_off", payload={"index": 0})]
    meta = make_meta([SegmentMeta("pupil", 0.0, 2.4, events)])
    m = modules(analyze_session(face_video, meta))
    assert "pupillometry" in m
    if m["pupillometry"].tier == "inconclusive":
        assert m["pupillometry"].retakes


def test_empty_session_still_reports_behavioral(face_video):
    m = modules(analyze_session(face_video, make_meta([])))
    assert "behavioral" in m
    assert "acuity" in m and m["acuity"].tier == "inconclusive"
