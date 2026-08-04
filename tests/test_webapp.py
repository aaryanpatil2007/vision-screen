from fastapi.testclient import TestClient

from webapp.app import app

from tests.test_analyzer import acuity_events, make_meta
from visionscreen.protocol import SegmentMeta


def test_index_served():
    c = TestClient(app)
    r = c.get("/")
    assert r.status_code == 200
    assert "VisionScreen" in r.text
    assert "Comprehensive vision screening" in r.text


def test_static_assets_served():
    c = TestClient(app)
    for path in ("/static/js/app.js", "/static/js/tracker.js",
                 "/static/js/stimuli.js", "/static/css/app.css"):
        assert c.get(path).status_code == 200, path


def test_config_letter_size():
    c = TestClient(app)
    r = c.get("/config", params={"logmar": 0.0, "distance_cm": 50, "px_per_cm": 37.8})
    assert r.status_code == 200
    assert 2.0 < r.json()["letter_px"] < 4.0  # ≈2.75 px at these params


def test_analyze_returns_report(face_video_bytes):
    c = TestClient(app)
    meta = make_meta([SegmentMeta("acuity_both", 0.0, 2.0, acuity_events())])
    r = c.post(
        "/analyze",
        files={"video": ("session.avi", face_video_bytes, "video/x-msvideo")},
        data={"meta": meta.to_json()},
    )
    assert r.status_code == 200
    assert "screening signal only" in r.text
    assert "acuity" in r.text
    assert "Save as PDF" in r.text        # the production report shell rendered


def test_analyze_rejects_malformed_meta(face_video_bytes):
    c = TestClient(app)
    r = c.post(
        "/analyze",
        files={"video": ("session.avi", face_video_bytes, "video/x-msvideo")},
        data={"meta": "{not json"},
    )
    assert r.status_code >= 400


def test_health_reports_learned_perception():
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["learned_perception"], bool)


def test_config_rejects_implausible_calibration():
    c = TestClient(app)
    assert c.get("/config", params={"logmar": 0, "distance_cm": 0,
                                    "px_per_cm": 37.8}).status_code == 422
    assert c.get("/config", params={"logmar": 0, "distance_cm": 50,
                                    "px_per_cm": 0}).status_code == 422


def test_analyze_rejects_empty_video():
    c = TestClient(app)
    meta = make_meta([SegmentMeta("acuity_both", 0.0, 2.0, acuity_events())])
    r = c.post("/analyze", files={"video": ("s.avi", b"", "video/x-msvideo")},
               data={"meta": meta.to_json()})
    assert r.status_code == 422
