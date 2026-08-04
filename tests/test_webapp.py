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
