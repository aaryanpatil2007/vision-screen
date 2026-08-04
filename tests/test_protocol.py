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
