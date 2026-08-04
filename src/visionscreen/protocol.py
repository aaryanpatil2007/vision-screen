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
