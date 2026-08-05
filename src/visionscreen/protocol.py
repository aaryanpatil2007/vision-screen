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
    #: Age in years, optional. Every prevalence in the differential engine is
    #: age-dependent and several steeply so — cataract, presbyopia and glaucoma
    #: all change by an order of magnitude across adult life — so without it
    #: the engine falls back to a 45-year-old's base rates and says so.
    age_years: float | None = None
    #: What the person wore during the test: "none", "glasses" or "contacts".
    #: This changes what every result *means*, not merely how precise it is.
    #: Uncorrected, acuity and the reflex estimate a refractive error.
    #: Corrected, they measure how well the current prescription is working —
    #: so a focusing error found while wearing contacts is evidence of
    #: under-correction, and quoting a ballpark prescription from it would be
    #: quoting the residual error as though it were the whole thing.
    wearing_correction: str | None = None
    #: Symptoms reported, as keys (e.g. "sudden_flashes"). Some conditions are
    #: reachable no other way: nothing measurable on a webcam detects a retinal
    #: detachment, but the symptom triad is decisive.
    symptoms: list[str] = field(default_factory=list)

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
        # tolerate metadata written before these fields existed
        d.setdefault("age_years", None)
        d.setdefault("wearing_correction", None)
        d.setdefault("symptoms", [])
        return cls(segments=segments, **d)
