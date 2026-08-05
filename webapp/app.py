from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from visionscreen.analyzer import analyze_session
from visionscreen.ml.infer import EyeSegmenter
from visionscreen.modules.acuity import letter_height_px
from visionscreen.protocol import SessionMeta
from visionscreen.diagnosis import differential_from_findings
from visionscreen.report import render_html

STATIC = Path(__file__).parent / "static"
MAX_VIDEO_BYTES = 400 * 1024 * 1024

log = logging.getLogger("visionscreen")
app = FastAPI(title="VisionScreen")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# Load the learned segmenter once at startup — per-request construction would
# reload the checkpoint for every frame batch.
_SEGMENTER: EyeSegmenter | None = None


def get_segmenter() -> EyeSegmenter:
    global _SEGMENTER
    if _SEGMENTER is None:
        _SEGMENTER = EyeSegmenter()
        log.info("segmenter available=%s device=%s", _SEGMENTER.available, _SEGMENTER.device)
    return _SEGMENTER


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> JSONResponse:
    seg = get_segmenter()
    return JSONResponse({
        "status": "ok",
        "learned_perception": seg.available,
        "device": seg.device,
    })


@app.get("/config")
def config(logmar: float, distance_cm: float, px_per_cm: float) -> dict:
    if not (0 < distance_cm <= 400) or not (1 < px_per_cm <= 2000):
        raise HTTPException(status_code=422, detail="implausible distance or screen scale")
    return {"letter_px": letter_height_px(logmar, distance_cm, px_per_cm)}


@app.post("/analyze")
async def analyze(video: UploadFile, meta: str = Form(...)) -> HTMLResponse:
    try:
        session = SessionMeta.from_json(meta)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid session metadata: {exc}")

    if session.fps <= 0 or session.px_per_cm <= 0 or session.distance_cm <= 0:
        raise HTTPException(status_code=422, detail="session metadata out of range")

    payload = await video.read()
    if not payload:
        raise HTTPException(status_code=422, detail="empty video upload")
    if len(payload) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="video too large")

    suffix = Path(video.filename or "session.webm").suffix or ".webm"
    started = time.time()
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(payload)
        tmp.flush()
        try:
            findings = analyze_session(Path(tmp.name), session, get_segmenter())
        except Exception as exc:
            log.exception("analysis failed")
            raise HTTPException(status_code=500, detail=f"analysis failed: {exc}")

    log.info(
        "analyzed session=%s segments=%d findings=%d in %.1fs",
        session.session_id, len(session.segments), len(findings), time.time() - started,
    )
    # The differential must never take the report down with it: a run that
    # produced usable measurements is still worth showing even if the
    # interpretation layer fails, so this degrades to the findings alone.
    conditions = []
    try:
        conditions = differential_from_findings(
            findings, age=session.age_years, symptoms=set(session.symptoms or []))
    except Exception:
        log.exception("differential failed; showing findings only")

    return HTMLResponse(render_html(findings, session.session_id,
                                    conditions=conditions,
                                    correction=session.wearing_correction))
