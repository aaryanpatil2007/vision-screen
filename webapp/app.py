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
