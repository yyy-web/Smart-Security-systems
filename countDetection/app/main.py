"""
FastAPI crowd counting service.

Endpoints:
  POST /api/upload/video        — upload video file
  POST /api/upload/model        — upload ONNX model
  POST /api/pipeline/start      — start processing
  POST /api/pipeline/stop       — stop processing
  POST /api/pipeline/pause      — pause
  POST /api/pipeline/resume     — resume
  POST /api/pipeline/line       — update counting line
  GET  /api/pipeline/status     — current status + counts
  WS   /ws/stream               — real-time frame + event stream
"""

import asyncio
import base64
import json
import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure project root in path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline import ProcessingPipeline, PipelineState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Crowd Flow Counter API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "crowd_counter_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

pipeline = ProcessingPipeline()

# WebSocket connection pool
_ws_clients: list[WebSocket] = []
_ws_lock = asyncio.Lock()
_event_loop: Optional[asyncio.AbstractEventLoop] = None


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline callbacks — called from background thread
# ──────────────────────────────────────────────────────────────────────────────

def _broadcast_sync(message: dict):
    """Schedule a broadcast from a non-async context (pipeline thread)."""
    if _event_loop and not _event_loop.is_closed():
        asyncio.run_coroutine_threadsafe(_broadcast(message), _event_loop)


async def _broadcast(message: dict):
    async with _ws_lock:
        dead = []
        for ws in _ws_clients:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            _ws_clients.remove(ws)


def _on_frame(jpeg_bytes: bytes, stats: dict):
    b64 = base64.b64encode(jpeg_bytes).decode()
    _broadcast_sync({"type": "frame", "data": b64, "stats": stats})


def _on_event(event: dict):
    _broadcast_sync({"type": "event", "data": event})


def _on_state(state: str):
    _broadcast_sync({"type": "state", "state": state})


pipeline.on_frame = _on_frame
pipeline.on_event = _on_event
pipeline.on_state = _on_state


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan event
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _event_loop
    _event_loop = asyncio.get_event_loop()
    yield


app = FastAPI(title="Crowd Flow Counter API", version="1.0.0", lifespan=lifespan)
# ──────────────────────────────────────────────────────────────────────────────

SUPPORTED_VIDEO = {".mp4", ".avi", ".flv", ".mov", ".mkv", ".wmv", ".ts", ".m4v"}
SUPPORTED_MODEL = {".onnx"}


@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_VIDEO:
        raise HTTPException(400, f"Unsupported video format: {suffix}. Supported: {SUPPORTED_VIDEO}")

    dest = UPLOAD_DIR / f"video_{int(time.time())}{suffix}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        info = pipeline.load_video(str(dest))
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, str(e))

    return {"status": "ok", "path": str(dest), "info": info}


@app.post("/api/upload/model")
async def upload_model(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_MODEL:
        raise HTTPException(400, "Only .onnx models are supported.")

    dest = UPLOAD_DIR / f"model_{int(time.time())}.onnx"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        pipeline.load_model(str(dest))
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, str(e))

    return {"status": "ok", "path": str(dest)}


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline control
# ──────────────────────────────────────────────────────────────────────────────

class LineConfig(BaseModel):
    x1: float  # relative [0,1]
    y1: float
    x2: float
    y2: float


@app.post("/api/pipeline/start")
async def start_pipeline():
    try:
        pipeline.start()
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"status": "running"}


@app.post("/api/pipeline/stop")
async def stop_pipeline():
    pipeline.stop()
    return {"status": "stopped"}


@app.post("/api/pipeline/pause")
async def pause_pipeline():
    pipeline.pause()
    return {"status": "paused"}


@app.post("/api/pipeline/resume")
async def resume_pipeline():
    pipeline.resume()
    return {"status": "running"}


@app.post("/api/pipeline/line")
async def set_line(cfg: LineConfig):
    pipeline.set_counting_line((cfg.x1, cfg.y1), (cfg.x2, cfg.y2))
    return {"status": "ok", "line": [cfg.x1, cfg.y1, cfg.x2, cfg.y2]}


@app.get("/api/pipeline/status")
async def get_status():
    return pipeline.get_status()


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket stream
# ──────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    async with _ws_lock:
        _ws_clients.append(websocket)
    try:
        while True:
            # Keep connection alive; control messages come via REST
            data = await websocket.receive_text()
            msg = json.loads(data)
            # Handle ping
            if msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    finally:
        async with _ws_lock:
            if websocket in _ws_clients:
                _ws_clients.remove(websocket)


# ──────────────────────────────────────────────────────────────────────────────
# Serve frontend
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>Crowd Counter API running. No frontend found.</h1>")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8014,
        reload=False,
        log_level="info",
    )