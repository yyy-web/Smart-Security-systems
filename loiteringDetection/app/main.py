"""
FastAPI 主应用：人员滞留检测系统后端
支持：文件上传、参数配置、SSE 流式推帧、数据导出
"""
import asyncio
import base64
import io
import json
import logging
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.processor import VideoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
app = FastAPI(title="人员滞留检测系统", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

processor = VideoProcessor()

task_state: Dict[str, Any] = {
    "running": False,
    "paused": False,
    "video_info": {},
    "current_frame": 0,
    "total_frames": 0,
    "error": "",
}

# 每个 SSE 连接注册一个 asyncio.Queue，广播式推送
_sse_clients: List[asyncio.Queue] = []
_detect_task: Optional[asyncio.Task] = None


# ──────────────────────────────────────────────────────────────
# Pydantic 模型
# ──────────────────────────────────────────────────────────────
class ParamsModel(BaseModel):
    conf_threshold: float = 0.4
    min_area: float = 500
    linger_threshold: float = 30.0
    tolerance_frames: int = 30
    speed_threshold: float = 50.0
    show_track_id: bool = True
    show_class: bool = True
    show_conf: bool = True
    show_trail: bool = True
    show_anchor: bool = True
    show_anchor_hline: bool = True   # show horizontal guide line at anchor Y
    show_linger_count: bool = True
    # perspective compensation
    anchor_mode: str = "foot"
    zone_y_offset: float = 0.0
    use_bev: bool = False


class ZonesModel(BaseModel):
    zones: Dict[int, List[List[int]]]


class BEVModel(BaseModel):
    src_pts: List[List[float]]
    bev_w: int = 400
    bev_h: int = 400


# ──────────────────────────────────────────────────────────────
# 工具：广播消息给所有 SSE 客户端
# ──────────────────────────────────────────────────────────────
async def _broadcast(payload: dict):
    dead = []
    for q in _sse_clients:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass  # 慢客户端丢帧，不阻塞推理
        except Exception:
            dead.append(q)
    for q in dead:
        try:
            _sse_clients.remove(q)
        except ValueError:
            pass


# ──────────────────────────────────────────────────────────────
# 文件上传
# ──────────────────────────────────────────────────────────────
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".flv", ".mov", ".mkv", ".ts", ".wmv"}
ALLOWED_MODEL_EXT = {".onnx"}


@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        raise HTTPException(400, f"不支持的视频格式: {ext}，支持: {ALLOWED_VIDEO_EXT}")
    dest = UPLOAD_DIR / f"video{ext}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        info = processor.open_video(str(dest))
        task_state["video_info"] = info
        task_state["total_frames"] = info["total_frames"]
        task_state["current_frame"] = 0
        preview = processor.get_preview_frame()
        return {"status": "ok", "video_info": info, "preview": preview}
    except Exception as e:
        logger.exception("open_video failed")
        raise HTTPException(500, str(e))


@app.post("/api/upload/model")
async def upload_model(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_MODEL_EXT:
        raise HTTPException(400, "仅支持 .onnx 模型")
    dest = UPLOAD_DIR / "model.onnx"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        processor.load_model(str(dest))
        return {"status": "ok", "message": "模型加载成功"}
    except Exception as e:
        logger.exception("load_model failed")
        raise HTTPException(500, str(e))


# ──────────────────────────────────────────────────────────────
# 区域 / 参数 / BEV
# ──────────────────────────────────────────────────────────────
@app.post("/api/zones")
async def set_zones(body: ZonesModel):
    zones = {int(k): [tuple(p) for p in v] for k, v in body.zones.items()}
    processor.set_zones(zones)
    return {"status": "ok", "zone_count": len(zones)}


@app.post("/api/params")
async def set_params(params: ParamsModel):
    processor.conf_threshold      = params.conf_threshold
    processor.min_area            = params.min_area
    processor.linger_threshold    = params.linger_threshold
    processor.tolerance_frames    = params.tolerance_frames
    processor.speed_threshold     = params.speed_threshold
    processor.show_track_id       = params.show_track_id
    processor.show_class          = params.show_class
    processor.show_conf           = params.show_conf
    processor.show_trail          = params.show_trail
    processor.show_anchor         = params.show_anchor
    processor.show_anchor_hline   = params.show_anchor_hline
    processor.show_linger_count   = params.show_linger_count
    processor.anchor_mode         = params.anchor_mode
    processor.zone_y_offset       = params.zone_y_offset
    processor.use_bev             = params.use_bev
    processor.update_perspective_params()
    return {"status": "ok"}


@app.post("/api/bev/calibrate")
async def bev_calibrate(body: BEVModel):
    ok = processor.set_bev(body.src_pts, bev_w=body.bev_w, bev_h=body.bev_h)
    return {"status": "ok" if ok else "error", "ready": processor.bev.is_ready}


@app.get("/api/bev/status")
async def bev_status():
    return {
        "ready": processor.bev.is_ready,
        "src_pts": processor.bev.src_pts,
        "bev_w": processor.bev.bev_w,
        "bev_h": processor.bev.bev_h,
    }


# ──────────────────────────────────────────────────────────────
# 检测控制
# ──────────────────────────────────────────────────────────────
@app.post("/api/detect/start")
async def start_detect():
    global _detect_task

    if processor.model is None:
        raise HTTPException(400, "请先上传 ONNX 模型")
    if processor.cap is None:
        raise HTTPException(400, "请先上传视频")

    # 如果已有任务在跑，先停掉
    if _detect_task and not _detect_task.done():
        task_state["running"] = False
        try:
            await asyncio.wait_for(_detect_task, timeout=2.0)
        except Exception:
            pass

    processor.reset()
    task_state["running"] = True
    task_state["paused"]  = False
    task_state["current_frame"] = 0
    task_state["error"] = ""

    _detect_task = asyncio.create_task(_detect_loop())
    logger.info("检测任务已启动")
    return {"status": "started"}


@app.post("/api/detect/stop")
async def stop_detect():
    task_state["running"] = False
    task_state["paused"]  = False
    await _broadcast({"type": "stopped"})
    return {"status": "stopped"}


@app.post("/api/detect/pause")
async def pause_detect():
    task_state["paused"] = not task_state["paused"]
    status = "paused" if task_state["paused"] else "resumed"
    return {"status": status}


# ──────────────────────────────────────────────────────────────
# SSE 推流（每个连接独立 Queue，广播模式）
# ──────────────────────────────────────────────────────────────
@app.get("/api/stream")
async def stream():
    q: asyncio.Queue = asyncio.Queue(maxsize=10)
    _sse_clients.append(q)
    logger.info(f"SSE 客户端连接，当前连接数: {len(_sse_clients)}")

    async def event_generator():
        try:
            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=10.0)
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    if data.get("type") in ("end", "stopped", "error"):
                        break
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            try:
                _sse_clients.remove(q)
            except ValueError:
                pass
            logger.info(f"SSE 客户端断开，剩余连接数: {len(_sse_clients)}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ──────────────────────────────────────────────────────────────
# 推理主循环（asyncio Task）
# ──────────────────────────────────────────────────────────────
async def _detect_loop():
    loop = asyncio.get_running_loop()
    fps = processor._fps or 25.0
    interval = 1.0 / fps
    logger.info(f"推理循环开始，视频 FPS={fps:.1f}")

    try:
        while task_state["running"]:
            if task_state["paused"]:
                await asyncio.sleep(0.05)
                continue

            t0 = time.time()

            # 在线程池读帧（避免阻塞事件循环）
            frame = await loop.run_in_executor(None, processor.read_frame)
            if frame is None:
                logger.info("视频读取完毕")
                task_state["running"] = False
                await _broadcast({"type": "end"})
                break

            # 在线程池推理
            result = await loop.run_in_executor(None, processor.process_frame, frame)
            if result is None:
                logger.warning("process_frame 返回 None，跳过本帧")
                await asyncio.sleep(interval)
                continue

            task_state["current_frame"] = result.frame_index

            payload = {
                "type": "frame",
                "frame_index": result.frame_index,
                "timestamp": round(result.timestamp, 2),
                "frame_b64": result.frame_b64,
                "detections": [
                    {
                        "track_id":       d.track_id,
                        "confidence":     d.confidence,
                        "class_name":     d.class_name,
                        "bbox":           d.bbox,
                        "center":         list(d.center),
                        "anchor":         list(d.anchor),
                        "is_lingering":   d.is_lingering,
                        "zone_id":        d.zone_id,
                        "linger_duration": d.linger_duration,
                        "speed":          d.speed,
                        "thumb_b64":      d.thumb_b64,
                    }
                    for d in result.detections
                ],
                "linger_alerts":   result.linger_alerts,    # 完整事件
                "linger_previews": result.linger_previews,  # 实时预览
            }

            await _broadcast(payload)

            elapsed = time.time() - t0
            sleep_t = max(0.001, interval - elapsed)
            await asyncio.sleep(sleep_t)

    except Exception as e:
        err_msg = traceback.format_exc()
        logger.error(f"推理循环异常:\n{err_msg}")
        task_state["running"] = False
        task_state["error"] = str(e)
        await _broadcast({"type": "error", "message": str(e)})


# ──────────────────────────────────────────────────────────────
# 截图 / 导出
# ──────────────────────────────────────────────────────────────
class ScreenshotModel(BaseModel):
    frame_b64: str


@app.post("/api/screenshot")
async def save_screenshot(body: ScreenshotModel):
    img_bytes = base64.b64decode(body.frame_b64)
    filename = f"screenshot_{int(time.time())}.jpg"
    path = UPLOAD_DIR / filename
    with open(path, "wb") as f:
        f.write(img_bytes)
    return FileResponse(str(path), filename=filename, media_type="image/jpeg")


@app.get("/api/export/log")
async def export_log(fmt: str = "json"):
    logs = processor.linger_log
    if fmt == "csv":
        lines = ["track_id,zone_id,duration,time"]
        for r in logs:
            lines.append(f"{r['track_id']},{r['zone_id']},{r['duration']},{r['time']}")
        content = "\n".join(lines).encode("utf-8-sig")
        return StreamingResponse(
            io.BytesIO(content),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=linger_log.csv"},
        )
    return JSONResponse(content=logs)


# ──────────────────────────────────────────────────────────────
# 系统状态
# ──────────────────────────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    return {
        **task_state,
        "sse_clients": len(_sse_clients),
        "model_loaded": processor.model is not None,
        "video_loaded": processor.cap is not None,
        "linger_count": len(processor.linger_log),
        "linger_log": processor.linger_log[-50:],
    }


# ──────────────────────────────────────────────────────────────
# 静态资源 & 前端入口
# ──────────────────────────────────────────────────────────────
static_path = Path("/home/bigmodel/ultralytics/loiteringDetection/static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory="/home/bigmodel/ultralytics/loiteringDetection/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = Path("/home/bigmodel/ultralytics/loiteringDetection/static/index.html")
    if html_file.exists():
        return HTMLResponse(html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>人员滞留检测系统</h1><p>前端文件未找到</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8082, reload=False)