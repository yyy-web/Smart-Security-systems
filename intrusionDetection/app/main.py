"""
区域入侵检测系统 - FastAPI 主服务
基于 YOLOv8s ONNX 推理 + ByteTrack 跟踪 + 多边形入侵判断
"""
import asyncio
import base64
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
CAPTURES_DIR = BASE_DIR / "captures"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
for d in [CAPTURES_DIR, LOGS_DIR, MODELS_DIR, STATIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────
class Zone(BaseModel):
    id: int
    name: str
    points: List[List[float]]  # [[x,y], ...]
    color: str = "#FF4444"


class DetectionConfig(BaseModel):
    confidence_threshold: float = 0.45
    min_area: float = 1000.0
    intrusion_frames: int = 3
    iou_threshold: float = 0.45
    show_track_id: bool = True
    show_class: bool = True
    show_confidence: bool = True
    show_trajectory: bool = True
    show_centroid: bool = True
    show_intrusion_count: bool = True
    alert_enabled: bool = True
    zones: List[Zone] = []


class IntrusionEvent(BaseModel):
    event_id: str
    track_id: int
    zone_id: int
    zone_name: str
    timestamp: str
    confidence: float
    bbox: List[float]
    snapshot_path: str = ""


# ─────────────────────────────────────────────
# ByteTrack-style Lightweight Tracker
# ─────────────────────────────────────────────
class KalmanTrack:
    """简化 Kalman 跟踪器，用于单目标状态估计"""
    count = 0

    def __init__(self, bbox: np.ndarray, score: float, cls: int):
        KalmanTrack.count += 1
        self.track_id = KalmanTrack.count
        self.cls = cls
        self.score = score
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.state = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, 0, 0], dtype=float)
        # Kalman matrices (constant velocity model)
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1
        self.H = np.eye(4, 8)
        self.P = np.eye(8) * 10
        self.R = np.eye(4) * 1
        self.Q = np.eye(8) * 0.01

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        self.age += 1

    def update(self, bbox: np.ndarray, score: float):
        z = bbox
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1
        self.score = score

    @property
    def bbox(self) -> np.ndarray:
        return self.state[:4].copy()


class ByteTracker:
    """轻量级 ByteTrack 实现，支持高低置信度二阶段匹配"""

    def __init__(self, max_age: int = 30, min_hits: int = 1, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[KalmanTrack] = []
        KalmanTrack.count = 0

    def reset(self):
        self.tracks = []
        KalmanTrack.count = 0

    @staticmethod
    def iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0

    def _match(self, detections: List[Tuple]) -> Tuple[List, List, List]:
        """匈牙利算法贪心近似匹配"""
        if not self.tracks or not detections:
            return [], list(range(len(self.tracks))), list(range(len(detections)))

        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for ti, track in enumerate(self.tracks):
            for di, (bbox, score, cls) in enumerate(detections):
                iou_matrix[ti, di] = self.iou(track.bbox, bbox)

        matched = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_dets = list(range(len(detections)))

        while True:
            if iou_matrix.size == 0:
                break
            idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            ti, di = idx
            if iou_matrix[ti, di] < self.iou_threshold:
                break
            matched.append((ti, di))
            if ti in unmatched_tracks:
                unmatched_tracks.remove(ti)
            if di in unmatched_dets:
                unmatched_dets.remove(di)
            iou_matrix[ti, :] = -1
            iou_matrix[:, di] = -1

        return matched, unmatched_tracks, unmatched_dets

    def update(self, detections: List[Tuple]) -> List[Dict]:
        """
        detections: [(bbox_xyxy, score, cls), ...]
        returns: [{"track_id", "bbox", "score", "cls"}, ...]
        """
        for track in self.tracks:
            track.predict()

        high_conf = [(b, s, c) for b, s, c in detections if s >= 0.6]
        low_conf = [(b, s, c) for b, s, c in detections if s < 0.6]

        # Stage 1: 高置信度匹配
        matched, unmatched_t, unmatched_d = self._match(high_conf)
        for ti, di in matched:
            self.tracks[ti].update(high_conf[di][0], high_conf[di][1])

        # Stage 2: 低置信度匹配未匹配轨迹
        remaining_tracks = [self.tracks[i] for i in unmatched_t]
        if remaining_tracks and low_conf:
            iou_matrix2 = np.zeros((len(remaining_tracks), len(low_conf)))
            for ti, track in enumerate(remaining_tracks):
                for di, (bbox, score, cls) in enumerate(low_conf):
                    iou_matrix2[ti, di] = self.iou(track.bbox, bbox)
            for ti in range(len(remaining_tracks)):
                best_di = np.argmax(iou_matrix2[ti])
                if iou_matrix2[ti, best_di] > self.iou_threshold:
                    remaining_tracks[ti].update(low_conf[best_di][0], low_conf[best_di][1])

        # 新建轨迹
        all_high_unmatched_d = [i for i in range(len(high_conf)) if i not in [d for _, d in matched]]
        for di in all_high_unmatched_d:
            bbox, score, cls = high_conf[di]
            self.tracks.append(KalmanTrack(bbox, score, cls))

        # 删除过期轨迹
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        results = []
        for track in self.tracks:
            if track.time_since_update == 0:
                results.append({
                    "track_id": track.track_id,
                    "bbox": track.bbox.tolist(),
                    "score": float(track.score),
                    "cls": track.cls,
                })
        return results


# ─────────────────────────────────────────────
# YOLOv8 ONNX Inference
# ─────────────────────────────────────────────
class YOLOv8Detector:
    """YOLOv8s ONNX 推理器，OpenCV DNN 后端"""

    def __init__(self, model_path: str, input_size: int = 640):
        self.input_size = input_size
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logger.info(f"Model loaded: {model_path}")

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """LetterBox 缩放 + 归一化"""
        h, w = frame.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2
        padded = cv2.copyMakeBorder(resized, pad_h, self.input_size - new_h - pad_h,
                                    pad_w, self.input_size - new_w - pad_w,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, (self.input_size, self.input_size),
                                     swapRB=True, crop=False)
        return blob, scale, (pad_w, pad_h)

    def postprocess(self, output: np.ndarray, scale: float, pad: Tuple[int, int],
                    orig_shape: Tuple[int, int], conf_thresh: float, iou_thresh: float,
                    target_cls: int = 0) -> List[Tuple]:
        """NMS 后处理，仅保留目标类别（默认 person=0）"""
        pred = output[0]  # (1, 84, 8400) → (84, 8400)
        if pred.ndim == 3:
            pred = pred[0]
        pred = pred.T  # (8400, 84)

        boxes, scores, classes = [], [], []
        for det in pred:
            obj_scores = det[4:]
            cls_id = int(np.argmax(obj_scores))
            if cls_id != target_cls:
                continue
            conf = float(obj_scores[cls_id])
            if conf < conf_thresh:
                continue
            cx, cy, bw, bh = det[0], det[1], det[2], det[3]
            pad_w, pad_h = pad
            x1 = (cx - bw / 2 - pad_w) / scale
            y1 = (cy - bh / 2 - pad_h) / scale
            x2 = (cx + bw / 2 - pad_w) / scale
            y2 = (cy + bh / 2 - pad_h) / scale
            x1 = max(0, min(x1, orig_shape[1]))
            y1 = max(0, min(y1, orig_shape[0]))
            x2 = max(0, min(x2, orig_shape[1]))
            y2 = max(0, min(y2, orig_shape[0]))
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(conf)
            classes.append(cls_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                results.append((np.array([x, y, x + w, y + h]), scores[i], classes[i]))
        return results

    def infer(self, frame: np.ndarray, conf_thresh: float = 0.45,
              iou_thresh: float = 0.45) -> List[Tuple]:
        blob, scale, pad = self.preprocess(frame)
        self.net.setInput(blob)
        output = self.net.forward()
        return self.postprocess(output, scale, pad, frame.shape[:2], conf_thresh, iou_thresh)


# ─────────────────────────────────────────────
# Intrusion Detector Core
# ─────────────────────────────────────────────
class IntrusionDetector:
    """入侵检测核心逻辑"""

    def __init__(self):
        self.zones: List[Zone] = []
        self.tracker = ByteTracker(max_age=30, iou_threshold=0.3)
        # track_id → {zone_id → consecutive_frame_count}
        self.zone_frame_count: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # track_id → set of confirmed intruded zone_ids
        self.confirmed_intrusions: Dict[int, set] = defaultdict(set)
        # track_id → deque of center points (trajectory)
        self.trajectories: Dict[int, deque] = defaultdict(lambda: deque(maxlen=50))
        # All intrusion events
        self.events: List[IntrusionEvent] = []
        # Intrusion snapshots: event_id → base64
        self.snapshots: Dict[str, str] = {}
        # track_id → last known bbox
        self.last_bbox: Dict[int, List[float]] = {}
        # track_id → last score
        self.last_score: Dict[int, float] = {}

    def reset(self):
        self.tracker.reset()
        self.zone_frame_count.clear()
        self.confirmed_intrusions.clear()
        self.trajectories.clear()
        self.events.clear()
        self.snapshots.clear()
        self.last_bbox.clear()
        self.last_score.clear()

    def set_zones(self, zones: List[Zone]):
        self.zones = zones

    @staticmethod
    def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
        """射线法判断点是否在多边形内"""
        x, y = point
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
                inside = not inside
            j = i
        return inside

    def process_frame(self, frame: np.ndarray, detections: List[Tuple],
                      config: DetectionConfig) -> Tuple[np.ndarray, List[Dict], List[IntrusionEvent]]:
        """
        处理单帧：跟踪 → 区域判断 → 入侵事件 → 可视化
        Returns: (annotated_frame, track_results, new_events)
        """
        h, w = frame.shape[:2]
        # 面积过滤
        filtered = []
        for bbox, score, cls in detections:
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area >= config.min_area:
                filtered.append((bbox, score, cls))

        # ByteTrack 更新
        tracks = self.tracker.update(filtered)

        # 记录当前帧活跃的 track_id
        active_ids = {t["track_id"] for t in tracks}

        # 清理不活跃的帧计数
        for tid in list(self.zone_frame_count.keys()):
            if tid not in active_ids:
                del self.zone_frame_count[tid]

        new_events = []
        track_results = []

        for track in tracks:
            tid = track["track_id"]
            bbox = track["bbox"]
            score = track["score"]
            cls = track["cls"]

            self.last_bbox[tid] = bbox
            self.last_score[tid] = score

            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            foot = (cx, y2)  # 底边中心点

            self.trajectories[tid].append((int(cx), int(cy)))

            # 判断是否在各区域内
            intruded_zones = []
            for zone in self.zones:
                in_zone = self.point_in_polygon(foot, zone.points)
                if in_zone:
                    self.zone_frame_count[tid][zone.id] += 1
                else:
                    self.zone_frame_count[tid][zone.id] = 0

                # 连续帧判定
                if (self.zone_frame_count[tid][zone.id] >= config.intrusion_frames
                        and zone.id not in self.confirmed_intrusions[tid]):
                    # 新入侵事件！
                    self.confirmed_intrusions[tid].add(zone.id)
                    event_id = str(uuid.uuid4())[:8]
                    snapshot_b64 = self._capture_snapshot(frame, bbox, tid)
                    event = IntrusionEvent(
                        event_id=event_id,
                        track_id=tid,
                        zone_id=zone.id,
                        zone_name=zone.name,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        confidence=score,
                        bbox=bbox,
                        snapshot_path=event_id,
                    )
                    self.events.append(event)
                    self.snapshots[event_id] = snapshot_b64
                    new_events.append(event)

                if zone.id in self.confirmed_intrusions[tid]:
                    intruded_zones.append(zone.id)

            is_intruder = len(intruded_zones) > 0
            # 裁剪当前帧目标图，用于表格实时展示
            frame_snap = self._capture_snapshot(frame, bbox, tid)
            track_results.append({
                "track_id": tid,
                "bbox": bbox,
                "score": score,
                "cls": COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else str(cls),
                "is_intruder": is_intruder,
                "intruded_zones": intruded_zones,
                "center": [cx, cy],
                "frame_snap": frame_snap,   # 当前帧裁剪图（base64）
            })

        # 可视化
        annotated = self._draw(frame.copy(), track_results, config)
        return annotated, track_results, new_events

    def _capture_snapshot(self, frame: np.ndarray, bbox: List[float], tid: int) -> str:
        """裁剪入侵人员图像并转 base64"""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(frame.shape[1], x2 + pad)
        y2 = min(frame.shape[0], y2 + pad)
        crop = frame[y1:y2, x1:x2]
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode()

    def _draw(self, frame: np.ndarray, tracks: List[Dict], config: DetectionConfig) -> np.ndarray:
        """绘制检测结果：仅绘制检测框 / 轨迹 / 标签。
        警戒区域多边形由前端 Canvas 叠加渲染，后端不重复绘制，避免双重显示。"""

        # 入侵总数
        if config.show_intrusion_count:
            total = len(self.events)
            cv2.rectangle(frame, (10, 10), (220, 45), (20, 20, 20), -1)
            cv2.putText(frame, f"Intrusions: {total}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)

        for t in tracks:
            bbox = t["bbox"]
            tid = t["track_id"]
            is_intruder = t["is_intruder"]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            color = (0, 0, 255) if is_intruder else (0, 200, 50)

            # 边框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 轨迹线
            if config.show_trajectory:
                traj = list(self.trajectories[tid])
                for i in range(1, len(traj)):
                    alpha = int(255 * i / len(traj))
                    cv2.line(frame, traj[i - 1], traj[i], (*color[:2], alpha), 2)

            # 中心点
            if config.show_centroid:
                cv2.circle(frame, (cx, int(y2)), 5, color, -1)

            # 标签
            label_parts = []
            if config.show_track_id:
                label_parts.append(f"ID:{tid}")
            if config.show_class:
                label_parts.append(t["cls"])
            if config.show_confidence:
                label_parts.append(f"{t['score']:.2f}")
            if is_intruder:
                label_parts.append("⚠ INTRUDER")

            if label_parts:
                label = " | ".join(label_parts)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                lx = max(x1, 0)
                ly = max(y1 - th - 8, 0)
                cv2.rectangle(frame, (lx, ly), (lx + tw + 6, ly + th + 8), color, -1)
                cv2.putText(frame, label, (lx + 3, ly + th + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return frame

    @staticmethod
    def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return (b, g, r)


# ─────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.detector: Optional[YOLOv8Detector] = None
        self.intrusion: IntrusionDetector = IntrusionDetector()
        self.config: DetectionConfig = DetectionConfig()
        self.video_path: Optional[str] = None
        self.running: bool = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame: Optional[np.ndarray] = None
        self.ws_clients: List[WebSocket] = []
        self.fps: float = 25.0

    def reset_detection(self):
        self.running = False
        self.intrusion.reset()
        if self.cap:
            self.cap.release()
            self.cap = None


state = AppState()

# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(title="区域入侵检测系统", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="captures")


@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "区域入侵检测系统 API"})


# ─────────────────────────────────────────────
# Upload Endpoints
# ─────────────────────────────────────────────
SUPPORTED_VIDEO_EXTS = {
    ".mp4", ".avi", ".flv", ".mkv", ".mov",
    ".wmv", ".ts", ".m4v", ".webm", ".rmvb", ".3gp",
}

@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的视频格式 '{suffix}'，支持格式: {', '.join(sorted(SUPPORTED_VIDEO_EXTS))}",
        )
    # 使用 uuid 安全文件名，避免中文/特殊字符路径问题
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    uploads_dir = BASE_DIR / "uploads"
    uploads_dir.mkdir(exist_ok=True)
    path = uploads_dir / safe_name
    with open(path, "wb") as f:
        content = await file.read()
        f.write(content)
    state.video_path = str(path)
    state.running = False

    # 读取视频元信息
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(
            status_code=400,
            detail="视频文件无法打开，请检查编码格式或安装对应解码库"
        )
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    cap.release()

    state.fps = fps
    thumbnail_b64 = ""
    if ret:
        state.current_frame = frame
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        thumbnail_b64 = base64.b64encode(buf.tobytes()).decode()

    return {
        "success": True,
        "filename": file.filename,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "thumbnail": thumbnail_b64,
    }


@app.post("/api/upload/model")
async def upload_model(file: UploadFile = File(...)):
    path = MODELS_DIR / file.filename
    with open(path, "wb") as f:
        content = await file.read()
        f.write(content)
    try:
        state.detector = YOLOv8Detector(str(path))
        return {"success": True, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"模型加载失败: {e}")


# ─────────────────────────────────────────────
# Config & Zones
# ─────────────────────────────────────────────
@app.post("/api/config")
async def update_config(config: DetectionConfig):
    state.config = config
    state.intrusion.set_zones(config.zones)
    return {"success": True}


@app.get("/api/config")
async def get_config():
    return state.config


# ─────────────────────────────────────────────
# Detection Control
# ─────────────────────────────────────────────
@app.post("/api/detect/start")
async def start_detection():
    if not state.video_path:
        raise HTTPException(status_code=400, detail="未选择视频")
    if not state.detector:
        raise HTTPException(status_code=400, detail="未加载模型")
    if state.running:
        return {"message": "已在运行"}
    state.reset_detection()
    state.running = True
    state.intrusion.set_zones(state.config.zones)
    asyncio.create_task(_detection_loop())
    return {"success": True, "message": "检测已启动"}


@app.post("/api/detect/stop")
async def stop_detection():
    state.running = False
    return {"success": True, "message": "检测已停止"}


async def _detection_loop():
    """异步检测主循环，通过 WebSocket 推送结果"""
    cap = cv2.VideoCapture(state.video_path)
    state.cap = cap
    frame_interval = 1.0 / state.fps

    while state.running and cap.isOpened():
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            # 视频播放完毕，循环
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        state.current_frame = frame

        # ONNX 推理
        try:
            detections = state.detector.infer(
                frame,
                conf_thresh=state.config.confidence_threshold,
                iou_thresh=state.config.iou_threshold,
            )
        except Exception as e:
            logger.error(f"推理错误: {e}")
            detections = []

        # 入侵检测
        annotated, track_results, new_events = state.intrusion.process_frame(
            frame, detections, state.config
        )

        # 编码帧
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_b64 = base64.b64encode(buf.tobytes()).decode()

        # 构造推送消息
        msg = {
            "type": "frame",
            "frame": frame_b64,
            "tracks": track_results,
            "new_events": [e.dict() for e in new_events],
            "total_intrusions": len(state.intrusion.events),
            "snapshots": {e.event_id: state.intrusion.snapshots.get(e.event_id, "")
                          for e in new_events},
        }

        # WebSocket 广播
        dead = []
        for ws in state.ws_clients:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            state.ws_clients.remove(ws)

        elapsed = time.time() - t0
        sleep_t = max(0, frame_interval - elapsed)
        await asyncio.sleep(sleep_t)

    cap.release()
    state.running = False
    for ws in state.ws_clients:
        try:
            await ws.send_json({"type": "stopped"})
        except Exception:
            pass


# ─────────────────────────────────────────────
# WebSocket
# ─────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in state.ws_clients:
            state.ws_clients.remove(websocket)


# ─────────────────────────────────────────────
# Data Export
# ─────────────────────────────────────────────
@app.get("/api/events")
async def get_events():
    return [e.dict() for e in state.intrusion.events]


@app.get("/api/events/export")
async def export_events():
    import csv, io
    fieldnames = ["event_id", "track_id", "zone_id", "zone_name",
                  "timestamp", "confidence", "bbox"]
    output = io.StringIO()
    # extrasaction='ignore' 忽略 Pydantic 模型中多余字段（如 snapshot_path）
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for e in state.intrusion.events:
        d = e.dict()
        d["bbox"] = str(d["bbox"])
        writer.writerow(d)
    csv_content = output.getvalue()
    csv_path = LOGS_DIR / f"intrusion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path.write_text(csv_content, encoding="utf-8-sig")   # utf-8-sig 让 Excel 正确识别中文
    return FileResponse(str(csv_path), media_type="text/csv; charset=utf-8",
                        filename=csv_path.name)


@app.get("/api/snapshot/{event_id}")
async def get_snapshot(event_id: str):
    snap = state.intrusion.snapshots.get(event_id)
    if not snap:
        raise HTTPException(status_code=404, detail="快照不存在")
    return {"image": snap}


@app.post("/api/capture")
async def capture_frame():
    """保存当前检测帧"""
    if state.current_frame is None:
        raise HTTPException(status_code=400, detail="无当前帧")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = CAPTURES_DIR / f"capture_{ts}.jpg"
    cv2.imwrite(str(path), state.current_frame)
    return {"success": True, "path": str(path), "filename": path.name}


@app.get("/api/status")
async def get_status():
    return {
        "running": state.running,
        "has_video": state.video_path is not None,
        "has_model": state.detector is not None,
        "total_events": len(state.intrusion.events),
        "zones": len(state.config.zones),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8010, reload=False)