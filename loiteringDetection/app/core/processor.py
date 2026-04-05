"""
Video processor: inference + tracking + linger detection + frame drawing
"""
import cv2
import time
import base64
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from core.inference import YOLOv8ONNXInference
from core.tracker import (
    IoUTracker, ZoneLingerDetector, TrackState, BEVTransform,
    ANCHOR_FOOT, ANCHOR_CENTER, ANCHOR_FOOT_INNER, ANCHOR_HEAD,
)

logger = logging.getLogger(__name__)

COLORS = [
    (0, 200, 255), (0, 255, 128), (255, 100, 0), (180, 0, 255),
    (0, 255, 200), (255, 200, 0), (100, 255, 0), (255, 0, 180),
    (0, 180, 255), (200, 255, 0),
]
LINGER_COLOR   = (0, 0, 255)
ZONE_COLORS    = [(0,255,255),(0,255,128),(255,128,0),(128,0,255),(0,200,255),(255,0,128)]
ANCHOR_COLOR   = (0, 255, 0)     # 锚点始终用鲜绿色，和目标框颜色区分
ANCHOR_LINE_COLOR = (0, 200, 0)  # 水平辅助线颜色


def track_color(tid: int) -> Tuple[int, int, int]:
    return COLORS[tid % len(COLORS)]


@dataclass
class DetectionResult:
    track_id: int
    confidence: float
    class_name: str
    bbox: List[float]
    center: Tuple[float, float]
    anchor: Tuple[float, float]
    is_lingering: bool
    zone_id: Optional[int]
    linger_duration: float
    speed: float
    thumb_b64: str = ""


@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    detections: List[DetectionResult]
    linger_alerts: List[dict]    # 完整事件：人已离开，duration = leave - enter
    linger_previews: List[dict]  # 实时预览：人还在区域内且已超阈值
    frame_b64: str
    anchor_points: List[Tuple[float, float]] = field(default_factory=list)


class VideoProcessor:
    def __init__(self):
        self.model: Optional[YOLOv8ONNXInference] = None
        self.tracker  = IoUTracker(anchor_mode=ANCHOR_FOOT)
        self.zone_detector = ZoneLingerDetector()
        self.bev = BEVTransform()
        self.cap: Optional[cv2.VideoCapture] = None

        # Detection params
        self.conf_threshold: float = 0.4
        self.min_area: float = 500
        self.linger_threshold: float = 30.0
        self.tolerance_frames: int = 30
        self.speed_threshold: float = 50.0

        # Perspective params — BEV intentionally OFF by default
        self.anchor_mode: str = ANCHOR_FOOT
        self.zone_y_offset: float = 0.0
        self.use_bev: bool = False   # 大多数场景不需要BEV，默认关闭

        # Display switches
        self.show_track_id: bool   = True
        self.show_class: bool      = True
        self.show_conf: bool       = True
        self.show_trail: bool      = True
        # ★ 锚点始终显示，不受开关隐藏——这是区域对齐的核心视觉反馈
        self.show_anchor: bool     = True
        # ★ 是否在帧上画"锚点水平辅助线"，帮助用户知道区域下边界应该画到哪里
        self.show_anchor_hline: bool = True
        self.show_linger_count: bool = True
        self.show_bev_points: bool   = False

        self.linger_log: List[dict] = []
        self._fps: float  = 25.0
        self._frame_idx: int = 0
        self._zones: Dict[int, List[Tuple[int, int]]] = {}
        self._notified: set = set()

    # ── config ──────────────────────────────────
    def load_model(self, path: str):
        self.model = YOLOv8ONNXInference(path, self.conf_threshold)
        logger.info("Model loaded: %s", path)

    def open_video(self, path: str) -> Dict:
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w     = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return {"fps": self._fps, "total_frames": total, "width": w, "height": h}

    def set_zones(self, zones: Dict[int, List[Tuple[int, int]]]):
        self._zones = zones
        self.zone_detector.set_zones(zones)
        self._sync_perspective()

    def set_bev(self, src_pts, bev_w=400, bev_h=400) -> bool:
        ok = self.bev.set_points(src_pts, bev_w=bev_w, bev_h=bev_h)
        if ok and self.use_bev:
            self.zone_detector.bev = self.bev
        return ok

    def _sync_perspective(self):
        """把透视参数同步到 tracker 和 zone_detector"""
        self.tracker.anchor_mode = self.anchor_mode
        self.zone_detector.zone_y_offset = self.zone_y_offset
        # BEV：只在明确启用且已标定时生效
        if self.use_bev and self.bev.is_ready:
            self.zone_detector.bev = self.bev
        else:
            self.zone_detector.bev = None

    def update_perspective_params(self):
        self._sync_perspective()

    def reset(self):
        self.tracker = IoUTracker(anchor_mode=self.anchor_mode)
        self.zone_detector.reset()
        self.linger_log.clear()
        self._notified.clear()
        self._frame_idx = 0
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── frame pipeline ──────────────────────────
    def process_frame(self, frame: np.ndarray) -> Optional[FrameResult]:
        if self.model is None:
            return None

        self.model.conf_threshold = self.conf_threshold
        self._sync_perspective()

        detections_raw = self.model.infer(frame, min_area=self.min_area)
        detections_raw = [d for d in detections_raw if d["class_id"] == 0]

        all_tracks = self.tracker.update(detections_raw, self._fps)
        self.tracker.remove_lost(self.tolerance_frames)

        # update() 现在返回本帧"完整滞留"事件列表
        # 完整事件 = 人已经真正离开区域，final_duration = leave_time - enter_time
        completed_events = self.zone_detector.update(
            all_tracks,
            linger_threshold=self.linger_threshold,
            speed_threshold=self.speed_threshold,
            tolerance_frames=self.tolerance_frames,
            fps=self._fps,
        )

        # ── 构建本帧报警列表 ──────────────────────────────────────
        # 两类事件都推送给前端：
        #   1. linger_alerts   : 完整事件（人已离开，final_duration 是真实滞留时长）
        #   2. linger_previews : 人还在区域内且已超阈值（实时预览，时长仍在增长）
        new_alerts   = []   # 完整事件（含最终时长）
        new_previews = []   # 实时预览事件（人还在区域内）

        # 完整事件：从 completed_events 里取，为每个事件找当前帧的缩略图
        # （此时目标可能已离开画面，尽量用最近帧的图）
        track_thumb_cache: dict = {}
        for t in all_tracks:
            if t.lost_frames == 0:
                track_thumb_cache[t.track_id] = self._crop_thumb(frame, t.bbox)

        for ev in completed_events:
            thumb = track_thumb_cache.get(ev["track_id"], "")
            alert = {
                "track_id":      ev["track_id"],
                "zone_id":       ev["zone_id"],
                "duration":      ev["final_duration"],   # ★ 真实滞留时长
                "enter_time":    time.strftime("%H:%M:%S",
                                     time.localtime(ev["enter_time"])),
                "leave_time":    time.strftime("%H:%M:%S",
                                     time.localtime(ev["leave_time"])),
                "time":          time.strftime("%H:%M:%S"),
                "thumb_b64":     thumb,
                "is_complete":   True,
            }
            self.linger_log.append(alert)
            new_alerts.append(alert)

        # 实时预览：人还在区域内且已超阈值，首次发现时推送一次"进行中"通知
        for t in all_tracks:
            if t.is_lingering and t.lost_frames == 0:
                if t.track_id not in self._notified:
                    self._notified.add(t.track_id)
                    thumb = track_thumb_cache.get(t.track_id, "")
                    new_previews.append({
                        "track_id":    t.track_id,
                        "zone_id":     t.zone_id,
                        "duration":    round(t.linger_duration, 1),  # 实时，还在增长
                        "time":        time.strftime("%H:%M:%S"),
                        "thumb_b64":   thumb,
                        "is_complete": False,   # 标记：人还没离开
                    })
            else:
                # 人离开后允许再次触发预览（下次进入区域）
                self._notified.discard(t.track_id)

        vis = self._draw(frame.copy(), all_tracks)
        _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_b64 = base64.b64encode(buf).decode()

        det_results = []
        anchor_points = []
        for t in all_tracks:
            if t.lost_frames > 0:
                continue
            anchor_points.append((round(t.anchor[0], 1), round(t.anchor[1], 1)))
            thumb = self._crop_thumb(frame, t.bbox)
            det_results.append(DetectionResult(
                track_id=t.track_id,
                confidence=round(t.confidence, 3),
                class_name=t.class_name,
                bbox=[round(v, 1) for v in t.bbox],
                center=(round(t.center[0], 1), round(t.center[1], 1)),
                anchor=(round(t.anchor[0], 1), round(t.anchor[1], 1)),
                is_lingering=t.is_lingering,
                zone_id=t.zone_id,
                linger_duration=round(t.linger_duration, 1),
                speed=round(t.speed, 1),
                thumb_b64=thumb,
            ))

        self._frame_idx += 1
        return FrameResult(
            frame_index=self._frame_idx,
            timestamp=self._frame_idx / max(self._fps, 1),
            detections=det_results,
            linger_alerts=new_alerts,       # 完整事件（已离开，时长已定）
            linger_previews=new_previews,   # 实时预览（还在区域内）
            frame_b64=frame_b64,
            anchor_points=anchor_points,
        )

    def read_frame(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def get_preview_frame(self) -> Optional[str]:
        if self.cap is None:
            return None
        pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        if not ret:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode()

    # ── drawing ─────────────────────────────────
    def _draw(self, frame: np.ndarray, tracks: List[TrackState]) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # ── 绘制滞留区域（含 Y 偏移） ──
        for i, (zone_id, polygon) in enumerate(self._zones.items()):
            color   = ZONE_COLORS[i % len(ZONE_COLORS)]
            shifted = [(int(px), int(py + self.zone_y_offset)) for px, py in polygon]
            pts     = np.array(shifted, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
            overlay = frame.copy()
            cv2.polylines(frame, [pts], True, color, 2)
            # 区域标签
            cx_z = int(sum(p[0] for p in shifted) / len(shifted))
            cy_z = int(sum(p[1] for p in shifted) / len(shifted))
            cv2.putText(frame, f"Zone {zone_id}", (cx_z - 30, cy_z),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # ── HUD：滞留人数 ──
        linger_count = sum(1 for t in tracks if t.is_lingering and t.lost_frames == 0)
        if self.show_linger_count:
            cv2.rectangle(frame, (10, 10), (240, 48), (0, 0, 0), -1)
            cv2.putText(frame, f"Lingering: {linger_count}", (15, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ── HUD：右上角锚点模式提示 ──
        anchor_label_map = {
            ANCHOR_FOOT:       "Anchor: foot (bottom center)",
            ANCHOR_FOOT_INNER: "Anchor: foot-inner",
            ANCHOR_CENTER:     "Anchor: center",
            ANCHOR_HEAD:       "Anchor: head",
        }
        alabel = anchor_label_map.get(self.anchor_mode, "")
        cv2.putText(frame, alabel, (w - 340, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

        # ── 提示文字：绘区域时看绿点 ──
        if not self._zones:
            msg = "Tip: green dots = anchor points  <-- draw zone bottom edge here"
            cv2.putText(frame, msg, (10, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

        # ── 每个目标 ──
        active_anchor_ys = []   # 收集所有锚点Y，用于画水平参考线

        for t in tracks:
            if t.lost_frames > 0:
                continue
            x1, y1, x2, y2 = [int(v) for v in t.bbox]
            color = LINGER_COLOR if t.is_lingering else track_color(t.track_id)
            ax, ay = int(t.anchor[0]), int(t.anchor[1])
            active_anchor_ys.append(ay)

            # 轨迹线
            if self.show_trail and len(t.history_centers) > 1:
                pts_trail = [(int(c[0]), int(c[1])) for c in t.history_centers]
                for k in range(1, len(pts_trail)):
                    alpha = k / len(pts_trail)
                    tc = tuple(int(c * alpha) for c in color)
                    cv2.line(frame, pts_trail[k-1], pts_trail[k], tc, 1)

            # 边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ★ 锚点 —— 绿色实心圆 + 外圈 + 十字，非常醒目
            # 这是用户绘制区域时最重要的视觉参考
            cv2.circle(frame, (ax, ay), 7, ANCHOR_COLOR, -1)
            cv2.circle(frame, (ax, ay), 9, (255, 255, 255), 1)
            cv2.line(frame, (ax - 14, ay), (ax + 14, ay), ANCHOR_COLOR, 1)
            cv2.line(frame, (ax, ay - 14), (ax, ay + 14), ANCHOR_COLOR, 1)

            # ★ 锚点 Y 坐标的水平虚线 —— 跨越整个画面，告诉用户区域底部该画在哪
            if self.show_anchor_hline:
                # 虚线效果：每 8px 画 4px
                x_cur = 0
                while x_cur < w:
                    x_end = min(x_cur + 4, w)
                    cv2.line(frame, (x_cur, ay), (x_end, ay), ANCHOR_LINE_COLOR, 1)
                    x_cur += 8
                cv2.putText(frame, f"foot y={ay}", (w - 110, ay - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, ANCHOR_LINE_COLOR, 1)

            # 标签（ID / 类别 / 置信度）
            label_parts = []
            if self.show_track_id: label_parts.append(f"#{t.track_id}")
            if self.show_class:    label_parts.append(t.class_name)
            if self.show_conf:     label_parts.append(f"{t.confidence:.2f}")
            label = " ".join(label_parts)
            if label:
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                ly = max(y1 - 5, lh + 5)
                cv2.rectangle(frame, (x1, ly - lh - 4), (x1 + lw + 4, ly + 2), color, -1)
                cv2.putText(frame, label, (x1 + 2, ly - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # 滞留时长标注
            if t.is_lingering:
                dur_label = f"Linger {t.linger_duration:.1f}s Zone{t.zone_id}"
                cv2.putText(frame, dur_label, (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, LINGER_COLOR, 1)

        return frame

    def _crop_thumb(self, frame: np.ndarray, bbox: List[float], size: int = 64) -> str:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return ""
        crop = cv2.resize(frame[y1:y2, x1:x2], (size, size))
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf).decode()