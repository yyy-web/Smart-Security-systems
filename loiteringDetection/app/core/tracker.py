"""
人员滞留跟踪核心算法
结合 ByteTrack 思想实现轻量级多目标跟踪 + 滞留检测

透视问题解决方案：
  1. 锚点模式（anchor_mode）：支持底部中心点、底部1/4处等，消除透视带来的判断偏差
  2. 鸟瞰变换（BEV Homography）：可选，通过单应性矩阵将目标点映射到俯视坐标系后再判断
  3. 区域Y轴偏移（zone_y_offset）：对所有区域统一向下平移，补偿绘制时的视觉误差
"""
import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque


# ─────────────────────────────────────────────
# 锚点模式常量
# ─────────────────────────────────────────────
ANCHOR_CENTER      = "center"       # bbox 中心（旧行为）
ANCHOR_FOOT        = "foot"         # bbox 底部中心（最接近站立点，推荐）
ANCHOR_FOOT_INNER  = "foot_inner"   # bbox 底部向上 1/4 处（鞋面区域，更鲁棒）
ANCHOR_HEAD        = "head"         # bbox 顶部中心（用于头顶区域检测）


def compute_anchor(bbox: List[float], mode: str = ANCHOR_FOOT) -> Tuple[float, float]:
    """
    根据 bbox [x1,y1,x2,y2] 和锚点模式计算判断点坐标。

    透视相机下，人站立在图像中的位置是 bbox 底部中心（foot），
    而非 bbox 中心。使用 foot 点可大幅减少"人明显在区域内但中心点
    却在区域外"的误判，尤其对俯仰角较大的摄像头效果最明显。

    foot_inner 是底部向上 10% 处，可以避免脚被裁剪时的误差。
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    h  = y2 - y1

    if mode == ANCHOR_FOOT:
        return (cx, y2)
    elif mode == ANCHOR_FOOT_INNER:
        return (cx, y2 - h * 0.10)
    elif mode == ANCHOR_HEAD:
        return (cx, y1)
    else:  # ANCHOR_CENTER
        return (cx, cy)


@dataclass
class TrackState:
    """单个跟踪目标的状态"""
    track_id: int
    bbox: List[float]           # [x1, y1, x2, y2]
    confidence: float
    class_name: str
    center: Tuple[float, float]

    # 锚点（用于区域判断，受 anchor_mode 控制）
    anchor: Tuple[float, float] = (0.0, 0.0)

    # 滞留相关
    in_zone: bool = False
    zone_id: Optional[int] = None
    enter_time: Optional[float] = None
    linger_duration: float = 0.0
    is_lingering: bool = False

    # 跟踪辅助
    lost_frames: int = 0
    speed: float = 0.0
    history_centers: deque = field(default_factory=lambda: deque(maxlen=30))
    last_seen: float = field(default_factory=time.time)

    def update_center(self, anchor_mode: str = ANCHOR_FOOT):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        self.anchor = compute_anchor(self.bbox, anchor_mode)
        # 速度用 foot 点轨迹，比中心点更稳定（不受 bbox 高度变化影响）
        self.history_centers.append(self.anchor)

    def compute_speed(self, fps: float = 25.0) -> float:
        """用锚点轨迹计算平均速度（像素/秒）"""
        if len(self.history_centers) < 2:
            self.speed = 0.0
            return self.speed
        recent = list(self.history_centers)[-min(10, len(self.history_centers)):]
        dists = []
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i-1][0]
            dy = recent[i][1] - recent[i-1][1]
            dists.append(math.sqrt(dx*dx + dy*dy))
        avg_dist_per_frame = sum(dists) / len(dists) if dists else 0
        self.speed = avg_dist_per_frame * fps
        return self.speed


class IoUTracker:
    """
    轻量级 IoU 跟踪器（不依赖外部库）
    支持：跟踪ID分配、短暂丢失容忍、速度计算、可配置锚点模式
    """

    def __init__(self, anchor_mode: str = ANCHOR_FOOT):
        self.tracks: Dict[int, TrackState] = {}
        self._next_id = 1
        self._iou_threshold = 0.3
        self.anchor_mode = anchor_mode   # 全局锚点模式，可在运行时修改

    def _iou(self, b1: List[float], b2: List[float]) -> float:
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: List[Dict], fps: float = 25.0) -> List[TrackState]:
        """
        detections: [{"bbox":[x1,y1,x2,y2], "confidence":float, "class_name":str}, ...]
        返回当前帧所有活跃 TrackState 列表
        """
        for t in self.tracks.values():
            t.lost_frames += 1

        matched_track_ids = set()
        matched_det_indices = set()

        track_ids = list(self.tracks.keys())
        for det_idx, det in enumerate(detections):
            best_iou = self._iou_threshold
            best_tid = None
            for tid in track_ids:
                if tid in matched_track_ids:
                    continue
                iou = self._iou(det["bbox"], self.tracks[tid].bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None:
                t = self.tracks[best_tid]
                t.bbox = det["bbox"]
                t.confidence = det["confidence"]
                t.class_name = det["class_name"]
                t.lost_frames = 0
                t.last_seen = time.time()
                t.update_center(self.anchor_mode)
                t.compute_speed(fps)
                matched_track_ids.add(best_tid)
                matched_det_indices.add(det_idx)

        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_indices:
                continue
            tid = self._next_id
            self._next_id += 1
            t = TrackState(
                track_id=tid,
                bbox=det["bbox"],
                confidence=det["confidence"],
                class_name=det["class_name"],
                center=(
                    (det["bbox"][0] + det["bbox"][2]) / 2,
                    (det["bbox"][1] + det["bbox"][3]) / 2,
                ),
                anchor=compute_anchor(det["bbox"], self.anchor_mode),
            )
            t.history_centers.append(t.anchor)
            self.tracks[tid] = t

        return list(self.tracks.values())

    def remove_lost(self, max_lost_frames: int):
        to_del = [tid for tid, t in self.tracks.items() if t.lost_frames > max_lost_frames]
        for tid in to_del:
            del self.tracks[tid]

    def get_active_tracks(self) -> List[TrackState]:
        return [t for t in self.tracks.values() if t.lost_frames == 0]


# ─────────────────────────────────────────────
# 鸟瞰变换辅助类
# ─────────────────────────────────────────────
class BEVTransform:
    """
    单应性矩阵（Homography）鸟瞰变换。

    使用方法：
      1. 在摄像头画面上选 4 个地面点（src_pts）
      2. 指定这 4 个点对应的俯视图坐标（dst_pts，通常是一个矩形）
      3. 调用 set_points() 计算矩阵
      4. 用 transform_point() 将任意图像点映射到俯视坐标
      5. 在俯视坐标系下绘制区域、做包含判断，精度大幅提升

    典型标定流程（4角点）：
      摄像头视图（梯形）       →  俯视图（矩形）
      左上(近)  右上(近)           (0,0)  (W,0)
      左下(远)  右下(远)           (0,H)  (W,H)
    """

    def __init__(self):
        self._M: Optional[np.ndarray] = None        # 3×3 单应性矩阵
        self._M_inv: Optional[np.ndarray] = None    # 逆矩阵（俯视→图像）
        self.src_pts: List[List[float]] = []         # 图像坐标系中的 4 个点
        self.dst_pts: List[List[float]] = []         # 俯视坐标系中对应的 4 个点
        self.bev_w: int = 400
        self.bev_h: int = 400

    def set_points(
        self,
        src_pts: List[List[float]],
        dst_pts: Optional[List[List[float]]] = None,
        bev_w: int = 400,
        bev_h: int = 400,
    ) -> bool:
        """
        src_pts: 图像坐标系 4 个角点 [[x,y], ...]，顺序：左上、右上、右下、左下
        dst_pts: 俯视坐标系 4 个对应点，默认自动生成矩形 [(0,0),(W,0),(W,H),(0,H)]
        """
        if len(src_pts) != 4:
            return False
        self.src_pts = src_pts
        self.bev_w = bev_w
        self.bev_h = bev_h
        if dst_pts is None:
            dst_pts = [[0, 0], [bev_w, 0], [bev_w, bev_h], [0, bev_h]]
        self.dst_pts = dst_pts
        try:
            src = np.float32(src_pts)
            dst = np.float32(dst_pts)
            self._M     = cv2_getPerspectiveTransform(src, dst)
            self._M_inv = cv2_getPerspectiveTransform(dst, src)
            return True
        except Exception:
            return False

    def transform_point(self, pt: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """将图像坐标映射到俯视坐标"""
        if self._M is None:
            return None
        p = np.float32([[pt]])
        out = _perspective_transform_point(p, self._M)
        return (float(out[0]), float(out[1]))

    def transform_zone(
        self, polygon: List[Tuple[int, int]]
    ) -> Optional[List[Tuple[float, float]]]:
        """将图像坐标系中的多边形顶点变换到俯视坐标系"""
        if self._M is None:
            return None
        result = []
        for pt in polygon:
            tp = self.transform_point(pt)
            if tp is None:
                return None
            result.append(tp)
        return result

    @property
    def is_ready(self) -> bool:
        return self._M is not None


def cv2_getPerspectiveTransform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """封装 OpenCV getPerspectiveTransform，延迟导入避免顶层依赖"""
    import cv2
    return cv2.getPerspectiveTransform(src, dst)


def _perspective_transform_point(
    pt: np.ndarray, M: np.ndarray
) -> Tuple[float, float]:
    """对单个点应用透视变换矩阵"""
    import cv2
    out = cv2.perspectiveTransform(pt, M)
    return (out[0][0][0], out[0][0][1])


# ─────────────────────────────────────────────
# 滞留检测器
# ─────────────────────────────────────────────
class ZoneLingerDetector:
    """
    滞留区域检测器：管理多个多边形区域的进出时间和滞留判定。

    透视补偿参数：
      zone_y_offset  : 所有区域统一向下平移的像素数（补偿绘制时偏高的问题）
      bev_transform  : 可选的鸟瞰变换对象，启用后在俯视坐标系做判断
    """

    def __init__(self):
        self.zones: Dict[int, List[Tuple[int, int]]] = {}
        self._linger_records: Dict[str, dict] = {}
        self.zone_y_offset: float = 0.0          # 像素，正值=向下平移
        self.bev: Optional[BEVTransform] = None  # 若不为 None 则启用 BEV 判断

    def set_zones(self, zones: Dict[int, List[Tuple[int, int]]]):
        self.zones = zones

    def _apply_offset(
        self, polygon: List[Tuple[int, int]]
    ) -> List[Tuple[float, float]]:
        """对多边形顶点应用 Y 轴偏移"""
        if self.zone_y_offset == 0:
            return polygon
        return [(x, y + self.zone_y_offset) for x, y in polygon]

    def _effective_zone_and_point(
        self,
        polygon: List[Tuple[int, int]],
        anchor: Tuple[float, float],
    ) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
        """
        根据配置返回（用于判断的区域多边形, 用于判断的目标点）。
        - 若启用 BEV：把区域顶点和目标点都变换到俯视坐标系
        - 否则：对区域做 Y 轴偏移，目标点保持不变
        """
        if self.bev and self.bev.is_ready:
            bev_poly = self.bev.transform_zone(polygon)
            bev_pt   = self.bev.transform_point(anchor)
            if bev_poly and bev_pt:
                return bev_poly, bev_pt
        # fallback: 仅做 Y 轴偏移
        return self._apply_offset(polygon), anchor

    def _point_in_polygon(
        self,
        point: Tuple[float, float],
        polygon: List[Tuple[float, float]],
    ) -> bool:
        """射线法判断点是否在多边形内（支持浮点坐标）"""
        x, y = point
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
            ):
                inside = not inside
            j = i
        return inside

    def update(
        self,
        tracks: List[TrackState],
        linger_threshold: float,
        speed_threshold: float,
        tolerance_frames: int,
        fps: float,
    ) -> List[dict]:
        """
        更新每个轨迹在各区域的滞留状态。

        时间计算逻辑（修正版）
        ─────────────────────────────────────────────────────────
        在区域内时：
          • 持续更新 last_active = now
          • in_zone_duration = now - enter_time  （实时预览用）
          • is_lingering = in_zone_duration >= linger_threshold

        离开区域时（超出容忍窗口后）：
          • final_duration = last_active - enter_time
            这才是"人在区域内停留的总时长"
          • 只有 final_duration >= linger_threshold 才算一次有效滞留
          • 触发 "linger_complete" 事件，携带 final_duration
          • 重置 TrackState（in_zone=False, linger_duration 保留 final_duration 供展示）

        返回值：本次 update 中新产生的"完整滞留"事件列表
        每个事件格式：
          {"track_id": int, "zone_id": int,
           "enter_time": float, "leave_time": float, "final_duration": float}
        ─────────────────────────────────────────────────────────
        """
        now = time.time()
        completed_events: List[dict] = []   # 本帧新完成的滞留事件

        for track in tracks:
            for zone_id, polygon in self.zones.items():
                key = f"{track.track_id}_{zone_id}"

                eff_poly, eff_pt = self._effective_zone_and_point(polygon, track.anchor)
                in_zone  = self._point_in_polygon(eff_pt, eff_poly)
                speed_ok = (speed_threshold <= 0) or (track.speed < speed_threshold)

                if in_zone and speed_ok and track.lost_frames == 0:
                    # ── 目标在区域内 ──────────────────────────
                    if key not in self._linger_records:
                        self._linger_records[key] = {
                            "enter_time":  now,
                            "last_active": now,
                            "completed":   False,   # 该 key 是否已经触发过完整事件
                        }
                    else:
                        self._linger_records[key]["last_active"] = now

                    record = self._linger_records[key]
                    # 实时预览时长 = now - enter_time（人还在区域里，时间还在跑）
                    in_zone_duration = now - record["enter_time"]

                    # 以最长滞留区域为准更新 TrackState
                    if in_zone_duration > track.linger_duration:
                        track.in_zone        = True
                        track.zone_id        = zone_id
                        track.enter_time     = record["enter_time"]
                        track.linger_duration = in_zone_duration   # 实时预览值
                        track.is_lingering   = in_zone_duration >= linger_threshold

                else:
                    # ── 目标不在区域（或速度过高 / 丢失）─────────
                    tolerance_seconds = tolerance_frames / max(fps, 1)
                    if key in self._linger_records:
                        record = self._linger_records[key]
                        elapsed_since_active = now - record["last_active"]

                        if elapsed_since_active > tolerance_seconds:
                            # 超出容忍窗口 → 确认离开，计算最终时长
                            leave_time     = record["last_active"]   # 最后一次在区域内的时刻
                            final_duration = leave_time - record["enter_time"]

                            # 只有达到阈值且尚未触发过完整事件才上报
                            if final_duration >= linger_threshold and not record["completed"]:
                                record["completed"] = True
                                completed_events.append({
                                    "track_id":       track.track_id,
                                    "zone_id":        zone_id,
                                    "enter_time":     record["enter_time"],
                                    "leave_time":     leave_time,
                                    "final_duration": round(final_duration, 1),
                                })

                            # 清除记录，重置 TrackState
                            del self._linger_records[key]
                            if track.zone_id == zone_id:
                                track.in_zone        = False
                                track.zone_id        = None
                                track.enter_time     = None
                                # ★ 保留 final_duration 供前端展示最后一次的时长
                                track.linger_duration = round(final_duration, 1) \
                                    if final_duration >= linger_threshold else 0.0
                                track.is_lingering   = False

        # 清理已消失轨迹的残留记录
        active_ids = {t.track_id for t in tracks}
        stale_keys = [k for k in list(self._linger_records)
                      if int(k.split("_")[0]) not in active_ids]
        for k in stale_keys:
            record = self._linger_records[k]
            # 目标消失也视为"离开"，同样触发完整事件
            final_duration = record["last_active"] - record["enter_time"]
            if final_duration >= linger_threshold and not record["completed"]:
                tid, zid = k.split("_", 1)
                completed_events.append({
                    "track_id":       int(tid),
                    "zone_id":        int(zid),
                    "enter_time":     record["enter_time"],
                    "leave_time":     record["last_active"],
                    "final_duration": round(final_duration, 1),
                })
            del self._linger_records[k]

        return completed_events

    def reset(self):
        self._linger_records.clear()