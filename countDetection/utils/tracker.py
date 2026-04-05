"""
ByteTrack-inspired lightweight multi-object tracker.
Uses IoU-based matching with Kalman Filter for smooth trajectory estimation.
Designed for pedestrian tracking in crowd counting scenarios.
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict
import time


class KalmanFilter:
    """
    Kalman filter for 2D bounding box tracking.
    State: [cx, cy, w, h, vx, vy, vw, vh]
    Observation: [cx, cy, w, h]
    """

    def __init__(self):
        self.dt = 1.0
        # State transition matrix (8x8)
        self.F = np.eye(8, dtype=np.float32)
        self.F[0, 4] = self.dt
        self.F[1, 5] = self.dt
        self.F[2, 6] = self.dt
        self.F[3, 7] = self.dt

        # Observation matrix (4x8)
        self.H = np.eye(4, 8, dtype=np.float32)

        # Process noise covariance
        self.Q = np.eye(8, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01

        # Observation noise covariance
        self.R = np.eye(4, dtype=np.float32)
        self.R[2:, 2:] *= 10.0

        # Initial state covariance
        self.P0 = np.eye(8, dtype=np.float32)
        self.P0[4:, 4:] *= 1000.0

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cx, cy, w, h = measurement
        x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        P = self.P0.copy()
        return x, P

    def predict(self, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = self.F @ x
        P = self.F @ P @ self.F.T + self.Q
        return x, P

    def update(self, x: np.ndarray, P: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y = z - self.H @ x
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(8) - K @ self.H) @ P
        return x, P


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-6)


def iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix."""
    n, m = len(boxes1), len(boxes2)
    mat = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            mat[i, j] = iou(boxes1[i], boxes2[j])
    return mat


def hungarian_match(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Simple greedy matching (sufficient for real-time tracking)."""
    matched = []
    used_rows, used_cols = set(), set()
    # Sort by cost descending (we want max IoU)
    indices = np.argsort(-cost_matrix.ravel())
    for idx in indices:
        r, c = divmod(idx, cost_matrix.shape[1])
        if r not in used_rows and c not in used_cols:
            if cost_matrix[r, c] > 0:
                matched.append((r, c))
                used_rows.add(r)
                used_cols.add(c)
    return matched


class Track:
    """Single tracked object with Kalman state and trajectory history."""

    _next_id = 1

    def __init__(self, detection: np.ndarray, kf: KalmanFilter):
        self.track_id = Track._next_id
        Track._next_id += 1

        x1, y1, x2, y2, conf = detection
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1

        self.kf = kf
        self.x, self.P = kf.initiate(np.array([cx, cy, w, h]))
        self.hits = 1
        self.misses = 0
        self.is_confirmed = False
        self.conf = conf
        self.trajectory = deque(maxlen=60)  # keep last 60 positions
        self.trajectory.append((cx, cy))
        self.counted = False           # whether this track has been counted
        self.cross_direction: Optional[str] = None  # 'in' or 'out'
        self.first_side: Optional[str] = None       # which side of line first seen

    def predict(self):
        self.x, self.P = self.kf.predict(self.x, self.P)

    def update(self, detection: np.ndarray):
        x1, y1, x2, y2, conf = detection
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        self.x, self.P = self.kf.update(self.x, self.P, np.array([cx, cy, w, h]))
        self.conf = conf
        self.hits += 1
        self.misses = 0
        self.trajectory.append((cx, cy))
        if self.hits >= 3:
            self.is_confirmed = True

    def mark_missed(self):
        self.misses += 1

    @property
    def box(self) -> np.ndarray:
        """Current estimated bounding box [x1,y1,x2,y2]."""
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    @property
    def center(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])


class ByteTracker:
    """
    ByteTrack-inspired tracker.
    Two-stage association: high-conf detections first, then low-conf.
    Reference: Zhang et al., ByteTrack (ECCV 2022).
    """

    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        match_thresh: float = 0.3,
        max_misses: int = 30,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_misses = max_misses

        self.kf = KalmanFilter()
        self.tracks: List[Track] = []

    def update(self, detections: np.ndarray) -> List[Track]:
        """
        Args:
            detections: (N,5) array [x1,y1,x2,y2,conf]
        Returns:
            List of active confirmed tracks.
        """
        # Predict all existing tracks
        for t in self.tracks:
            t.predict()

        if len(detections) == 0:
            for t in self.tracks:
                t.mark_missed()
            self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
            return [t for t in self.tracks if t.is_confirmed]

        # Split detections by confidence
        high_mask = detections[:, 4] >= self.high_thresh
        low_mask = (detections[:, 4] >= self.low_thresh) & ~high_mask
        dets_high = detections[high_mask]
        dets_low = detections[low_mask]

        # Stage 1: match high-conf dets with all tracks
        unmatched_tracks, unmatched_dets_high = self._associate(
            self.tracks, dets_high
        )

        # Stage 2: match low-conf dets with unmatched tracks
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
        still_unmatched, _ = self._associate(remaining_tracks, dets_low)

        # Mark truly unmatched tracks as missed
        missed_track_indices = {self.tracks.index(remaining_tracks[i]) for i in still_unmatched}
        for i, t in enumerate(self.tracks):
            if i in missed_track_indices or (
                i in unmatched_tracks and i not in [self.tracks.index(remaining_tracks[j]) for j in range(len(remaining_tracks)) if j not in still_unmatched]
            ):
                t.mark_missed()

        # Create new tracks for unmatched high-conf detections
        for i in unmatched_dets_high:
            self.tracks.append(Track(dets_high[i], self.kf))

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        return [t for t in self.tracks if t.is_confirmed]

    def _associate(
        self,
        tracks: List[Track],
        detections: np.ndarray,
    ) -> Tuple[List[int], List[int]]:
        """Returns (unmatched_track_indices, unmatched_det_indices)."""
        if not tracks or len(detections) == 0:
            return list(range(len(tracks))), list(range(len(detections)))

        track_boxes = np.array([t.box for t in tracks])
        det_boxes = detections[:, :4]

        iou_mat = iou_matrix(track_boxes, det_boxes)
        matches = hungarian_match(iou_mat)

        matched_tracks, matched_dets = set(), set()
        for ti, di in matches:
            if iou_mat[ti, di] >= self.match_thresh:
                tracks[ti].update(detections[di])
                matched_tracks.add(ti)
                matched_dets.add(di)

        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        return unmatched_tracks, unmatched_dets

    def reset(self):
        self.tracks = []
        Track._next_id = 1