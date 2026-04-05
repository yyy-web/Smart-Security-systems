"""
Line-crossing crowd flow counter.

Algorithm:
  1. Define a virtual counting line by two endpoints.
  2. For each confirmed track, record which side of the line
     the track center is on (computed via cross-product sign).
  3. When the side flips AND the track has been seen for ≥ min_hits
     frames on each side, increment the counter and mark the track.
  4. Direction (IN / OUT) is determined by the sign of the crossing.

Cross-product side test:
  Given line A→B and point P:
    side = sign((B-A) × (P-A))
  +1  → left  side  (we call it "in")
  -1  → right side  (we call it "out")
"""

import math
from typing import Tuple, Dict, Optional, List
import numpy as np
from dataclasses import dataclass, field
import time


@dataclass
class CrossingEvent:
    track_id: int
    direction: str          # 'in' or 'out'
    timestamp: float
    position: Tuple[float, float]
    frame_index: int


@dataclass
class CounterState:
    count_in: int = 0
    count_out: int = 0
    events: List[CrossingEvent] = field(default_factory=list)


def _side(ax, ay, bx, by, px, py) -> int:
    """Return +1 or -1 based on which side of A→B the point P lies."""
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if cross > 0:
        return 1
    elif cross < 0:
        return -1
    else:
        return 0   # exactly on the line (rare)


class LineCounter:
    """
    Virtual-line crossing counter.

    Parameters
    ----------
    line_start, line_end : (x, y) in pixel coords (can be relative 0-1 or absolute)
    min_confirm_frames   : track must be seen this many frames before counting
    crossing_buffer      : minimum frames between two counts for same track (safety)
    """

    def __init__(
        self,
        line_start: Tuple[float, float],
        line_end: Tuple[float, float],
        min_confirm_frames: int = 3,
    ):
        self.line_start = line_start
        self.line_end = line_end
        self.min_confirm_frames = min_confirm_frames

        self.state = CounterState()
        # Maps track_id → last known side (+1/-1)
        self._track_side: Dict[int, int] = {}
        # Maps track_id → frames seen on current side
        self._track_frames_on_side: Dict[int, int] = {}

    # ------------------------------------------------------------------
    def update_line(self, start: Tuple[float, float], end: Tuple[float, float]):
        self.line_start = start
        self.line_end = end

    def reset(self):
        self.state = CounterState()
        self._track_side.clear()
        self._track_frames_on_side.clear()

    # ------------------------------------------------------------------
    def update(
        self,
        tracks,                  # list of Track objects (from tracker.py)
        frame_index: int,
        timestamp: Optional[float] = None,
    ) -> List[CrossingEvent]:
        """
        Process one frame of confirmed tracks.
        Returns list of new crossing events this frame.
        """
        if timestamp is None:
            timestamp = time.time()

        ax, ay = self.line_start
        bx, by = self.line_end
        new_events: List[CrossingEvent] = []

        active_ids = set()

        for track in tracks:
            tid = track.track_id
            cx, cy = track.center
            active_ids.add(tid)

            current_side = _side(ax, ay, bx, by, cx, cy)
            if current_side == 0:
                continue

            prev_side = self._track_side.get(tid, 0)

            if prev_side == 0:
                # First observation — just record side
                self._track_side[tid] = current_side
                self._track_frames_on_side[tid] = 1
                continue

            if current_side == prev_side:
                self._track_frames_on_side[tid] = (
                    self._track_frames_on_side.get(tid, 0) + 1
                )
            else:
                # Side flip detected — count if track is confirmed
                frames_on_old = self._track_frames_on_side.get(tid, 0)
                if frames_on_old >= self.min_confirm_frames and track.hits >= self.min_confirm_frames:
                    # Determine direction
                    # +1→-1 : crossed from left to right → "out"
                    # -1→+1 : crossed from right to left → "in"
                    direction = "out" if prev_side == 1 else "in"

                    event = CrossingEvent(
                        track_id=tid,
                        direction=direction,
                        timestamp=timestamp,
                        position=(cx, cy),
                        frame_index=frame_index,
                    )
                    new_events.append(event)
                    self.state.events.append(event)

                    if direction == "in":
                        self.state.count_in += 1
                    else:
                        self.state.count_out += 1

                # Update to new side
                self._track_side[tid] = current_side
                self._track_frames_on_side[tid] = 1

        # Clean up stale tracks
        gone = set(self._track_side.keys()) - active_ids
        for tid in gone:
            self._track_side.pop(tid, None)
            self._track_frames_on_side.pop(tid, None)

        return new_events

    # ------------------------------------------------------------------
    @property
    def total(self) -> int:
        return self.state.count_in + self.state.count_out

    @property
    def count_in(self) -> int:
        return self.state.count_in

    @property
    def count_out(self) -> int:
        return self.state.count_out

    def get_summary(self) -> dict:
        return {
            "total": self.total,
            "in": self.count_in,
            "out": self.count_out,
            "events": [
                {
                    "track_id": e.track_id,
                    "direction": e.direction,
                    "timestamp": e.timestamp,
                    "frame": e.frame_index,
                }
                for e in self.state.events[-50:]  # last 50 events
            ],
        }