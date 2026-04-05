"""
Visualization helpers for crowd counting overlay.
Draws bounding boxes, track IDs, trajectory tails,
the virtual counting line, and direction indicators.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import colorsys


def _track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a stable, visually distinct BGR color per track_id."""
    hue = (track_id * 0.618033988749895) % 1.0  # golden ratio spread
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


def draw_detections(
    frame: np.ndarray,
    tracks,
    show_trajectory: bool = True,
    trajectory_len: int = 30,
) -> np.ndarray:
    """Draw tracked persons with bounding boxes and trajectory tails."""
    for track in tracks:
        color = _track_color(track.track_id)
        box = track.box.astype(int)
        x1, y1, x2, y2 = box

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"#{track.track_id}  {track.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Trajectory tail
        if show_trajectory and len(track.trajectory) >= 2:
            pts = list(track.trajectory)[-trajectory_len:]
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                c = tuple(int(v * alpha) for v in color)
                cv2.line(
                    frame,
                    (int(pts[i - 1][0]), int(pts[i - 1][1])),
                    (int(pts[i][0]), int(pts[i][1])),
                    c, 2, cv2.LINE_AA,
                )

        # Crossed indicator
        if track.counted or track.cross_direction is not None:
            cx, cy = int((x1 + x2) / 2), y1 - 20
            cv2.putText(
                frame, "✓", (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2, cv2.LINE_AA,
            )

    return frame


def draw_counting_line(
    frame: np.ndarray,
    start: Tuple[float, float],
    end: Tuple[float, float],
    count_in: int,
    count_out: int,
    line_color: Tuple[int, int, int] = (0, 220, 255),
    thickness: int = 3,
) -> np.ndarray:
    """
    Draw the virtual counting line with animated glow effect and counters.
    """
    x1, y1 = int(start[0]), int(start[1])
    x2, y2 = int(end[0]), int(end[1])

    # Glow effect: draw wider blurred line underneath
    overlay = frame.copy()
    cv2.line(overlay, (x1, y1), (x2, y2), line_color, thickness + 6)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # Main line
    cv2.line(frame, (x1, y1), (x2, y2), line_color, thickness, cv2.LINE_AA)

    # Endpoint circles
    cv2.circle(frame, (x1, y1), 7, line_color, -1)
    cv2.circle(frame, (x2, y2), 7, line_color, -1)

    # Direction arrow at midpoint
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    # Perpendicular direction for arrows
    dx, dy = x2 - x1, y2 - y1
    length = max(1, (dx ** 2 + dy ** 2) ** 0.5)
    perp_x, perp_y = -dy / length, dx / length  # perpendicular (in direction)

    arrow_len = 24
    # IN arrow (perpendicular, one direction)
    in_end = (int(mx + perp_x * arrow_len), int(my + perp_y * arrow_len))
    cv2.arrowedLine(frame, (mx, my), in_end, (50, 255, 120), 2, cv2.LINE_AA, tipLength=0.4)
    cv2.putText(frame, "IN", in_end, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 255, 120), 1, cv2.LINE_AA)

    # OUT arrow
    out_end = (int(mx - perp_x * arrow_len), int(my - perp_y * arrow_len))
    cv2.arrowedLine(frame, (mx, my), out_end, (80, 80, 255), 2, cv2.LINE_AA, tipLength=0.4)
    cv2.putText(frame, "OUT", out_end, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1, cv2.LINE_AA)

    return frame


def draw_hud(
    frame: np.ndarray,
    count_in: int,
    count_out: int,
    active_tracks: int,
    fps: float,
    frame_idx: int,
) -> np.ndarray:
    """Draw semi-transparent HUD with counters in top-left corner."""
    h, w = frame.shape[:2]
    hud_w, hud_h = 240, 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + hud_w, 8 + hud_h), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Border
    cv2.rectangle(frame, (8, 8), (8 + hud_w, 8 + hud_h), (0, 220, 255), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "CROWD COUNTER", (16, 28), font, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (16, 33), (8 + hud_w - 8, 33), (0, 220, 255), 1)
    cv2.putText(frame, f"IN  : {count_in:>4}", (16, 52), font, 0.52, (50, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, f"OUT : {count_out:>4}", (16, 72), font, 0.52, (80, 80, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"TOTAL: {count_in + count_out:>3}", (16, 92), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"TRACKS:{active_tracks}  FPS:{fps:.1f}  F:{frame_idx}",
        (16, 112), font, 0.38, (150, 150, 150), 1, cv2.LINE_AA,
    )

    return frame