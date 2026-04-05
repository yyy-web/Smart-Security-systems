"""
Video processing pipeline: orchestrates detector → tracker → counter → visualizer.
Runs in a background thread, streams JPEG frames via an asyncio queue.
"""

import cv2
import numpy as np
import threading
import time
import logging
import base64
from typing import Optional, Tuple, Callable, Dict, Any
from pathlib import Path

from utils.detector import YOLOv8Detector
from utils.tracker import ByteTracker
from utils.counter import LineCounter
from utils.visualizer import draw_detections, draw_counting_line, draw_hud

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".mp4", ".avi", ".flv", ".mov", ".mkv", ".wmv", ".ts", ".m4v"}


class PipelineState:
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class ProcessingPipeline:
    """
    Thread-based video processing pipeline.
    Provides frame callback and event callback interfaces.
    """

    def __init__(self):
        self.detector: Optional[YOLOv8Detector] = None
        self.tracker: Optional[ByteTracker] = None
        self.counter: Optional[LineCounter] = None

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused by default

        self.state = PipelineState.IDLE

        # Line in relative coords (0–1) → converted per-frame
        self.line_rel: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

        # Callbacks
        self.on_frame: Optional[Callable[[bytes, Dict], None]] = None
        self.on_event: Optional[Callable[[Dict], None]] = None
        self.on_state: Optional[Callable[[str], None]] = None

        # Stats
        self.frame_index = 0
        self.fps_measured = 0.0
        self._fps_window = []

        # Current video info
        self.video_path: Optional[str] = None
        self.video_info: Dict = {}

    # ------------------------------------------------------------------
    def load_model(self, model_path: str, conf: float = 0.35, iou: float = 0.45):
        suffix = Path(model_path).suffix.lower()
        if suffix != ".onnx":
            raise ValueError("Only ONNX models are supported.")
        self.detector = YOLOv8Detector(model_path, conf_threshold=conf, iou_threshold=iou)
        self.tracker = ByteTracker(high_thresh=conf, low_thresh=0.1)
        logger.info(f"Model loaded: {model_path}")

    def load_video(self, video_path: str) -> Dict:
        suffix = Path(video_path).suffix.lower()
        if suffix not in SUPPORTED_EXTS:
            raise ValueError(f"Unsupported video format: {suffix}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        # Read first frame for preview
        ret, first_frame = cap.read()
        first_frame_b64 = None
        if ret and first_frame is not None:
            _, jpeg = cv2.imencode(".jpg", first_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            first_frame_b64 = base64.b64encode(jpeg.tobytes()).decode()
        cap.release()
        self.video_path = video_path
        self.video_info = info
        logger.info(f"Video loaded: {video_path} {info}")
        info["first_frame"] = first_frame_b64
        return info

    def set_counting_line(self, rel_start: Tuple[float, float], rel_end: Tuple[float, float]):
        """Set line in relative [0,1] coordinates. Converted to pixels per frame."""
        self.line_rel = (rel_start, rel_end)
        if self.counter is not None:
            # Will be updated each frame; reset cross state
            self.counter.reset()

    # ------------------------------------------------------------------
    def start(self):
        if self.state == PipelineState.RUNNING:
            return
        if self.detector is None:
            raise RuntimeError("No model loaded.")
        if self.video_path is None:
            raise RuntimeError("No video loaded.")

        self._stop_event.clear()
        self._pause_event.set()

        # Reset line counter
        line_start = (0.2, 0.5) if self.line_rel is None else self.line_rel[0]
        line_end = (0.8, 0.5) if self.line_rel is None else self.line_rel[1]
        self.counter = LineCounter(line_start, line_end)
        self.tracker.reset()
        self.frame_index = 0

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.state = PipelineState.RUNNING
        if self.on_state:
            self.on_state(self.state)

    def stop(self):
        self._stop_event.set()
        self._pause_event.set()  # unblock if paused
        if self._thread:
            self._thread.join(timeout=5)
        self.state = PipelineState.STOPPED
        if self.on_state:
            self.on_state(self.state)

    def pause(self):
        self._pause_event.clear()
        self.state = PipelineState.PAUSED
        if self.on_state:
            self.on_state(self.state)

    def resume(self):
        self._pause_event.set()
        self.state = PipelineState.RUNNING
        if self.on_state:
            self.on_state(self.state)

    # ------------------------------------------------------------------
    def _run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.state = PipelineState.ERROR
            return

        target_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_delay = 1.0 / target_fps

        try:
            while not self._stop_event.is_set():
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break

                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    # End of video — emit final stats and stop
                    if self.on_event:
                        self.on_event({"type": "video_end", "summary": self.counter.get_summary()})
                    break

                frame = self._process_frame(frame)
                self.frame_index += 1

                # Measure FPS
                elapsed = time.time() - t0
                self._fps_window.append(elapsed)
                if len(self._fps_window) > 30:
                    self._fps_window.pop(0)
                self.fps_measured = 1.0 / (sum(self._fps_window) / len(self._fps_window) + 1e-6)

                # Encode and emit
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if self.on_frame:
                    stats = {
                        "count_in": self.counter.count_in,
                        "count_out": self.counter.count_out,
                        "total": self.counter.total,
                        "frame": self.frame_index,
                        "fps": round(self.fps_measured, 1),
                    }
                    self.on_frame(jpeg.tobytes(), stats)

                # Throttle to video fps
                sleep_time = frame_delay - (time.time() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            self.state = PipelineState.ERROR
        finally:
            cap.release()
            if self.state != PipelineState.ERROR:
                self.state = PipelineState.STOPPED
            if self.on_state:
                self.on_state(self.state)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        H, W = frame.shape[:2]

        # 1. Detect persons
        detections = self.detector.detect(frame)

        # 2. Track
        tracks = self.tracker.update(detections)

        # 3. Convert relative line to pixel coords
        if self.line_rel is not None:
            (rx1, ry1), (rx2, ry2) = self.line_rel
            px_start = (rx1 * W, ry1 * H)
            px_end = (rx2 * W, ry2 * H)
            self.counter.update_line(px_start, px_end)
        else:
            px_start = (0.2 * W, 0.5 * H)
            px_end = (0.8 * W, 0.5 * H)
            self.counter.update_line(px_start, px_end)

        # 4. Count crossings
        new_events = self.counter.update(tracks, self.frame_index)
        for ev in new_events:
            if self.on_event:
                self.on_event({
                    "type": "crossing",
                    "track_id": ev.track_id,
                    "direction": ev.direction,
                    "frame": ev.frame_index,
                    "timestamp": ev.timestamp,
                    "count_in": self.counter.count_in,
                    "count_out": self.counter.count_out,
                    "total": self.counter.total,
                })

        # 5. Render
        frame = draw_detections(frame, tracks)
        frame = draw_counting_line(
            frame, px_start, px_end,
            self.counter.count_in, self.counter.count_out,
        )
        frame = draw_hud(
            frame,
            self.counter.count_in,
            self.counter.count_out,
            len(tracks),
            self.fps_measured,
            self.frame_index,
        )
        return frame

    # ------------------------------------------------------------------
    def get_status(self) -> Dict:
        return {
            "state": self.state,
            "frame": self.frame_index,
            "fps": round(self.fps_measured, 1),
            "counts": self.counter.get_summary() if self.counter else {},
            "video_info": self.video_info,
        }