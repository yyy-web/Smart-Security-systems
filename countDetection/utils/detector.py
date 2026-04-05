"""
YOLOv8s ONNX detector with pure OpenCV preprocessing.

Pipeline:
  1. Letterbox resize to 640×640 (preserves aspect ratio, no distortion).
  2. BGR→RGB, HWC→CHW, uint8→float32 /255.
  3. ONNX Runtime inference.
  4. Decode raw output (8400 anchors × [cx,cy,w,h,cls...]).
  5. NMS with configurable IoU and confidence thresholds.
  6. Scale boxes back to original image coords.
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# YOLOv8 COCO class index for "person"
PERSON_CLASS_ID = 0


class Letterbox:
    """
    Letterbox resize: fit image into target size keeping aspect ratio,
    pad remainder with gray (114, 114, 114).
    """

    def __init__(self, target_size: int = 640, pad_color: int = 114):
        self.target = target_size
        self.pad_color = pad_color

    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Returns:
            padded_img  : (target, target, 3) uint8
            scale       : float, scale factor applied to both dims
            padding     : (pad_left, pad_top) in pixels
        """
        h, w = img.shape[:2]
        scale = min(self.target / h, self.target / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_top = (self.target - new_h) // 2
        pad_left = (self.target - new_w) // 2
        pad_bottom = self.target - new_h - pad_top
        pad_right = self.target - new_w - pad_left

        padded = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=(self.pad_color, self.pad_color, self.pad_color),
        )
        return padded, scale, (pad_left, pad_top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Pure-NumPy NMS. Returns kept indices sorted by score descending."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou_vals = inter / (union + 1e-6)
        order = order[1:][iou_vals <= iou_threshold]
    return keep


class YOLOv8Detector:
    """
    YOLOv8s ONNX detector, person class only.

    Parameters
    ----------
    model_path       : path to .onnx weight file
    conf_threshold   : minimum detection confidence
    iou_threshold    : NMS IoU threshold
    input_size       : model input resolution (default 640)
    use_gpu          : try CUDA execution provider if True
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        use_gpu: bool = False,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.letterbox = Letterbox(input_size)

        providers = ["CPUExecutionProvider"]
        if use_gpu:
            providers = ["CUDAExecutionProvider"] + providers

        logger.info(f"Loading ONNX model: {model_path}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info(
            f"Model loaded. Input: {self.input_name}, "
            f"Providers: {self.session.get_providers()}"
        )

    # ------------------------------------------------------------------
    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """BGR frame → (1,3,H,W) float32 blob + metadata for unscaling."""
        padded, scale, padding = self.letterbox(frame)
        # BGR → RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        # HWC → CHW, normalize
        blob = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)  # (1,3,H,W)
        return blob, scale, padding

    def postprocess(
        self,
        output: np.ndarray,           # (1, num_classes+4, 8400)
        scale: float,
        padding: Tuple[int, int],
        orig_shape: Tuple[int, int],  # (H, W)
    ) -> np.ndarray:
        """
        Decode YOLOv8 raw output to [x1,y1,x2,y2,conf] for persons.
        YOLOv8 output layout: (1, 4+num_cls, num_anchors) — transposed from v5.
        """
        out = output[0]  # (4+num_cls, 8400)
        # Transpose to (8400, 4+num_cls)
        out = out.T  # (8400, 4+num_cls)

        # Box coords (cx, cy, w, h) and class scores
        boxes_cxcywh = out[:, :4]
        class_scores = out[:, 4:]  # (8400, num_cls)

        # Filter person class only
        person_scores = class_scores[:, PERSON_CLASS_ID]
        mask = person_scores >= self.conf_threshold

        if mask.sum() == 0:
            return np.empty((0, 5), dtype=np.float32)

        boxes_cxcywh = boxes_cxcywh[mask]
        scores = person_scores[mask]

        # cx,cy,w,h → x1,y1,x2,y2  (still in letterboxed coords)
        x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

        # Unscale: remove padding, undo resize
        pad_left, pad_top = padding
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale

        # Clip to image bounds
        H, W = orig_shape
        x1 = np.clip(x1, 0, W)
        x2 = np.clip(x2, 0, W)
        y1 = np.clip(y1, 0, H)
        y2 = np.clip(y2, 0, H)

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = nms(boxes, scores, self.iou_threshold)

        result = np.zeros((len(keep), 5), dtype=np.float32)
        for i, k in enumerate(keep):
            result[i] = [x1[k], y1[k], x2[k], y2[k], scores[k]]
        return result

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Full detection pipeline on a single BGR frame.

        Returns
        -------
        detections : (N, 5) float32 — [x1, y1, x2, y2, confidence]
        """
        orig_h, orig_w = frame.shape[:2]
        blob, scale, padding = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: blob})
        dets = self.postprocess(outputs[0], scale, padding, (orig_h, orig_w))
        return dets