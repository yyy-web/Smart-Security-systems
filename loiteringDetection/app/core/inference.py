"""
YOLOv8 ONNX 推理引擎
使用 onnxruntime + OpenCV 实现高效前处理/后处理
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

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
    "toothbrush"
]


class YOLOv8ONNXInference:
    """
    YOLOv8s ONNX 推理封装
    - letterbox 前处理（保持宽高比）
    - NMS 后处理
    - 支持任意输入尺寸模型
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.4, nms_threshold: float = 0.45):
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError("请安装 onnxruntime: pip install onnxruntime")

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # 优先使用 GPU，回退 CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape   # [1, 3, H, W]
        self.input_h = input_shape[2] if isinstance(input_shape[2], int) else 640
        self.input_w = input_shape[3] if isinstance(input_shape[3], int) else 640
        logger.info(f"ONNX 模型加载成功，输入尺寸 {self.input_h}x{self.input_w}")

    def letterbox(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        letterbox 缩放：保持宽高比，灰边填充
        返回：(处理后图像, 缩放比例, (pad_left, pad_top))
        """
        h, w = img.shape[:2]
        scale = min(self.input_h / h, self.input_w / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        pad_top = (self.input_h - new_h) // 2
        pad_left = (self.input_w - new_w) // 2

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        out = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        out[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        return out, scale, (pad_left, pad_top)

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """BGR → RGB → letterbox → NCHW float32 归一化"""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lb, scale, pad = self.letterbox(rgb)
        blob = lb.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # HWC → NCHW
        return blob, scale, pad

    def postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad: Tuple[int, int],
        orig_shape: Tuple[int, int],
        min_area: float = 0,
    ) -> List[Dict]:
        """
        output shape: [1, 84, 8400]  (YOLOv8 格式)
        返回: [{"bbox":[x1,y1,x2,y2], "confidence":float, "class_id":int, "class_name":str}]
        """
        pred = output[0]                     # (84, 8400)
        pred = pred.T                        # (8400, 84)
        boxes = pred[:, :4]                  # cx, cy, w, h
        scores = pred[:, 4:]                 # (8400, 80)
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        mask = confidences >= self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # cx,cy,w,h → x1,y1,x2,y2（letterbox 坐标系）
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        # 还原到原图坐标
        pad_left, pad_top = pad
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale

        # 裁剪到图像范围
        oh, ow = orig_shape
        x1 = np.clip(x1, 0, ow)
        y1 = np.clip(y1, 0, oh)
        x2 = np.clip(x2, 0, ow)
        y2 = np.clip(y2, 0, oh)

        # OpenCV NMS
        bboxes_cv = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            bboxes_cv, confidences.tolist(), self.conf_threshold, self.nms_threshold
        )

        results = []
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                w_ = float(x2[i] - x1[i])
                h_ = float(y2[i] - y1[i])
                area = w_ * h_
                if area < min_area:
                    continue
                cid = int(class_ids[i])
                results.append({
                    "bbox": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                    "confidence": float(confidences[i]),
                    "class_id": cid,
                    "class_name": COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else str(cid),
                })
        return results

    def infer(self, img: np.ndarray, min_area: float = 0) -> List[Dict]:
        """完整推理流程"""
        blob, scale, pad = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: blob})
        return self.postprocess(outputs[0], scale, pad, img.shape[:2], min_area)