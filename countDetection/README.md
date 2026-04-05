# 客流计数系统 | Crowd Flow Counter

基于 YOLOv8 + ByteTrack 的实时人流计数系统，支持视频上传、在线检测和双向客流统计。

## 功能特性

- **YOLOv8 目标检测**: 使用 ONNX 模型进行高效推理，仅检测行人 (person class)
- **ByteTrack 多目标跟踪**: Kalman 滤波 + IoU 匹配，稳定的轨迹跟踪
- **虚拟线计数**: 可自定义计数线，支持进出双向统计
- **实时可视化**: 轨迹追踪、边界框、计数线、HUD 信息面板
- **Web 界面**: 基于 FastAPI + WebSocket 的实时流式传输
- **拖拽上传**: 支持视频和模型文件拖拽上传

## 系统架构

```
app/
├── main.py          # FastAPI 服务，WebSocket 流式传输
└── pipeline.py      # 视频处理流水线 (detector → tracker → counter → visualizer)

utils/
├── detector.py      # YOLOv8 ONNX 检测器 (Letterbox + NMS)
├── tracker.py       # ByteTrack 跟踪器 (Kalman Filter + IoU matching)
├── counter.py       # 虚拟线计数器 (cross-product side test)
└── visualizer.py    # 可视化绘制 (轨迹、计数线、HUD)

static/
└── index.html       # 前端界面 (Sci-Fi 风格)
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备模型

需要 YOLOv8s ONNX 模型文件。可以使用 ultralytics 导出:

```bash
pip install ultralytics
yolo export model=yolov8s.pt format=onnx imgsz=640
```

### 2. 启动服务

```bash
cd /home/bigmodel/ultralytics/countDetection
python app/main.py
```

服务将在 `http://127.0.0.1:8014` 启动。

### 3. 使用界面

1. **上传视频**: 点击或拖拽上传视频文件 (支持 mp4, avi, flv, mov, mkv, wmv, ts, m4v)
2. **上传模型**: 上传 YOLOv8s.onnx 模型文件
3. **配置计数线**: 调整起点和终点坐标 (相对坐标 0-1)
4. **开始检测**: 点击"开始检测"按钮
5. **查看结果**: 实时查看客流统计和事件日志

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | Web 界面 |
| `/api/upload/video` | POST | 上传视频文件 |
| `/api/upload/model` | POST | 上传 ONNX 模型 |
| `/api/pipeline/start` | POST | 开始检测 |
| `/api/pipeline/stop` | POST | 停止检测 |
| `/api/pipeline/pause` | POST | 暂停检测 |
| `/api/pipeline/resume` | POST | 继续检测 |
| `/api/pipeline/line` | POST | 更新计数线 |
| `/api/pipeline/status` | GET | 获取当前状态 |
| `/ws/stream` | WS | 实时帧和事件流 |

## 算法说明

### 计数原理

1. 定义虚拟计数线 (起点 → 终点)
2. 对每个确认的跟踪目标，计算其中心点在计数线哪一侧 (使用叉积)
3. 当目标从一侧穿越到另一侧时，触发计数
4. 方向判定:
   - 从左到右 (左侧 → 右侧): OUT
   - 从右到左 (右侧 → 左侧): IN

### 参数说明

- **min_confirm_frames**: 目标需确认的最小帧数 (默认 3)
- **conf_threshold**: 检测置信度阈值 (默认 0.35)
- **iou_threshold**: NMS IoU 阈值 (默认 0.45)

## 视频演示

系统会实时显示:
- 带颜色的边界框 (每个 track 有唯一颜色)
- 轨迹尾迹
- 计数线 (带发光效果)
- 进出方向箭头
- HUD 信息面板 (IN/OUT/TOTAL/FPS/FRAME)

## 注意事项

- 模型必须是 ONNX 格式
- 视频格式需为常见容器格式
- GPU 加速需要安装 `onnxruntime-gpu`
- 推荐视频分辨率不超过 1920x1080

## 许可证
yyy
MIT License
