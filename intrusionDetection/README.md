# 区域入侵检测系统 | Intrusion Detection System

基于 YOLOv8 + ByteTrack 的实时区域入侵检测系统，支持多边形区域绘制、入侵事件抓拍和实时告警。

## 功能特性

- **YOLOv8 目标检测**: 使用 OpenCV DNN 后端加载 ONNX 模型，高效推理
- **ByteTrack 多目标跟踪**: Kalman 滤波 + IoU 匹配，稳定的轨迹跟踪
- **自定义警戒区域**: 支持多边形区域绘制，可配置多个检测区域
- **入侵事件判定**: 基于连续帧判断，避免误报
- **实时告警**: 入侵时触发视觉告警，显示入侵目标信息
- **抓拍记录**: 自动抓拍入侵目标图像，支持查看和导出
- **数据导出**: 支持 CSV 格式导出入侵日志

## 系统架构

```
intrusionDetection/
├── app/
│   └── main.py          # FastAPI 服务，核心检测逻辑
├── static/
│   └── index.html       # 前端界面 (Sci-Fi 风格)
├── models/              # ONNX 模型存储目录
├── captures/            # 画面抓拍存储目录
└── logs/                # 日志导出目录
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
cd /home/bigmodel/ultralytics/intrusionDetection
python app/main.py
```

服务将在 `http://127.0.0.1:8010` 启动。

### 3. 使用界面

1. **上传视频**: 选择视频文件 (支持 mp4, avi, flv, mkv, mov, wmv, ts, m4v, rmvb, 3gp)
2. **上传模型**: 上传 YOLOv8s.onnx 模型文件
3. **绘制区域**: 点击"绘制区域"，在视频画布上依次点击定点，完成后点击"完成"
4. **配置参数**: 调整置信度阈值、最小面积、入侵帧数等参数
5. **开始检测**: 点击"开始检测"按钮
6. **查看结果**: 实时查看入侵事件、抓拍图像和日志

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | Web 界面 |
| `/api/upload/video` | POST | 上传视频文件 |
| `/api/upload/model` | POST | 上传 ONNX 模型 |
| `/api/config` | POST | 更新检测配置 |
| `/api/config` | GET | 获取当前配置 |
| `/api/detect/start` | POST | 开始检测 |
| `/api/detect/stop` | POST | 停止检测 |
| `/api/events` | GET | 获取入侵事件列表 |
| `/api/events/export` | GET | 导出 CSV 日志 |
| `/api/snapshot/{event_id}` | GET | 获取入侵快照 |
| `/api/capture` | POST | 保存当前帧 |
| `/api/status` | GET | 获取系统状态 |
| `/ws` | WS | 实时帧和事件流 |

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| confidence_threshold | 检测置信度阈值 | 0.45 |
| min_area | 最小目标面积 (像素²) | 1000 |
| intrusion_frames | 连续帧判定阈值 | 3 |
| iou_threshold | NMS IoU 阈值 | 0.45 |

## 入侵判定逻辑

1. 使用射线法判断目标底边中心点是否在多边形区域内
2. 目标需在区域内连续停留 `intrusion_frames` 帧才判定为入侵
3. 同一目标在同一区域只触发一次入侵事件
4. 目标离开区域后重置帧计数

## 可视化说明

- **红色框**: 入侵目标
- **绿色框**: 正常目标
- **轨迹线**: 目标移动轨迹
- **中心点**: 目标底边中心
- **区域填充**: 半透明警戒区域

## 注意事项

- 模型必须是 ONNX 格式
- 视频格式需为常见容器格式
- 绘制区域时建议至少 3 个点
- 入侵帧数设置过小可能导致误报

## 许可证
yyy
MIT License
