# 人员滞留检测系统 | Loitering Detection System

基于 YOLOv8 + IoU Tracker 的实时人员滞留检测系统，支持多区域配置、透视补偿和实时告警。

## 功能特性

- **YOLOv8 目标检测**: 使用 ONNX Runtime 推理，支持 GPU/CPU
- **轻量级 IoU 跟踪**: 基于 IoU 匹配的多目标跟踪，支持短暂丢失容忍
- **多区域滞留检测**: 支持自定义多个多边形警戒区域
- **透视补偿**: 多种锚点模式 + Y 轴偏移 + 可选 BEV 变换
- **实时告警**: 滞留超时触发告警，支持实时预览和完整事件通知
- **SSE 流式推帧**: 基于 Server-Sent Events 的实时视频流
- **数据导出**: 支持 CSV/JSON 格式导出滞留日志

## 系统架构

```
loiteringDetection/
├── app/
│   ├── main.py          # FastAPI 服务入口
│   └── core/
│       ├── inference.py # YOLOv8 ONNX 推理引擎
│       ├── tracker.py   # 跟踪器 + 滞留检测核心算法
│       └── processor.py # 视频处理流水线
├── static/
│   └── index.html       # 前端界面
└── uploads/             # 上传文件和截图存储
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
cd /home/bigmodel/ultralytics/loiteringDetection
python app/main.py
```

服务将在 `http://127.0.0.1:8082` 启动。

### 3. 使用界面

1. **上传视频**: 导入视频文件 (支持 mp4, avi, flv, mov, mkv, ts, wmv)
2. **上传模型**: 上传 YOLOv8s.onnx 模型文件
3. **绘制区域**: 点击"新建区域"，在视频画布上依次点击定点，双击或点击"完成区域"结束
4. **配置参数**: 调整置信度阈值、滞留时间、锚点模式等参数
5. **开始检测**: 点击"开始检测"按钮
6. **查看结果**: 实时查看滞留人员、告警日志和数据表格

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | Web 界面 |
| `/api/upload/video` | POST | 上传视频文件 |
| `/api/upload/model` | POST | 上传 ONNX 模型 |
| `/api/zones` | POST | 设置检测区域 |
| `/api/params` | POST | 更新检测参数 |
| `/api/detect/start` | POST | 开始检测 |
| `/api/detect/stop` | POST | 停止检测 |
| `/api/detect/pause` | POST | 暂停/继续 |
| `/api/stream` | GET | SSE 实时推流 |
| `/api/export/log` | GET | 导出日志 (CSV/JSON) |
| `/api/screenshot` | POST | 保存截图 |
| `/api/status` | GET | 获取系统状态 |

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| conf_threshold | 检测置信度阈值 | 0.40 |
| min_area | 最小目标面积 (像素²) | 500 |
| linger_threshold | 滞留判定时间 (秒) | 30 |
| tolerance_frames | 短暂丢失容忍帧数 | 30 |
| speed_threshold | 速度阈值 (像素/秒，0=关闭) | 50 |
| anchor_mode | 锚点模式 | foot |
| zone_y_offset | 区域 Y 轴偏移 (像素) | 0 |

## 锚点模式

系统支持多种锚点模式以适应不同的摄像头视角:

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `foot` (推荐) | 边界框底部中心 | 大多数监控场景 |
| `foot_inner` | 底部向上 10% 处 | 脚部可能被遮挡时 |
| `center` | 边界框中心 | 水平视角摄像头 |
| `head` | 边界框顶部中心 | 头顶区域检测 |

## 透视补偿

针对俯仰角较大的摄像头，系统提供以下补偿方案:

1. **锚点模式**: 使用脚部锚点代替中心点，减少透视误差
2. **Y 轴偏移**: 统一向下平移所有区域，补偿绘制误差
3. **水平辅助线**: 显示锚点 Y 坐标的水平线，辅助区域绘制
4. **BEV 变换**: 可选的鸟瞰变换 (需要手动标定)

## 滞留判定逻辑

1. 目标底边中心点 (锚点) 进入区域
2. 持续计算滞留时长 = 当前时间 - 进入时间
3. 滞留时长 >= `linger_threshold` 时标记为滞留
4. 目标离开区域后，记录完整事件 (进入时间、离开时间、最终时长)
5. 速度超过 `speed_threshold` 的目标不计入滞留

## 可视化说明

- **绿色框**: 正常移动目标
- **红色框**: 滞留目标
- **绿色圆点**: 锚点 (判断点)
- **绿色虚线**: 锚点水平辅助线
- **彩色多边形**: 检测区域 (半透明填充)
- **轨迹线**: 目标移动路径

## 注意事项

- 模型必须是 ONNX 格式
- 绘制区域时，建议将区域下边界对齐锚点水平辅助线
- 如果区域整体偏高，可使用 Y 轴偏移参数补偿
- 推荐视频分辨率不超过 1920x1080

## 许可证
yyy
MIT License
