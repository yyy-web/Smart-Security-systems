Ultralytics 智能视觉检测套件
基于 YOLOv8 + ByteTrack 的实时智能检测系统集合，涵盖人流计数、区域入侵和人员滞留三大核心场景。

📦 包含项目
项目	功能	核心算法
countDetection	客流双向计数	YOLOv8 + ByteTrack + 虚拟线交叉
intrusionDetection	区域入侵检测	YOLOv8 + ByteTrack + 多边形区域判定
loiteringDetection	人员滞留检测	YOLOv8 + IoU Tracker + 锚点透视补偿

🚀 特性
纯 ONNX 推理 - 支持 CPU/GPU，无需 PyTorch 运行时
实时 Web 界面 - FastAPI + WebSocket/SSE 流式推帧
可视化标注 - 前端 Canvas 交互式绘制检测区域
数据导出 - 支持 CSV/JSON 格式日志导出
透视补偿 - 多种锚点模式 + Y 轴偏移 + BEV 变换

🛠 技术栈
YOLOv8 (ONNX) | ByteTrack | OpenCV | FastAPI | WebSocket

每个文件内附：项目依赖和需知

运行后界面示例；
![3c34addcdef8ef3d4b5f8db27d96d2b9](https://github.com/user-attachments/assets/6a3cc989-e435-441b-baf0-36393c92a2b8)
![a26c56af772bcce48af9ab6eeef87840](https://github.com/user-attachments/assets/7622a9d2-4aa3-48c5-b74e-f9a8829adc82)
![63f660da6782ff8aa5151bb95f57aff4](https://github.com/user-attachments/assets/39b492bc-e501-4961-a183-87d28646c8f5)
