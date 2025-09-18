# Yolov8Demo
This is a demo I tried with yolov8. I used the official dataset to complete the recognition of the computer screen and mouse tracking.

这个项目是一个基于YOLOv8的屏幕目标检测工具。

主要功能包括：

1. 1.
   屏幕捕获 : 使用 mss 库捕获屏幕的指定区域。
2. 2.
   YOLOv8目标检测 : 加载YOLOv8模型（ yolov8n.pt ），在捕获的屏幕帧上进行目标检测。
3. 3.
   目标选择与追踪 : 能够选择主要目标（例如，距离屏幕中心最近的目标）。
4. 4.
   鼠标自动移动 : 将鼠标平滑移动到检测到的主要目标位置。
5. 5.
   实时显示 : 可选地在OpenCV窗口中实时显示检测结果，包括边界框、置信度、类别标签和FPS性能信息。
6. 6.
   热键控制 :
   - F2 : 切换检测的开启/关闭状态。
   - F3 : 切换检测框的显示/隐藏。
   - F4 : 重新校准屏幕检测区域。
   - ESC : 退出程序。
7. 7.
   可配置性 : 通过 DEFAULT_CONFIG 字典配置检测设备（CPU/GPU）、模型路径、目标类别、置信度阈值、推理图像大小、检测区域、鼠标移动速度、最大FPS、是否显示窗口和性能信息，以及检测框颜色。
项目结构：

- `main.py` : 包含 YOLOScreenDetector 类，实现上述所有核心功能。
- `config.py` : 目前为空，但 `main.py` 中定义了 DEFAULT_CONFIG ，推测 `config.py` 可能用于存放用户自定义配置。
- models/ : 可能用于存放YOLO模型文件，目前有一个 yolov8n.pt 。
- data/custom_dataset/images/ 和 data/custom_dataset/lables/ : 用于存放自定义数据集的图像和标签，表明该项目可能支持自定义模型的训练或使用。
- `requirements.txt` : 目前为空，但根据 `main.py` 中的导入，项目依赖 opencv-python , numpy , pyautogui , keyboard , mss , ultralytics , torch 等库。
- utils/ : 可能用于存放一些辅助工具函数。
总结来说，这是一个用于实时屏幕目标检测和自动鼠标追踪的Python应用，主要用于自动化或辅助用户在屏幕上与特定目标进行交互。
