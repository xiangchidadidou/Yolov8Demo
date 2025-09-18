import cv2
import numpy as np
import pyautogui
import keyboard
import time
from mss import mss
from ultralytics import YOLO
import torch
import threading


class YOLOScreenDetector:
    def __init__(self, config):
        self.config = config

        # 设备配置
        self.device = self.config['device']
        print(f"使用设备: {self.device}")

        # 加载YOLOv8模型
        try:
            self.model = YOLO(self.config['model_path']).to(self.device)
            print(f"成功加载模型: {self.config['model_path']}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

        # 屏幕捕获设置
        self.sct = mss()

        # 设置检测区域
        self.update_monitor_region()

        # 状态控制
        self.detection_active = False
        self.show_detection = True  # 新增：是否显示检测框
        self.last_target = None
        self.running = True

        # 性能监控
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        # 创建显示窗口
        if self.config['show_window']:
            cv2.namedWindow('YOLOv8 Screen Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('YOLOv8 Screen Detection', 800, 600)

    def update_monitor_region(self):
        """更新屏幕捕获区域"""
        screen_width, screen_height = pyautogui.size()

        if self.config['detection_region'] == 'full':
            self.monitor = {
                "top": 0,
                "left": 0,
                "width": screen_width,
                "height": screen_height,
                "mon": 0
            }
        else:  # center region
            self.monitor = {
                "top": int(screen_height * self.config['region_top']),
                "left": int(screen_width * self.config['region_left']),
                "width": int(screen_width * self.config['region_width']),
                "height": int(screen_height * self.config['region_height']),
                "mon": 0
            }

        print(f"检测区域: {self.monitor}")

    def grab_screen(self):
        """捕获指定屏幕区域"""
        try:
            img = np.array(self.sct.grab(self.monitor))
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            print(f"屏幕捕获错误: {e}")
            return None

    def find_targets(self, frame):
        """使用YOLOv8检测目标"""
        try:
            results = self.model.predict(
                source=frame,
                conf=self.config['confidence_threshold'],
                classes=self.config['target_classes'],
                device=self.device,
                verbose=False,
                imgsz=self.config['inference_size']
            )

            return results[0]
        except Exception as e:
            print(f"目标检测错误: {e}")
            return None

    def draw_detections(self, frame, results):
        """在帧上绘制检测结果"""
        if results is None or results.boxes is None:
            return frame

        # 复制帧以避免修改原始图像
        display_frame = frame.copy()

        # 获取类别名称
        class_names = self.model.names

        for box in results.boxes:
            # 获取边界框坐标和置信度
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())

            # 绘制边界框
            color = self.config['box_colors'][cls_id % len(self.config['box_colors'])]
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # 绘制标签背景
            label = f"{class_names[cls_id]}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display_frame, (int(x1), int(y1) - label_size[1] - 5),
                          (int(x1) + label_size[0] + 5, int(y1)), color, -1)

            # 绘制标签文本
            cv2.putText(display_frame, label, (int(x1) + 2, int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 绘制性能信息
        if self.config['show_performance']:
            cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Detections: {len(results.boxes)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return display_frame

    def select_primary_target(self, boxes):
        """选择主要目标"""
        if boxes is None or boxes.boxes is None or len(boxes.boxes) == 0:
            return None

        frame_center_x = self.monitor["width"] / 2
        frame_center_y = self.monitor["height"] / 2

        min_distance = float('inf')
        primary_target = None

        for box in boxes.boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # 计算与屏幕中心的距离
            distance = np.sqrt((center_x - frame_center_x) ** 2 + (center_y - frame_center_y) ** 2)

            if distance < min_distance:
                min_distance = distance
                primary_target = (center_x, center_y)

        return primary_target

    def move_to_target(self, target):
        """将鼠标移动到目标位置"""
        if target is None:
            return

        try:
            # 将相对坐标转换为绝对屏幕坐标
            abs_x = self.monitor["left"] + target[0]
            abs_y = self.monitor["top"] + target[1]

            # 平滑移动鼠标
            pyautogui.moveTo(abs_x, abs_y, duration=self.config['mouse_move_duration'])

        except pyautogui.FailSafeException:
            print("鼠标移动到屏幕边缘，触发安全保护")
        except Exception as e:
            print(f"鼠标移动错误: {e}")

    def toggle_detection(self):
        """切换检测状态"""
        self.detection_active = not self.detection_active
        status = "开启" if self.detection_active else "关闭"
        print(f"检测{status}")

    def toggle_display(self):
        """切换显示状态"""
        self.show_detection = not self.show_detection
        status = "开启" if self.show_detection else "关闭"
        print(f"显示检测框{status}")

    def print_status(self):
        """打印状态信息"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if elapsed_time > 1:  # 每秒更新一次FPS
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = current_time

            status_text = (f"FPS: {self.fps:.1f} | "
                           f"检测: {'开' if self.detection_active else '关'} | "
                           f"显示: {'开' if self.show_detection else '关'} | "
                           f"F2:切换检测 F3:显示框 ESC:退出")
            print(f"\r{status_text}", end='')

    def run(self):
        """主循环"""
        print("=" * 60)
        print("YOLOv8屏幕检测脚本启动!")
        print("热键控制:")
        print("  F2  - 切换检测状态")
        print("  F3  - 切换显示检测框")
        print("  F4  - 重新校准屏幕区域")
        print("  ESC - 退出程序")
        print("=" * 60)

        # 注册热键
        keyboard.add_hotkey('f2', self.toggle_detection)
        keyboard.add_hotkey('f3', self.toggle_display)
        keyboard.add_hotkey('f4', self.update_monitor_region)

        try:
            while self.running:
                # 捕获屏幕
                frame = self.grab_screen()
                if frame is None:
                    time.sleep(0.1)
                    continue

                if self.detection_active:
                    # 检测目标
                    results = self.find_targets(frame)

                    # 选择主要目标
                    target = self.select_primary_target(results)

                    # 移动鼠标到目标
                    if target:
                        self.move_to_target(target)
                        self.last_target = target

                    # 显示检测结果
                    if self.show_detection and results is not None:
                        frame = self.draw_detections(frame, results)

                # 显示窗口
                if self.config['show_window']:
                    cv2.imshow('YOLOv8 Screen Detection', frame)

                    # 检查窗口是否被用户关闭
                    if cv2.getWindowProperty('YOLOv8 Screen Detection', cv2.WND_PROP_VISIBLE) < 1:
                        print("\n检测窗口被关闭，退出程序")
                        break

                # 更新性能计数
                self.frame_count += 1
                self.print_status()

                # 处理OpenCV事件
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键
                    break

                # 短暂休眠
                time.sleep(1 / self.config['max_fps'])

        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"\n程序运行错误: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        keyboard.unhook_all()
        if self.config['show_window']:
            cv2.destroyAllWindows()
        self.running = False
        print("\n脚本已安全退出")


# 增强的配置文件
DEFAULT_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': 'yolov8n.pt',
    'target_classes': [0],  # COCO数据集类别: 0=person
    'confidence_threshold': 0.5,
    'inference_size': 320,
    'detection_region': 'center',
    'region_top': 0.25,
    'region_left': 0.25,
    'region_width': 0.5,
    'region_height': 0.5,
    'mouse_move_duration': 0.1,
    'max_fps': 30,
    'show_window': True,  # 新增：是否显示窗口
    'show_performance': True,  # 新增：是否显示性能信息
    'box_colors': [  # 新增：检测框颜色
        (0, 255, 0),  # 绿色 - 类别0
        (255, 0, 0),  # 蓝色 - 类别1
        (0, 0, 255),  # 红色 - 类别2
        (255, 255, 0),  # 青色 - 类别3
        (255, 0, 255),  # 粉色 - 类别4
        (0, 255, 255),  # 黄色 - 类别5
    ]
}

if __name__ == "__main__":
    # 初始化配置
    config = DEFAULT_CONFIG.copy()

    # 创建检测器实例
    detector = YOLOScreenDetector(config)

    # 运行主循环
    detector.run()