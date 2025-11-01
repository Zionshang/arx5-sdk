"""采集相机的照片和机械臂的位姿并保存成文件。
这里以intel realsense 相机为例， 其他相机数据读取可能需要对应修改。

改动：接入 ARX5 机械臂 API，实时读取末端位姿并保存。
保存格式：x,y,z,roll,pitch,yaw （角度为弧度）
按键说明：
  - h：采集一帧图像并保存对应的机械臂末端位姿
  - t：退出采集
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import os
import sys
import threading

count = 0

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 使路径相对于当前脚本目录，避免工作目录变化带来的问题
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
image_save_path = os.path.join(_THIS_DIR, "collect_data")
os.makedirs(image_save_path, exist_ok=True)

# 将根目录加入 sys.path，便于导入 arx5_interface
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# 导入键盘控制功能
import keyboard_teleop
from arx5_interface import Arx5CartesianController, EEFState, Gain, LogLevel


def data_collect(model="X5", interface="can0"):
    global count
    
    # 初始化机械臂控制器
    controller = Arx5CartesianController(model, interface)
    controller.reset_to_home()
    
    try:
        # 启动键盘控制线程
        ctrl_thread = threading.Thread(target=lambda: keyboard_teleop.start_keyboard_teleop(controller), daemon=True)
        ctrl_thread.start()
        print("已启动键盘控制，用方向键等控制机械臂移动，在图像窗口按 h 采集，按 q 退出采集")

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
                                               cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
            cv2.imshow("detection", color_image)  # 窗口显示，显示名为 Capture_Video

            k = cv2.waitKey(1) & 0xFF  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
            if k == ord('t'):
                print("收到退出指令，结束采集...")
                break
            if k == ord('h'):  # 键盘按一下h, 保存当前照片和机械臂位姿（在图像窗口内按键）
                print(f"采集第{count}组数据...")
                # 从 ARX5 获取当前末端位姿 [x,y,z,roll,pitch,yaw]，弧度
                try:
                    eef_state = controller.get_eef_state()
                    pose = eef_state.pose_6d().tolist()
                except Exception as e:
                    print(f"获取机械臂位姿失败：{e}")
                    continue

                print(f"机械臂pose: {pose}")

                poses_txt = os.path.join(image_save_path, 'poses.txt')
                with open(poses_txt, 'a+', encoding='utf-8') as f:
                    # 以逗号分隔保存一行
                    pose_ = [str(float(i)) for i in pose]
                    new_line = f"{','.join(pose_)}\n"
                    f.write(new_line)

                    img_path = os.path.join(image_save_path, f"images{count}.jpg")
                    cv2.imwrite(img_path, color_image)
                    print(f"已保存: {img_path}")
                    count += 1
    finally:
        # 资源清理
        try:
            controller.reset_to_home()
            controller.set_to_damping()
        except Exception:
            pass
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    model = sys.argv[1] if len(sys.argv) > 1 else "X5"
    interface = sys.argv[2] if len(sys.argv) > 2 else "can0"
    
    print(f"启动数据采集程序，机械臂型号: {model}, 接口: {interface}")
    print("键盘控制说明：")
    print("  - 方向键: 控制 X/Y 轴移动")
    print("  - Page Up/Down: 控制 Z 轴移动") 
    print("  - Q/A: 控制 Roll 轴旋转")
    print("  - W/S: 控制 Pitch 轴旋转")
    print("  - E/D: 控制 Yaw 轴旋转")
    print("  - R/F: 开合夹爪")
    print("  - Space: 重置到初始位置")
    print("图像窗口控制：")
    print("  - H: 采集当前图像和位姿")
    print("  - T: 退出程序")
    
    data_collect(model, interface)
