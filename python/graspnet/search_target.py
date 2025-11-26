import os
import sys
import time
import numpy as np
import cv2
import pyrealsense2 as rs

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import local modules
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from arx5_interface import Arx5CartesianController, EEFState
from target_position_estimation import get_target_position
import grasp_process as gp

def init_arm():
    return Arx5CartesianController("X5", "can0")

def move_arm(controller, target_pose, duration=3.0):
    cfg = controller.get_controller_config()
    now = controller.get_timestamp() + cfg.default_preview_time
    eef_state = controller.get_eef_state()
    
    cmd_start = EEFState()
    cmd_start.pose_6d()[:] = eef_state.pose_6d()
    cmd_start.gripper_pos = eef_state.gripper_pos
    cmd_start.timestamp = now
    
    cmd_end = EEFState()
    cmd_end.pose_6d()[:] = target_pose
    cmd_end.gripper_pos = eef_state.gripper_pos
    cmd_end.timestamp = now + duration
    
    controller.set_eef_traj([cmd_start, cmd_end])
    time.sleep(duration + 0.5)

def init_yolo():
    weights = os.path.join(CUR_DIR, 'yolo11', 'best_atec.pt')
    if os.path.exists(weights):
        return gp.YOLO(weights)
    print(f"YOLO weights not found at {weights}")
    return None

def init_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    align = rs.align(rs.stream.color)
    pipeline.start(config)
    return pipeline, align

def main():
    # 1. Initialize Resources
    controller = init_arm()
    controller.reset_to_home()
    yolo_model = init_yolo()
    pipeline, align = init_camera()
    
    here = left = right = opposite = None

    if not yolo_model:
        return here, left, right, opposite

    # 2. Define Search Positions
    # Pos 0: User defined
    pos0 = np.array([0.1602, 0.001, 0.2645, -0., 0.62, 0.])
    # Pos 1: Placeholder (Look Left)
    pos1 = np.array([0.1602, 0.15, 0.2645, -0., 0.62, 0.3]) 
    # Pos 2: Placeholder (Look Right)
    pos2 = np.array([0.1602, -0.15, 0.2645, -0., 0.62, -0.3])
    # Pos 3: Placeholder (Look Forward/Up)
    pos3 = np.array([0.25, 0.0, 0.4, 0.0, 0.2, 0.0]) 

    positions = [pos0, pos1, pos2, pos3]
    found_idx = -1
    target_pos = None
    
    try:
        # 3. Search Loop
        for i, pos in enumerate(positions):
            print(f"\n[Info] Moving to Search Position {i}...")
            move_arm(controller, pos)
            
            # Get current state for estimation
            current_state = controller.get_eef_state()
            
            print(f"[Info] Scanning at Position {i}...")
            target_pos = get_target_position(pipeline, align, yolo_model, current_state)
            
            if target_pos is not None:
                found_idx = i
                break
        
        # 4. Logic Check
        if found_idx == -1:
            print("\n[Result] Target not detected in any position.")
        else:
            print(f"\n[Result] Target detected at Position {found_idx}")
            print(f"Target Position (Base Frame): {target_pos}")

            # Case: Found at Position 3
            if found_idx == 3:
                opposite = True
                print(">>> 目标物体在对面，需要移动到对面去。")
            
            # Case: Found at Position 0, 1, 2
            else:
                dist = np.linalg.norm(target_pos) # Distance from base origin
                print(f"Distance to target: {dist:.3f}m")
                y = target_pos[1]

                if found_idx == 1:
                    left = True
                elif found_idx == 2:
                    right = True
                elif found_idx == 0:
                    if dist < 0.70 and 0 < target_pos[0] < 0.65 and -0.5 < target_pos[1] < 0.5 and 0 < target_pos[2] < 0.3:
                        here = True
                        print(">>> 目标物体在当前位置附近，无需移动。")
                        frames = pipeline.wait_for_frames()
                        aligned_frames = align.process(frames)
                        color_frame = aligned_frames.get_color_frame()
                        if color_frame:
                            img = np.asanyarray(color_frame.get_data())
                            res = yolo_model.predict(img, conf=0.4, verbose=False)
                            if res:
                                ts = int(time.time())
                                vis_path = os.path.join(CUR_DIR, "doc", f"search_success_{ts}.jpg")
                                os.makedirs(os.path.dirname(vis_path), exist_ok=True)
                                cv2.imwrite(vis_path, res[0].plot())
                                print(f" - 检测结果图已保存: {vis_path}")
                    else:
                        if y > 0:
                            left = True
                        else:
                            right = True
            if left:
                print(">>> 目标物体在左侧，需要向左移动。")
            if right:
                print(">>> 目标物体在右侧，需要向右移动。")

  

    finally:
        pipeline.stop()
        print("[Info] Returning to home position...")
        controller.reset_to_home()
        time.sleep(3.5)
    
    return here, left, right, opposite

if __name__ == "__main__":
    main()
