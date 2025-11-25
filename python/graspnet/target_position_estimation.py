import os
import sys
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import local modules
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

import grasp_process as gp
from grasp_keycontrol import init_arm_controller, init_realsense, init_yolo, handeye_rotation, handeye_translation, build_eef_cmd

def get_target_position():
    # 1. Initialize Arm and Move to Observation Pose
    controller = init_arm_controller()
    # Target observation pose
    obs_pose = np.array([0.1602, 0.001, 0.2645, -0., 0.62, 0.], dtype=float)
    
    print(f"Moving to observation pose: {obs_pose}")
    # Get current state to plan trajectory
    cfg = controller.get_controller_config()
    curr_state = controller.get_eef_state()
    curr_pose = curr_state.pose_6d()
    curr_grip = curr_state.gripper_pos
    now = controller.get_timestamp() + cfg.default_preview_time
    
    # Move to pose (give it 3 seconds)
    controller.set_eef_traj([
        build_eef_cmd(curr_pose, curr_grip, now),
        build_eef_cmd(obs_pose, curr_grip, now + 3.0)
    ])
    time.sleep(3.5) # Wait for move to complete

    # 2. Initialize Sensors and Model
    pipeline, align = init_realsense()
    yolo_model, yolo_params = init_yolo(gp.ROOT_DIR)
    
    if yolo_model is None:
        print("Error: YOLO model not found.")
        return

    # 3. Capture Frame
    # Warmup/Wait for frames
    for _ in range(10):
        pipeline.wait_for_frames()
    
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    
    if not color_frame or not depth_frame:
        print("Error: Could not capture frames.")
        return

    color_img = np.asanyarray(color_frame.get_data())
    depth_img = np.asanyarray(depth_frame.get_data())

    # 4. YOLO Detection
    print("Running YOLO detection...")
    results = yolo_model.predict(color_img, conf=0.4, iou=0.7, verbose=False)
    
    target_pos_base = np.array([10.0, 10.0, 10.0]) # Default large value

    if results and results[0].boxes and len(results[0].boxes) > 0:
        # Get best detection (highest confidence)
        best_box = results[0].boxes[0]
        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
        
        # Calculate center
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)
        
        # Get depth at center (handle 0 depth by searching neighborhood if needed, but keeping simple)
        d = depth_img[v, u] * 0.001 # Scale to meters (assuming 1mm unit, check scale)
        # Note: grasp_keycontrol uses factor_depth from CameraInfo, usually 1/scale. 
        # D435i scale is usually 0.001. L515 might be different.
        # grasp_keycontrol uses: factor_depth = 999.999... which implies depth_img / factor_depth ~ meters
        # Let's use the value from grasp_keycontrol logic
        depth_scale = 1.0 / 1000.0 # Standard Realsense
        # Or better, check how grasp_process does it. 
        # grasp_keycontrol: factor_depth = 999.999952502551
        # So d_meters = d_raw / factor_depth
        d_meters = depth_img[v, u] / 999.999952502551
        
        if d_meters > 0:
            print(f"Target detected at pixel ({u}, {v}), depth: {d_meters:.4f}m")
            
            # 5. Coordinate Conversion
            # Camera Intrinsics (D435i from grasp_keycontrol)
            fx, fy = 606.44, 606.48
            cx, cy = 322.35, 239.54
            
            # Pixel to Camera
            z_c = d_meters
            x_c = (u - cx) * z_c / fx
            y_c = (v - cy) * z_c / fy
            p_cam = np.array([x_c, y_c, z_c, 1.0])
            
            # Camera to End-Effector
            T_cam2ee = np.eye(4)
            T_cam2ee[:3, :3] = np.array(handeye_rotation)
            T_cam2ee[:3, 3] = np.array(handeye_translation)
            
            # End-Effector to Base
            # Get current actual pose
            curr_state = controller.get_eef_state()
            curr_pose = curr_state.pose_6d() # [x, y, z, rx, ry, rz]
            
            R_ee2base = R.from_euler('xyz', curr_pose[3:], degrees=False).as_matrix()
            T_ee2base = np.eye(4)
            T_ee2base[:3, :3] = R_ee2base
            T_ee2base[:3, 3] = curr_pose[:3]
            
            # Chain transform: Base <- EE <- Cam <- Point
            p_base = T_ee2base @ (T_cam2ee @ p_cam)
            target_pos_base = p_base[:3]
            
            print(f"Target Position in Base Frame: {target_pos_base}")
        else:
            print("Warning: Invalid depth at center pixel.")
    else:
        print("No target detected.")

    # Cleanup
    pipeline.stop()
    return target_pos_base

if __name__ == "__main__":
    pos = get_target_position()
    print(f"Final Result: {pos}")
