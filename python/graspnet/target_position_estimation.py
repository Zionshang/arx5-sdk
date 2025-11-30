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

#手眼标定外参
handeye_rotation = [[-0.02489131, -0.16662419 , 0.98570624],
 [-0.99968  ,   0.00859452, -0.02379136],
 [-0.00450745, -0.98598302, -0.1667848 ]]
handeye_translation = [-0.09760795,0.02448454,0.0883561]
def get_target_position(pipeline=None, align=None, yolo_model=None,current_state=None, target_class=None):
    # 相机预热
    for _ in range(10):
        pipeline.wait_for_frames()
    
    target_pos_base = None 
    detected_class = None

    best_box = None
    color_img = depth_img = None
    for attempt in range(20):
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            print("Error: Could not capture frames.")
            return

        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())

        print(f"Running YOLO detection (attempt {attempt + 1}/20)...")
        results = yolo_model.predict(color_img, conf=0.4, iou=0.7, verbose=False)
        vis_img = None
        if results:
            try:
                vis_img = results[0].plot()
            except Exception:
                vis_img = None
        if vis_img is not None:
            ts_label = time.strftime("%Y%m%d_%H%M%S")
            vis_dir = os.path.join(CUR_DIR, "doc", "obs_yolo")
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(vis_dir, f"yolo_detect_{ts_label}.jpg")
            cv2.imwrite(vis_path, vis_img)
            print(f"YOLO detection visualization saved to {vis_path}")

        if results and results[0].boxes:
            boxes = results[0].boxes
            if target_class:
                boxes = [b for b in boxes if yolo_model.names[int(b.cls)] == target_class]
            
            if boxes:
                temp_box = max(boxes, key=lambda b: b.conf)
                x1, y1, x2, y2 = temp_box.xyxy[0].cpu().numpy()
                u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                if depth_img[v, u] > 0:
                    best_box = temp_box
                    detected_class = yolo_model.names[int(best_box.cls)]
                    break

    if best_box is not None:
        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
        
        # Calculate center
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)
        
        # Get depth at center (handle 0 depth by searching neighborhood if needed, but keeping simple)
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
            curr_pose = current_state.pose_6d() # [x, y, z, rx, ry, rz]
            
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
        print("No target detected after 20 attempts.")

    return target_pos_base, detected_class

if __name__ == "__main__":
    pos, cls = get_target_position()
    print(f"Final Result: {pos}, Class: {cls}")
