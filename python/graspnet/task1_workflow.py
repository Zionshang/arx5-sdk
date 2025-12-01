import os
import sys
import time
import numpy as np
import keyboard
from arx5_interface import Arx5CartesianController, EEFState

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import grasp.py functionality
import grasp
import grasp_process as gp

def release_and_home(controller):
    print("[Info] Executing Release and Home sequence...")
    
    # Target release pose (Placeholder)
    release_pose = np.array([0.3, 0.0, 0.2, 0.0, 1.57, 0.0], dtype=float)
    
    cfg = controller.get_controller_config()
    grip_max = cfg.gripper_width
    
    # 1. Move to release pose (keep gripper state)
    now = controller.get_timestamp() + cfg.default_preview_time
    eef_state = controller.get_eef_state()
    current_grip = eef_state.gripper_pos
    
    controller.set_eef_traj([
        grasp.build_eef_cmd(eef_state.pose_6d().copy(), current_grip, now),
        grasp.build_eef_cmd(release_pose, current_grip, now + 3.0)
    ])
    time.sleep(3.5)
    
    # 2. Open gripper
    now = controller.get_timestamp() + cfg.default_preview_time
    controller.set_eef_traj([
        grasp.build_eef_cmd(release_pose, current_grip, now),
        grasp.build_eef_cmd(release_pose, grip_max, now + 1)
    ])
    time.sleep(1.5)
    
    # 3. Reset to home
    print("[Info] Resetting to home...")
    controller.reset_to_home()
    time.sleep(3.5)

def main():
    print("=== Task 1 Full Workflow ===")
    # Mock args for grasp.short_loop
    # 直接复用 grasp_process 的参数解析；若未提供则注入默认 checkpoint 路径
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_ckpt = os.path.join(ROOT_DIR, 'python', 'graspnet', 'checkpoint', 'checkpoint-rs.tar')
    if '--checkpoint_path' not in sys.argv:
        sys.argv += ['--checkpoint_path', default_ckpt]
    args = gp.parse_args()

    # Initialize controller once
    controller = grasp.init_arm_controller()

    while True:
        if keyboard.is_pressed('z'):
            print("\n>>> 'z' pressed: Starting Grasp Sequence...")
            try:
                grasp.short_loop(args)
            except Exception as e:
                print(f"[Error] Grasp sequence failed: {e}")
            
            # Wait for key release to avoid multiple triggers
            while keyboard.is_pressed('z'): time.sleep(0.1)
            print(">>> Grasp Sequence Finished. Waiting for command...")

        elif keyboard.is_pressed('f'):
            print("\n>>> 'f' pressed: Releasing and Homing...")
            try:
                release_and_home(controller)
            except Exception as e:
                print(f"[Error] Release sequence failed: {e}")
            
            while keyboard.is_pressed('f'): time.sleep(0.1)
            print(">>> Release Finished. Waiting for command...")

        elif keyboard.is_pressed('q'):
            print("\n>>> Quitting...")
            break
        
        time.sleep(0.05)

if __name__ == "__main__":
    # Ensure root privileges for keyboard and can0 if needed
    if os.geteuid() != 0:
        print("Warning: This script might need sudo for keyboard/CAN access.")
    
    main()
