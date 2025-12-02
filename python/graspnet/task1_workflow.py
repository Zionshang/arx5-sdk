import os
import sys
import time
import numpy as np
# from pynput import keyboard

# Ensure compiled module under ../python is discoverable before importing it
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PY_ROOT = os.path.dirname(CUR_DIR)  # .../python
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

from arx5_interface import Arx5CartesianController, EEFState

# Add project root to path
ROOT_DIR = CUR_DIR
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import grasp.py functionality
import grasp
import grasp_process as gp
import lcm
import struct

# Global state
last_obj_id = -1
lc = lcm.LCM()

def release_and_home(controller):
    print("[Info] Executing Release and Home sequence...")
    
    # Target release pose (Placeholder)
    release_pose = np.array([ 0.4922,  0.001,  0.3725, -0.,  0.29, 0.    ], dtype=float)
    
    cfg = controller.get_controller_config()
    # gripper_width 在 robot_config 里，不在 controller_config
    robot_cfg = controller.get_robot_config()
    grip_max = robot_cfg.gripper_width
    
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

def send_status(status, obj_id):
    # status: 0=Success, 1=Fail
    # obj_id: YOLO class ID
    msg = struct.pack("ii", status, obj_id)
    lc.publish("ARM_STATUS", msg)
    print(f"[LCM] Sent Status: {status}, ID: {obj_id}")

def main():
    print("=== Task 1 Full Workflow (LCM Control) ===")
    # Mock args for grasp.short_loop
    default_ckpt = os.path.join(PY_ROOT, 'graspnet', 'checkpoint', 'checkpoint-rs.tar')
    if '--checkpoint_path' not in sys.argv:
        sys.argv += ['--checkpoint_path', default_ckpt]
    args = gp.parse_args()

    # Initialize controller once
    controller = grasp.init_arm_controller()

    def on_command(channel, data):
        global last_obj_id
        try:
            cmd = struct.unpack("i", data)[0]
            print(f"\n>>> Received LCM Command: {cmd}")
            
            if cmd == 0: # Grasp
                try:
                    obj_id = grasp.short_loop(args)
                    if obj_id is not None and obj_id >= 0:
                        last_obj_id = obj_id
                        send_status(0, obj_id)
                    else:
                        raise Exception("Grasp returned failure code")
                except Exception as e:
                    print(f"[Error] Grasp failed: {e}")
                    controller.reset_to_home()
                    send_status(1, -1)

            elif cmd == 1: # Place
                try:
                    release_and_home(controller)
                    send_status(0, last_obj_id)
                except Exception as e:
                    print(f"[Error] Place failed: {e}")
                    controller.reset_to_home()
                    send_status(1, last_obj_id)
                    
        except Exception as e:
            print(f"[LCM Error] {e}")

    lc.subscribe("ARM_CMD", on_command)
    print("Waiting for LCM commands on channel 'ARM_CMD'...")
    
    try:
        while True:
            lc.handle()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
