import time
from queue import Queue
import os
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
from arx5_interface import (
    Arx5CartesianController,
    ControllerConfig,
    ControllerConfigFactory,
    EEFState,
    Gain,
    LogLevel,
    RobotConfigFactory,
)

from joycon.joyconrobotics import JoyconRobotics

import time
import click


def start_joycon_teleop(controller: Arx5CartesianController, joycon: JoyconRobotics):

    # Deadzone threshold to avoid jitter when joycon is at rest
    deadzone_threshold = 0.01
    
    cmd_dt = 0.01
    preview_time = 0.1
    window_size = 5
    joycon_queue = Queue(window_size)
    robot_config = controller.get_robot_config()
    controller_config = controller.get_controller_config()

    print("Starting teleop control with pre-calibrated Joycon...")

    def get_filtered_joycon_output():
        pose, gripper_pos, control_button = joycon.get_control()
        
        # Extract 6D pose (x, y, z, roll, pitch, yaw) - this is already absolute pose
        state = np.array(pose[:6], dtype=np.float64)
        
        # Apply deadzone to reduce jitter
        mask = np.abs(state) > deadzone_threshold
        state = state * mask

        if (
            joycon_queue.maxsize > 0
            and joycon_queue._qsize() == joycon_queue.maxsize
        ):
            joycon_queue._get()

        joycon_queue.put(state)

        filtered_state = np.mean(np.array(list(joycon_queue.queue)), axis=0)
        return filtered_state, gripper_pos, control_button

    # Get initial home pose as offset
    home_pose = controller.get_home_pose()
    
    start_time = time.monotonic()
    loop_cnt = 0
    
    while True:
        eef_state = controller.get_eef_state()
        print(
            f"Time elapsed: {time.monotonic() - start_time:.03f}s, x: {eef_state.pose_6d()[0]:.03f}, y: {eef_state.pose_6d()[1]:.03f}, z: {eef_state.pose_6d()[2]:.03f}",
            end="\r",
        )
        
        # Get joycon state (absolute pose from joycon)
        joycon_pose, joycon_gripper_pos, control_button = get_filtered_joycon_output()
        
        # Check for exit command (control_button == 10 means both buttons pressed)
        if control_button == 10:
            print("\nBoth buttons pressed. Resetting to home and exiting...")
            break  # Exit the control loop
        
        # Use joycon pose directly as target, offset from home position
        target_pose_6d = home_pose + joycon_pose
        
        # Update gripper position based on joycon gripper control
        # Map joycon gripper range [0, 0.8] to robot gripper range [0, gripper_width]
        target_gripper_pos = joycon_gripper_pos * robot_config.gripper_width / 0.8
        
        # Clamp gripper position
        if target_gripper_pos >= robot_config.gripper_width:
            target_gripper_pos = robot_config.gripper_width
        elif target_gripper_pos <= 0:
            target_gripper_pos = 0
            
        loop_cnt += 1
        while time.monotonic() < start_time + loop_cnt * cmd_dt:
            pass

        current_timestamp = controller.get_timestamp()
        eef_cmd = EEFState()
        eef_cmd.pose_6d()[:] = target_pose_6d
        eef_cmd.gripper_pos = target_gripper_pos
        eef_cmd.timestamp = current_timestamp + preview_time
        controller.set_eef_cmd(eef_cmd)


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
def main(model: str, interface: str):
    # Initialize Joycon first (includes 2s calibration)
    print("Initializing Joycon controller (calibrating for 2 seconds)...")
    joycon = JoyconRobotics(
        device="right",
        translation_frame="local",
        direction_reverse=[1, 1, 1],
        euler_reverse=[-1, -1, 1],
        offset_position_m=[0, 0, 0],
        limit_dof=True,
        glimit=[[-0.3, -0.3, -0.3, -1.0, -1.0, -1.0], 
                [0.3, 0.3, 0.3, 1.0, 1.0, 1.0]],
        gripper_limit=[0.0, 0.8],
    )
    print("Joycon calibration completed.")
    
    # Now initialize robot controller
    print("Initializing robot controller...")
    controller = Arx5CartesianController(model, interface)
    controller.reset_to_home()

    robot_config = controller.get_robot_config()
    gain = Gain(robot_config.joint_dof)
    controller.set_log_level(LogLevel.DEBUG)
    np.set_printoptions(precision=4, suppress=True)
    
    try:
        start_joycon_teleop(controller, joycon)
    except KeyboardInterrupt:
        print(f"\nTeleop recording is terminated. Resetting to home.")
        controller.reset_to_home()
        controller.set_to_damping()
    finally:
        # Clean up joycon connection
        joycon.disconnect()
        print("Joycon disconnected")


if __name__ == "__main__":
    main()