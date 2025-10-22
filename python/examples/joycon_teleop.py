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
    EEFState,
    Gain,
    LogLevel,
)

from joycon.joyconrobotics import JoyconRobotics

import time
import click


def start_joycon_teleop(controller: Arx5CartesianController, joycon: JoyconRobotics):
    
    cmd_dt = 0.01
    preview_time = 0.1
    window_size = 5
    joycon_queue = Queue(window_size)
    print("Starting teleop control with pre-calibrated Joycon...")

    def get_filtered_joycon_output():
        pose, gripper_pos, control_button = joycon.get_control()
        
        # Extract 6D pose (x, y, z, roll, pitch, yaw) - this is already absolute pose
        state = np.array(pose[:6], dtype=np.float64)

        if (
            joycon_queue.maxsize > 0
            and joycon_queue._qsize() == joycon_queue.maxsize
        ):
            joycon_queue._get()

        joycon_queue.put(state)

        filtered_state = np.mean(np.array(list(joycon_queue.queue)), axis=0)
        return filtered_state, gripper_pos, control_button
    
    start_time = time.monotonic()
    loop_cnt = 0
    
    while True:
        eef_state = controller.get_eef_state()
        
        # Get joycon state (absolute pose from joycon)
        joycon_pose, gripper_pos, control_button = get_filtered_joycon_output()

        print(
            f"Current --- time: {time.monotonic() - start_time:.03f}s, x: {eef_state.pose_6d()[0]:.03f}, y: {eef_state.pose_6d()[1]:.03f}, z: {eef_state.pose_6d()[2]:.03f}, roll: {eef_state.pose_6d()[3]:.03f}, pitch: {eef_state.pose_6d()[4]:.03f}, yaw: {eef_state.pose_6d()[5]:.03f}, gripper: {eef_state.gripper_pos:.03f}    ",
        end="\r",
        )
        
        # Check for exit command (control_button == 10 means both buttons pressed)
        if control_button == 10:
            print("\nBoth buttons pressed. Resetting to home and exiting...")
            break  # Exit the control loop
        
        # wait
        loop_cnt += 1
        target_time = start_time + loop_cnt * cmd_dt
        sleep_time = target_time - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)

        current_timestamp = controller.get_timestamp()
        eef_cmd = EEFState()
        eef_cmd.pose_6d()[:] = joycon_pose
        eef_cmd.gripper_pos = gripper_pos
        eef_cmd.timestamp = current_timestamp + preview_time
        controller.set_eef_cmd(eef_cmd)


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
def main(model: str, interface: str):
    # Initialize robot controller
    print("Initializing robot controller...")
    controller = Arx5CartesianController(model, interface)
    controller.reset_to_home()

    robot_config = controller.get_robot_config()
    gain = Gain(robot_config.joint_dof)
    controller.set_log_level(LogLevel.DEBUG)
    
    # Initialize Joycon (includes 2s calibration)
    print("Initializing Joycon controller (calibrating for 2 seconds)...")
    joycon = JoyconRobotics(
        device="right",
        translation_frame="local",
        direction_reverse=[1, 1, 1],
        euler_reverse=[-1, -1, 1],
        home_position=controller.get_home_pose().tolist()[:3],
        limit_dof=True,
        glimit=[[0.0, -0.5, -0.5, -1.3, -1.3, -1.3], 
                [0.5, 0.5, 0.5, 1.3, 1.3, 1.3]],
        gripper_limit=[0.0, robot_config.gripper_width],
    )
    print("Joycon calibration completed.")

    np.set_printoptions(precision=4, suppress=True)
    start_joycon_teleop(controller, joycon)
    controller.reset_to_home()  
    controller.set_to_damping()    
    joycon.disconnect()


if __name__ == "__main__":
    main()