#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
一个最小可用的测试：在 Python 里继承 IHardwareInterface，
把它注入到 Arx5JointController验证功能是否正常
"""

import os
import sys
import time
from typing import List

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
from arx5_interface import (
	IHardwareInterface,
	Arx5JointController,
	RobotConfigFactory,
	ControllerConfigFactory,
	LogLevel,
	JointState,
	RobotConfig
)


class MockHardware(IHardwareInterface):
	"""一个极简的“假硬件”实现，仅用于验证数据通路。
	"""

	def __init__(self, robot_cfg: RobotConfig):
		super().__init__()
		self.robot_cfg = robot_cfg

	def read_state(self):
		joint_state = JointState(self.robot_cfg.joint_dof)
		joint_state.pos = np.ones(self.robot_cfg.joint_dof)
		joint_state.gripper_pos = self.robot_cfg.gripper_width
		joint_state.gripper_vel = 0.0
		joint_state.gripper_torque = 0.05
		print(f"Read state - Joint positions: {joint_state.pos}")
		return joint_state


	def send_joint_command(self, joint_index: int, kp: float, kd: float, pos: float, vel: float, torque: float) -> None:
		print(f"Send joint command - Joint positions: {pos}")

	def send_gripper_command(self, kp: float, kd: float, pos: float, vel: float, torque: float) -> None:
		pass
	def enable_joint(self, joint_index: int) -> None:
		pass

	def enable_gripper(self) -> None:
		pass

	def set_zero_at_current_joint(self, joint_index: int) -> None:
		pass

	def set_zero_at_current_gripper(self) -> None:
		pass


def main():
	model = "X5"
	robot_cfg = RobotConfigFactory.get_instance().get_config(model)
	ctrl_cfg = ControllerConfigFactory.get_instance().get_config("joint_controller", robot_cfg.joint_dof)

	hw = MockHardware(robot_cfg)
	controller = Arx5JointController(model, hw)
	controller.set_log_level(LogLevel.INFO)
	controller.reset_to_home()


if __name__ == "__main__":
	main()

