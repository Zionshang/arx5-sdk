#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MujocoHardware: 极简的 MuJoCo 硬件接口（Python）。

- 固定 X5 命名：joint1..joint6 + joint7/joint8（gripper），actuator "gripper"
- 仅位置控制：send_* 写 data.ctrl；read_state 读 qpos/qvel，torque 置 0
- 不在类内步进；步进放到外部线程或主循环
- 必须注入外部创建的 mj_model/mj_data（统一全局状态）
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import mujoco as mj

from arx5_interface import IHardwareInterface, JointState, RobotConfigFactory



class MujocoHardware(IHardwareInterface):
    def __init__(self, robot_model: str,
                 mj_model: mj.MjModel,
                 mj_data: mj.MjData):
        super().__init__()
        # 配置
        self.robot_cfg = RobotConfigFactory.get_instance().get_config(robot_model)
        self.dof = int(self.robot_cfg.joint_dof)

        # MuJoCo 模型/数据（外部注入）
        self.model = mj_model
        self.data = mj_data

        # 固定名称映射（X5）
        joint_names = [f"joint{i}" for i in range(1, self.dof + 1)]
        actuator_names = joint_names
        self._jnt_ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        self._jnt_qposadr = [int(self.model.jnt_qposadr[jid]) for jid in self._jnt_ids]
        self._jnt_dofadr = [int(self.model.jnt_dofadr[jid]) for jid in self._jnt_ids]
        self._act_ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, n) for n in actuator_names]

        self._gid_1 = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "joint7")
        self._gid_2 = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "joint8")
        self._gid1_qposadr = int(self.model.jnt_qposadr[self._gid_1])
        self._gid2_qposadr = int(self.model.jnt_qposadr[self._gid_2])
        self._gid1_dofadr = int(self.model.jnt_dofadr[self._gid_1])
        self._gid2_dofadr = int(self.model.jnt_dofadr[self._gid_2])

        self._gripper_act = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "gripper")

        self._lock = threading.Lock()


    # ---- IHardwareInterface 接口实现 ----
    def read_state(self) -> JointState:
        with self._lock:
            joint_state = JointState(self.dof)
            for i in range(self.dof):
                qadr = self._jnt_qposadr[i]
                dadr = self._jnt_dofadr[i]
                joint_state.pos()[i] = float(self.data.qpos[qadr])
                joint_state.vel()[i] = float(self.data.qvel[dadr])
                joint_state.torque()[i] = 0.0

            # gripper（宽度 = joint7 - joint8）
            joint_state.gripper_pos = self.data.qpos[self._gid1_qposadr] - self.data.qpos[self._gid2_qposadr]
            joint_state.gripper_vel = self.data.qvel[self._gid1_dofadr] - self.data.qvel[self._gid2_dofadr]
            joint_state.gripper_torque = 0.0

            return joint_state

    def send_joint_command(
        self, joint_index: int, kp: float, kd: float, pos: float, vel: float, torque: float
    ) -> None:
        aid = self._act_ids[joint_index]
        with self._lock:
            val = float(pos)
            self.data.ctrl[aid] = val

    def send_gripper_command(self, kp: float, kd: float, pos: float, vel: float, torque: float) -> None:
        # pos 为开口宽度（m）-> joint7 目标 = pos/2（因为 joint8 = -joint7）
        with self._lock:
            self.data.ctrl[self._gripper_act] = 0.5 * pos

    def enable_joint(self, joint_index: int) -> None:
        return

    def enable_gripper(self) -> None:
        return

    def set_zero_at_current_joint(self, joint_index: int) -> None:
        return

    def set_zero_at_current_gripper(self) -> None:
        return

    @property
    def lock(self):
        return self._lock


__all__ = ["MujocoHardware"]
