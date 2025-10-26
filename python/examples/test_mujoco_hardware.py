#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 MuJoCo 仿真实现的 IHardwareInterface，在控制器中运行一个最小验证流程。

先确保安装 MuJoCo：
    pip install mujoco
"""

import os
import sys
import time
import threading
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from arx5_interface import Arx5CartesianController, LogLevel
from peripherals.mujoco_hardware import MujocoHardware
import mujoco as mj
import mujoco.viewer as mjv


def build_system(robot_model: str, xml_path: str):
    mj_model = mj.MjModel.from_xml_path(xml_path)
    mj_data = mj.MjData(mj_model)

    # Python 侧设置一个很小的初始偏置，避免第一拍读到全 0
    eps_joint = 1e-3
    for i in range(1, 7):
        jid = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_JOINT, f"joint{i}")
        if jid != -1:
            qadr = int(mj_model.jnt_qposadr[jid])
            mj_data.qpos[qadr] = eps_joint
    mj.mj_forward(mj_model, mj_data)

    hw = MujocoHardware(robot_model=robot_model, mj_model=mj_model, mj_data=mj_data)
    controller = Arx5CartesianController(robot_model, hw)
    controller.set_log_level(LogLevel.INFO)
    return hw, controller, mj_model, mj_data


def start_stepper(hw: MujocoHardware, mj_model, mj_data, dt: float = 0.002):
    """启动固定频率的仿真步进线程（唯一调用 mj_step）。返回 (stop_flag, thread)。"""
    stop = {"flag": False}

    def _loop():
        next_t = time.perf_counter()
        while not stop["flag"]:
            next_t += dt
            with hw.lock:
                mj.mj_step(mj_model, mj_data)
            now = time.perf_counter()
            sleep_s = next_t - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = now

    th = threading.Thread(target=_loop, daemon=True)
    th.start()
    return stop, th

def run_viewer_loop(hw: MujocoHardware, mj_model, mj_data, controller: Arx5JointController, stop: dict,
                    view_hz: float = 60.0):
    """在主线程创建并运行可视化窗口，直到关闭或 stop.flag=True。"""
    refresh = 1.0 / max(1.0, float(view_hz))
    with mjv.launch_passive(mj_model, mj_data) as viewer:
        last_log = time.time()
        while viewer.is_running() and not stop.get("flag", False):
            # 为避免与步进线程竞争，渲染前加锁
            with hw.lock:
                viewer.sync()
            now = time.time()
            if now - last_log > 1.0:
                js = controller.get_joint_state()
                pos_arr = js.pos() if callable(js.pos) else js.pos
                print(f"t={controller.get_timestamp():.3f}s pos={np.array(pos_arr).round(3)} gripper={js.gripper_pos:.3f}m")
                last_log = now
            time.sleep(refresh)


def main():
    robot_model = "X5"
    xml_path = "/home/zishang/python_workspace/arx5-sdk/models/mujoco_asserts/arx_x5/scene.xml"

    hw, controller, mj_model, mj_data = build_system(robot_model, xml_path)
    stop, th = start_stepper(hw, mj_model, mj_data, dt=0.002)
    controller.reset_to_home()
    run_viewer_loop(hw, mj_model, mj_data, controller, stop, view_hz=60.0)
    stop["flag"] = True
    th.join(timeout=1.0)


if __name__ == "__main__":
    main()
