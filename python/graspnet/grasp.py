from __future__ import annotations

import os
import sys
import time
from typing import Optional, Tuple, Any

import numpy as np
import cv2
import open3d as o3d
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
# 确保可以导入项目根下的 arx5_interface（与 python/examples 中用法保持一致）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    try:
        os.chdir(ROOT_DIR)
    except Exception:
        pass
from arx5_interface import Arx5CartesianController, EEFState
from grasp2base.convert import convert_new
# 同目录导入现有实现（其内部已设置 models/utils/graspnetAPI 路径）
# 确保可以导入同目录下的 grasp_process.py
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)
import grasp_process as gp
#手眼标定外参
handeye_rotation = [[-0.02489131, -0.16662419 , 0.98570624],
 [-0.99968  ,   0.00859452, -0.02379136],
 [-0.00450745, -0.98598302, -0.1667848 ]]
handeye_translation = [-0.09760795,0.02448454,0.0883561]

T_o3d = np.eye(4, dtype=np.float64)
T_o3d[:3, :3] = np.array([[1.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0],
                          [0.0, 0.0, -1.0]], dtype=np.float64)
 

# =============== 机械臂控制（简洁接口） ===============
_ARM_CONTROLLER = None  # 缓存控制器，避免重复初始化

def init_arm_controller(model: str = "X5", interface: str = "can0"):
    """初始化并返回机械臂控制器（懒加载，重复调用直接复用）。"""
    global _ARM_CONTROLLER
    if _ARM_CONTROLLER is not None:
        return _ARM_CONTROLLER
    # 确保可以导入 arx5_interface
    PY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PY_ROOT not in sys.path:
        sys.path.insert(0, PY_ROOT)
    _ARM_CONTROLLER = Arx5CartesianController(model, interface)
    return _ARM_CONTROLLER

def arm_time_and_state(model: str = "X5", interface: str = "can0"):
    ctrl = init_arm_controller(model, interface)
    cfg = ctrl.get_controller_config()
    base_ts = ctrl.get_timestamp() + cfg.default_preview_time
    eef_state = ctrl.get_eef_state()
    return ctrl, base_ts, eef_state

def build_eef_cmd(pose: np.ndarray, grip: float, timestamp: float):
    cmd = EEFState()
    cmd.pose_6d()[:] = pose
    cmd.gripper_pos = grip
    cmd.timestamp = timestamp
    return cmd

# --------------------------- 小工具：初始化 ---------------------------
def init_yolo(root_dir: str, target_class_id: int = 46):
    """初始化 YOLO 分割模型（若不可用则返回 None）。

    返回 (yolo_model, yolo_predict_params)
    """
    yolo_model = None
    params = None
    try:
        if getattr(gp, "_HAS_YOLO", False) and getattr(gp, "YOLO", None) is not None:
            weights = os.path.join(root_dir, 'yolo11', 'yolo11n-seg.pt')
            yolo_model = gp.YOLO(weights)
            params = {"conf": 0.4, "iou": 0.7, "classes": [target_class_id]}
    except Exception as e:
        print(f"[Warn] YOLO init failed: {e}")
        yolo_model, params = None, None
    return yolo_model, params


def init_realsense(color_w: int = 640, color_h: int = 480):
    """初始化 RealSense（颜色/深度对齐到彩色），返回 (pipeline, align)。"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, color_w, color_h, rs.format.z16, 30)
    align = rs.align(rs.stream.color)
    pipeline.start(config)

    return pipeline, align


def make_camera_info(color_w: int, color_h: int) -> gp.CameraInfo:
    """构造相机内参（与 grasp_process.py main 中保持一致，默认 L515 内参）。"""
    # L515
    # intrinsic = np.array(
    #     [[607.451721, 0.0, 329.049744],
    #      [0.0, 607.656555, 248.114029],
    #      [0.0, 0.0, 1.0]], dtype=np.float32
    # )
    # factor_depth = float(3999.999810)
    # D435i
    intrinsic = np.array([[606.44, 0.0, 322.35], [0.0, 606.48, 239.54], [0.0, 0.0, 1.0]], dtype=np.float32)
    factor_depth = float(999.999952502551)
    return gp.CameraInfo(float(color_w), float(color_h),
                         intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                         factor_depth)


def grasp_control_step0(grasp_translation, grasp_rotation, width, current_pose, handeye_rotation, handeye_translation):
    
    #打印位姿信息
    np.set_printoptions(precision=5, suppress=True)
    print(f"grasp_translation (m):\n{grasp_translation}")
    print(f"grasp_rotation_matrix:\n{grasp_rotation}")
    print(f"width (m): {width:.5f}")

    # gripper_length 单位为米；负号在 convert_new 内部已处理为沿 -X 方向后退
    base_pose, _ = convert_new(
        grasp_translation,
        grasp_rotation,
        current_pose,
        handeye_rotation,
        handeye_translation,
        gripper_length=0.01,
    )

    # 正式执行部分
    base_pose_np = np.array(base_pose, dtype=float)
    base_xyz = base_pose_np[:3]
    base_rxyz = base_pose_np[3:]

    # 预抓取计算01：
    pre_grasp_pose_01 = base_pose_np.copy()
    pre_grasp_pose_01[0] -= 0.16  # x 值减去 0.16m
    pre_grasp_pose_01[2] += 0.14  # z 值增加 0.14m
    pre_grasp_pose_01[3:] = [0., 0.8, 0.]  # rx, ry, rz
    print(f"pre-grasp_pose_01:\n{pre_grasp_pose_01}")

    controller, now, eef_state = arm_time_and_state()
    grip_now = eef_state.gripper_pos

    controller.set_eef_traj([
        build_eef_cmd(current_pose, grip_now, now),
        build_eef_cmd(pre_grasp_pose_01, grip_now, now + 2.0),
    ])
def grasp_control_step1(grasp_translation, grasp_rotation, width, current_pose, handeye_rotation, handeye_translation):
    
    #打印位姿信息
    np.set_printoptions(precision=5, suppress=True)
    print(f"grasp_translation (m):\n{grasp_translation}")
    print(f"grasp_rotation_matrix:\n{grasp_rotation}")
    print(f"width (m): {width:.5f}")

    # gripper_length 单位为米；负号在 convert_new 内部已处理为沿 -X 方向后退
    base_pose, _ = convert_new(
        grasp_translation,
        grasp_rotation,
        current_pose,
        handeye_rotation,
        handeye_translation,
        gripper_length=0.01,
    )
    print("[DEBUG] 基坐标系抓取位姿:", base_pose)

    # 正式执行部分
    base_pose_np = np.array(base_pose, dtype=float)
    base_xyz = base_pose_np[:3]
    base_rxyz = base_pose_np[3:]


    controller, now, eef_state = arm_time_and_state()
    grip_now = eef_state.gripper_pos
    # grip_target = float(controller.get_robot_config().gripper_width - 0.03)
    # Ensure grip_target meets width-0.03 but is not negative
    grip_target = max(0.0, float(width - 0.03))
    lift_pose = base_pose_np.copy()
    lift_pose[2] += 0.1  # raise 10 cm after the grasp closes

    # 最终位置：回到关节复位状态的EEF位姿（假设关节0时的EEF位姿为[0,0,0,0,0,0]，夹爪保持grip_target）
    final_pose = np.array([ 0.2402, 0.001, 0.1565, -0., 0.,  0. ], dtype=float)

    controller.set_eef_traj([
        build_eef_cmd(current_pose, grip_now, now),
        build_eef_cmd(base_pose_np, grip_now, now + 2.0),
        build_eef_cmd(base_pose_np, grip_target, now + 4.0),
        build_eef_cmd(lift_pose, grip_target, now + 6.0),
        build_eef_cmd(final_pose, grip_target, now + 10.0),
    ])


def acquire_and_pregrasp(net, device, pipeline, align, camera_info, args, pcd, yolo_model, yolo_params):
    handeye_rot = np.array(handeye_rotation, dtype=float)
    handeye_trans = np.array(handeye_translation, dtype=float)
    grasp = None
    while True:
        color, depth = capture_frame(pipeline, align)
        if color is None or depth is None:
            continue
        mask, _ = gp.yolo_get_mask(yolo_model, color, yolo_params)
        if mask is None:
            continue

        _, _, eef_state = arm_time_and_state()
        current_pose = eef_state.pose_6d().copy()

        grasp = gp.run_graspnet_for_mask(
            net, device, color, depth, camera_info, args, pcd, T_o3d, mask,
            current_pose, handeye_rot, handeye_trans
        )
        if grasp is not None:
           break
    if grasp is not None:
        grasp_control_step0(
            grasp['translation'], grasp['rotation_matrix'], grasp['width'],
            current_pose, handeye_rotation, handeye_translation
        )
        time.sleep(3)
    return grasp


def acquire_and_execute_final_grasp(net, device, pipeline, align, camera_info, args, pcd, yolo_model, yolo_params):
    handeye_rot = np.array(handeye_rotation, dtype=float)
    handeye_trans = np.array(handeye_translation, dtype=float)
    grasp_candidates = []

    while len(grasp_candidates) < 5:
        color, depth = capture_frame(pipeline, align)
        if color is None or depth is None:
            continue
        mask, _ = gp.yolo_get_mask(yolo_model, color, yolo_params)
        if mask is None:
            continue

        _, _, eef_state = arm_time_and_state()
        current_pose = eef_state.pose_6d().copy()

        grasp = gp.run_graspnet_for_mask(
            net, device, color, depth, camera_info, args, pcd, T_o3d, mask,
            current_pose, handeye_rot, handeye_trans
        )
        if grasp is None:
            continue

        angle_x = grasp.get('angle_x')
        if angle_x is None:
            continue
        grasp_candidates.append((grasp, current_pose))

    best_grasp, _ = min(
        grasp_candidates,
        key=lambda item: item[0].get('angle_x', float('inf'))
    )

    # Re-read pose before execution to minimize drift.
    _, _, eef_state = arm_time_and_state()
    exec_pose = eef_state.pose_6d().copy()

    grasp_control_step1(
        best_grasp['translation'],
        best_grasp['rotation_matrix'],
        best_grasp['width'],
        exec_pose,
        handeye_rotation,
        handeye_translation
    )
    time.sleep(11)
    return best_grasp


# --------------------------- 主循环（精炼） ---------------------------
def capture_frame(pipeline: Any, align: Any, timeout_ms: int = 10000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Grab a synchronized color/depth frame pair from the RealSense pipeline.
    超时或偶发异常时返回 (None, None) 而不是抛出，让上层继续循环，避免程序退出导致机械臂进入阻尼。
    """
    try:
        frames = pipeline.wait_for_frames(timeout_ms)
        if not frames:
            return None, None
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        return color, depth
    except Exception:
        # 取帧超时或摄像头临时不可用，返回空让主循环继续
        return None, None


def short_loop(args):
    """主流程：初始化 -> 循环处理 -> 窗口与键盘交互。"""
    # 模型
    net, device = gp.get_net(args.checkpoint_path, args.num_view)

    # YOLO
    yolo_model, yolo_params = init_yolo(gp.ROOT_DIR, target_class_id=46)

    # 可视化
    pcd = o3d.geometry.PointCloud()
    
    # arm_init
    controller = init_arm_controller()
    controller.reset_to_home()
    # 预抓取位姿
    # prep_pose = np.array([ 0.1522 ,0.001 , 0.2205 , -0. , 1.07 , 0. ], dtype=float)
    #竖直向下
    # prep_pose = np.array([ 0.2442, 0.001 , 0.2365 ,-0. , 1.35 , 0. ], dtype=float)
    #斜向下
    prep_pose = np.array([ 0.2122 ,0.001 ,0.2 ,-0., 0.66  , 0. ], dtype=float)
    _, start_ts, eef_state = arm_time_and_state()
    grip_home = eef_state.gripper_pos
    grip_max = controller.get_robot_config().gripper_width

    controller.set_eef_traj([
        build_eef_cmd(eef_state.pose_6d().copy(), grip_home, start_ts),
        build_eef_cmd(prep_pose, grip_home, start_ts + 3.0),
        build_eef_cmd(prep_pose, grip_max, start_ts + 5.0),
    ])
    time.sleep(6)
    # RealSense + 相机内参
    color_w, color_h = 640, 480
    camera_info = make_camera_info(color_w, color_h)
    pipeline, align = init_realsense(color_w, color_h)

    try:
        if yolo_model is None:
            print('[Info] YOLO 未初始化，无法执行自动抓取流程。')
            return

        print('[Info] Starting pre-grasp acquisition...')
        pre_grasp_info = acquire_and_pregrasp(
            net, device, pipeline, align, camera_info, args, pcd, yolo_model, yolo_params
        )
        if pre_grasp_info is None:
            print('[Warn] 未能生成有效的预抓取，终止流程。')
            return

        print('[Info] Collecting final grasp candidates...')
        acquire_and_execute_final_grasp(
            net, device, pipeline, align, camera_info, args, pcd, yolo_model, yolo_params
        )
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

# --------------------------- 入口 ---------------------------
def main():
    # 直接复用 grasp_process 的参数解析；若未提供则注入默认 checkpoint 路径
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_ckpt = os.path.join(ROOT_DIR, 'python', 'graspnet', 'checkpoint', 'checkpoint-rs.tar')
    if '--checkpoint_path' not in sys.argv:
        sys.argv += ['--checkpoint_path', default_ckpt]
    args = gp.parse_args()
    short_loop(args)


if __name__ == '__main__':
    main()
