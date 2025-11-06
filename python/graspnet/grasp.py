"""
精炼版实时抓取推理入口。

功能：
- 复用 grasp_process.py 中已有的函数（YOLO 分割、点云准备、GraspNet 推理与可视化）。
- 负责初始化相机、模型与可视化，并在循环中调用现成函数得到抓取结果：
  translation, rotation_matrix, width。

注意：不修改 grasp_process.py，尽量保持本文件简洁。
"""

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
handeye_rotation = [[-0.02489131, -0.16662419, 0.98570624],
                   [-0.99968, 0.00859452, -0.02379136],
                   [-0.00450745, -0.98598302, -0.1667848]]
handeye_translation = [-0.0702653, 0.03149889, 0.06295003]

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


# def init_vis(window_name: str = 'GraspNet Live', w: int = 1280, h: int = 720):
#     """初始化 Open3D 可视化器与点云，返回 (vis, pcd, T)。"""
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name=window_name, width=w, height=h)
#     pcd = o3d.geometry.PointCloud()
#     vis.add_geometry(pcd)
#     # 调整显示方向
#     T = np.eye(4, dtype=np.float64)
#     T[:3, :3] = np.array([[1.0, 0.0, 0.0],
#                           [0.0, -1.0, 0.0],
#                           [0.0, 0.0, -1.0]], dtype=np.float64)
#     return vis, pcd, T

def grasp_control(grasp_translation, grasp_rotation, width, current_pose, handeye_rotation, handeye_translation):
    
    #打印位姿信息
    np.set_printoptions(precision=5, suppress=True)
    print(f"grasp_translation (m):\n{grasp_translation}")
    print(f"grasp_rotation_matrix:\n{grasp_rotation}")
    print(f"width (m): {width:.5f}")
    # 抓取位姿计算
    base_pose = convert_new(grasp_translation, grasp_rotation, current_pose, handeye_rotation, handeye_translation)
    print("[DEBUG] 基坐标系抓取位姿:", base_pose)

    # 正式执行部分
    base_pose_np = np.array(base_pose, dtype=float)
    base_xyz = base_pose_np[:3]
    base_rxyz = base_pose_np[3:]

    # 预抓取计算01：
    pre_grasp_offset_01 = 0.15
    pre_grasp_pose_01 = np.array(base_pose, dtype=float).copy()
    # # 按 SDK 约定使用 XYZ 顺序（roll, pitch, yaw，弧度）构造旋转矩阵
    # rotation_mat = R.from_euler('XYZ', pre_grasp_pose_01[3:], degrees=False).as_matrix()
    # x_axis = rotation_mat[:, 0]
    # pre_grasp_pose_01[:3] -= x_axis * pre_grasp_offset_01
    # pre_grasp_pose_01[2] += pre_grasp_offset_01
    rotation_mat = R.from_euler('xyz', pre_grasp_pose_01[3:], degrees=False).as_matrix()
    x_axis = rotation_mat[:, 0]
    pre_grasp_pose_01[:3] -= x_axis * pre_grasp_offset_01
    rotation_mat_01 = R.from_euler('xyz', pre_grasp_pose_01[3:], degrees=False).as_matrix()
    print(f"pre-grasp_pose_01:\n{pre_grasp_pose_01}")
    print(f"pre-rotation_matrix-01:\n{rotation_mat_01}")

    #预抓取计算02：
    # pre_grasp_offset_02 = 0.1
    # pre_grasp_pose_02 = np.array(base_pose, dtype=float).copy()
    # 按 SDK 约定使用 XYZ 顺序（roll, pitch, yaw，弧度）构造旋转矩阵
    # rotation_mat = R.from_euler('XYZ', pre_grasp_pose_02[3:], degrees=False).as_matrix()
    # x_axis = rotation_mat[:, 0]
    # pre_grasp_pose_02[:3] -= x_axis * pre_grasp_offset_02
    # pre_grasp_pose_02[2] += pre_grasp_offset_02
    # rotation_mat_02 = R.from_euler('XYZ', pre_grasp_pose_02[3:], degrees=False).as_matrix()
    # print(f"pre-grasp_pose_02:\n{pre_grasp_pose_02}")
    # print(f"pre-rotation_matrix-02:\n{rotation_mat_02}")

    controller, now, eef_state = arm_time_and_state()
    # grip_max = controller.get_robot_config().gripper_width
    grip_now = eef_state.gripper_pos

    controller.set_eef_traj([
        build_eef_cmd(current_pose, grip_now, now),
        build_eef_cmd(pre_grasp_pose_01, grip_now, now + 5.0),
        # build_eef_cmd(pre_grasp_pose_02, grip_max, now + 11.0),
    ])




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
    if yolo_model is None:
        print('[Info] YOLO not available; skipping segmentation.')

    # 可视化
    # vis, pcd, T = init_vis()
    pcd = o3d.geometry.PointCloud()
    # gripper_geoms = []
    last_grasp_info = None
    seg_vis = None

    # arm_init
    controller = init_arm_controller()
    controller.reset_to_home()
    # 预抓取位姿
    # prep_pose = np.array([0.3808947838198619,
    #                      0.0010951536627964757,
    #                      0.23226317113384085,
    #                      0.003068532940281973,
    #                      1.2510476636525898,
    #                      0.002855123071475998], dtype=float)
    prep_pose = np.array([ 0.1522 ,0.001 , 0.2205 , -0. , 1.07 , 0. ], dtype=float)
    _, start_ts, eef_state = arm_time_and_state()
    grip_home = eef_state.gripper_pos
    grip_max = controller.get_robot_config().gripper_width

    controller.set_eef_traj([
        build_eef_cmd(eef_state.pose_6d().copy(), grip_home, start_ts),
        build_eef_cmd(prep_pose, grip_home, start_ts + 6.0),
        build_eef_cmd(prep_pose, grip_max, start_ts + 10.0),
    ])

    # RealSense + 相机内参
    color_w, color_h = 640, 480
    camera_info = make_camera_info(color_w, color_h)
    pipeline, align = init_realsense(color_w, color_h)


    try:
        while True:
            color, depth = capture_frame(pipeline, align)
            if color is None or depth is None:
                print("[Warn] 未能获取相机帧，跳过本次循环。")
                continue

            cv2.imshow('Color (raw)', color)
            depth_colormap = gp.depth_to_colormap(depth)
            if depth_colormap is not None:
                cv2.imshow('Depth (aligned)', depth_colormap)
            if seg_vis is not None:
                cv2.imshow('YOLO Segmentation', seg_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                print("[Safety] Returning arm to home pose.")
                controller.reset_to_home()
            elif key == ord('t'):
                if yolo_model is not None:
                    workspace_mask, seg_vis_candidate = gp.yolo_get_mask(yolo_model, color, yolo_params)
                    seg_vis = seg_vis_candidate
                else:
                    workspace_mask = np.where(depth > 0, 255, 0).astype(np.uint8)
                    seg_vis = None

                if workspace_mask is None:
                    print('[Info] YOLO 未检测到目标，跳过抓取生成。')
                    # if gripper_geoms:
                    #     for g in gripper_geoms:
                    #         try:
                    #             vis.remove_geometry(g)
                    #         except Exception:
                    #             pass
                    # gripper_geoms = []
                    last_grasp_info = None
                else:
                    last_grasp_info = gp.run_graspnet_for_mask(
                        net, device, color, depth, camera_info, args, pcd, T_o3d, workspace_mask
                    )
            elif key == ord('z'):
                print("\n===== Current Grasp (camera frame) =====")
                if last_grasp_info is not None:
                    grasp_translation = last_grasp_info['translation']
                    grasp_rotation = last_grasp_info['rotation_matrix']
                    grasp_width = last_grasp_info['width']
                    _, _, eef_state = arm_time_and_state()
                    current_pose = eef_state.pose_6d().copy()
                    grasp_control(grasp_translation, grasp_rotation, grasp_width, current_pose, handeye_rotation, handeye_translation)
                    time.sleep(15)
                    print('当前末端执行器位姿:', controller.get_eef_state().pose_6d())
                    # time.sleep(20)
                    # controller.reset_to_home()
                    # time.sleep(10)
                    # break
                else:
                    print("No grasp available yet.")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


# --------------------------- 入口 ---------------------------
def main():
    # 直接复用 grasp_process 的参数解析；若未提供则注入默认 checkpoint 路径
    default_ckpt = os.path.join(CUR_DIR, 'checkpoint', 'checkpoint-rs.tar')
    if '--checkpoint_path' not in sys.argv:
        sys.argv += ['--checkpoint_path', default_ckpt]
    args = gp.parse_args()
    short_loop(args)


if __name__ == '__main__':
    main()
