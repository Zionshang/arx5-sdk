from typing import Optional, Sequence
from scipy.spatial.transform import Rotation as SciRot


class PoseVisualizer:
    """
    简单的 3D 位姿可视化器（matplotlib）。
        - 严格模式：初始化选择姿态格式 orientation_format='euler' 或 'quat'（不自动推断）
        - update 适配：
            Euler: [x,y,z, roll, pitch, yaw]
            Quaternion: [x,y,z, qx, qy, qz, qw] 或 [x,y,z, qx, qy, qz]（缺省时推断 qw）
    """

    def __init__(
        self,
        axis_len: float = 0.1,
        world_axis_len: float = 0.2,
        window_title: str = "Joycon Pose",
        limits: Optional[Sequence[Sequence[float]]] = None,
        orientation_format: str = "quat",
    ) -> None:
        self.axis_len = axis_len
        self.world_axis_len = world_axis_len
        self.window_title = window_title
        self.orientation_format = (orientation_format or "quat").lower()
        if self.orientation_format not in ("euler", "quat"):
            raise ValueError("orientation_format must be 'euler' or 'quat'")

        self.fig = None
        self.ax = None
        self._limits = [list(x) for x in limits] if limits else []

    def start(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        plt.ion()
        self.fig = plt.figure(figsize=(6, 6))
        try:
            self.fig.canvas.manager.set_window_title(self.window_title)
        except Exception:
            pass
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")

        if not self._limits:
            lim = 0.5
            self._limits = [[-lim, lim], [-lim, lim], [-lim, lim]]

        self._apply_limits()
        self._draw_world_axes()

        plt.tight_layout()
        plt.show(block=False)

    def _apply_limits(self):
        if not self.ax:
            return
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self._limits
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_zlim(zmin, zmax)
        self._set_equal_aspect()

    def _set_equal_aspect(self):
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self._limits
        max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
        xm = (xmax + xmin) / 2.0
        ym = (ymax + ymin) / 2.0
        zm = (zmax + zmin) / 2.0
        r = max_range / 2.0
        self.ax.set_xlim(xm - r, xm + r)
        self.ax.set_ylim(ym - r, ym + r)
        self.ax.set_zlim(zm - r, zm + r)

    def _draw_world_axes(self):
        L = self.world_axis_len
        self.ax.plot([0, L], [0, 0], [0, 0], color="#ffaaaa", linewidth=1)
        self.ax.plot([0, 0], [0, L], [0, 0], color="#aaffaa", linewidth=1)
        self.ax.plot([0, 0], [0, 0], [0, L], color="#aaaaff", linewidth=1)

    @staticmethod
    def _basis_from_euler_xyz(roll: float, pitch: float, yaw: float):
        Rm = SciRot.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()
        ex = (Rm[0, 0], Rm[1, 0], Rm[2, 0])
        ey = (Rm[0, 1], Rm[1, 1], Rm[2, 1])
        ez = (Rm[0, 2], Rm[1, 2], Rm[2, 2])
        return ex, ey, ez

    @staticmethod
    def _basis_from_quaternion(qx: float, qy: float, qz: float, qw: float):
        Rm = SciRot.from_quat([qx, qy, qz, qw]).as_matrix()
        ex = (Rm[0, 0], Rm[1, 0], Rm[2, 0])
        ey = (Rm[0, 1], Rm[1, 1], Rm[2, 1])
        ez = (Rm[0, 2], Rm[1, 2], Rm[2, 2])
        return ex, ey, ez

    def _infer_basis(self, pose: Sequence[float]):
        """严格根据 orientation_format 从 pose 中计算旋转基向量。"""
        n = len(pose)
        if n < 6:
            raise ValueError("pose 长度必须 >= 6")

        if self.orientation_format == "euler":
            if n < 6:
                raise ValueError("Euler 模式需要 [x,y,z, roll, pitch, yaw]")
            roll, pitch, yaw = pose[3], pose[4], pose[5]
            return self._basis_from_euler_xyz(roll, pitch, yaw)

        # quat 模式
        if n >= 7:
            qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
        else:
            # 允许缺省 w：按单位四元数补齐
            qx, qy, qz = pose[3], pose[4], pose[5]
            qw_sq = max(0.0, 1.0 - (qx*qx + qy*qy + qz*qz))
            qw = (qw_sq) ** 0.5
        norm = (qx*qx + qy*qy + qz*qz + qw*qw) ** 0.5
        if norm > 1e-9:
            qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        return PoseVisualizer._basis_from_quaternion(qx, qy, qz, qw)

    def update(self, pose: Sequence[float]):
        if self.ax is None:
            self.start()
        if pose is None or len(pose) < 6:
            return

        x, y, z = pose[0], pose[1], pose[2]
        ex, ey, ez = self._infer_basis(pose)
        L = self.axis_len
        x_end = (x + ex[0] * L, y + ex[1] * L, z + ex[2] * L)
        y_end = (x + ey[0] * L, y + ey[1] * L, z + ey[2] * L)
        z_end = (x + ez[0] * L, y + ez[1] * L, z + ez[2] * L)

        # 重绘
        self.ax.cla()
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")
        self._apply_limits()
        self._draw_world_axes()
        self.ax.plot([x, x_end[0]], [y, x_end[1]], [z, x_end[2]], color="r", linewidth=3)
        self.ax.plot([x, y_end[0]], [y, y_end[1]], [z, y_end[2]], color="g", linewidth=3)
        self.ax.plot([x, z_end[0]], [y, z_end[1]], [z, z_end[2]], color="b", linewidth=3)

        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self._limits
        pad = 0.05
        changed = False
        if not (xmin <= x <= xmax):
            self._limits[0] = [min(xmin, x - pad), max(xmax, x + pad)]
            changed = True
        if not (ymin <= y <= ymax):
            self._limits[1] = [min(ymin, y - pad), max(ymax, y + pad)]
            changed = True
        if not (zmin <= z <= zmax):
            self._limits[2] = [min(zmin, z - pad), max(zmax, z + pad)]
            changed = True
        if changed:
            self._apply_limits()

        import matplotlib.pyplot as plt
        plt.pause(0.001)

    def close(self):
        try:
            import matplotlib.pyplot as plt
            if self.fig is not None:
                plt.close(self.fig)
        except Exception:
            pass
