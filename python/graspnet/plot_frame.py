# -*- coding: utf-8 -*-
# 可视化：世界坐标系 与 由 (R, t) 定义的坐标系（x/y/z 轴）
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def set_axes_equal(ax):
    # 让 3D 坐标轴比例相等
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    ranges = [x_limits[1]-x_limits[0], y_limits[1]-y_limits[0], z_limits[1]-z_limits[0]]
    centers = [(x_limits[0]+x_limits[1])/2.0, (y_limits[0]+y_limits[1])/2.0, (z_limits[0]+z_limits[1])/2.0]
    radius = 0.5 * max(ranges)
    ax.set_xlim3d([centers[0]-radius, centers[0]+radius])
    ax.set_ylim3d([centers[1]-radius, centers[1]+radius])
    ax.set_zlim3d([centers[2]-radius, centers[2]+radius])

if __name__ == "__main__":
    # 1) 定义旋转矩阵 R 和平移 t（示例：先绕Z 30°，再绕Y 20°，再绕X 10°）
    def rot_x(a): 
        c, s = np.cos(a), np.sin(a)
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    def rot_y(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    def rot_z(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])
    R = rot_y(1.5181385170733657) @ rot_y(0.2492888226322032) @ rot_x(-1.7258007925283423)

    R = np.array([[ 0.25098,  0.92983,  0.2691 ],
 [-0.78852,  0.35764, -0.50034],
 [-0.56147 ,-0.08662 , 0.82295]])


    t = np.array([0.41321, 0.04455, 0.11441])  # 平移（单位：米）

    # 2) 轴长度
    L = 0.1  # 每根轴的显示长度
    ex, ey, ez = np.array([L,0,0]), np.array([0,L,0]), np.array([0,0,L])

    # 3) 世界坐标系（原点与轴）
    Ow = np.zeros(3)
    Wx, Wy, Wz = ex, ey, ez

    # 4) 目标坐标系（由 R,t 定义）——轴方向是 R 的列向量
    Of = t
    Fx, Fy, Fz = R[:,0]*L, R[:,1]*L, R[:,2]*L  # 分别对应 x/y/z 轴方向

    # 5) 画图
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')

    # 世界系
    ax.quiver(*Ow, *Wx, color='r', linewidth=2)
    ax.quiver(*Ow, *Wy, color='g', linewidth=2)
    ax.quiver(*Ow, *Wz, color='b', linewidth=2)
    ax.scatter(*Ow, color='k', s=30)
    ax.text(*(Ow + np.array([0,0,0.02])), 'World', color='k')

    # 目标系（R,t）
    ax.quiver(*Of, *Fx, color='r', linestyle='--', linewidth=2)
    ax.quiver(*Of, *Fy, color='g', linestyle='--', linewidth=2)
    ax.quiver(*Of, *Fz, color='b', linestyle='--', linewidth=2)
    ax.scatter(*Of, color='m', s=30)
    ax.text(*(Of + np.array([0,0,0.02])), 'Frame (R,t)', color='m')

    # 辅助范围（包含世界轴端点与目标轴端点）
    pts = np.stack([
        Ow, Ow+Wx, Ow+Wy, Ow+Wz,
        Of, Of+Fx, Of+Fy, Of+Fz
    ], axis=0)
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    margin = 0.05
    ax.set_xlim(mins[0]-margin, maxs[0]+margin)
    ax.set_ylim(mins[1]-margin, maxs[1]+margin)
    ax.set_zlim(mins[2]-margin, maxs[2]+margin)

    set_axes_equal(ax)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('World frame (solid) and Frame(R,t) (dashed)')
    plt.tight_layout()
    plt.show()