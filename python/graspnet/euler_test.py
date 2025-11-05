"""极简验证：
SciPy from_euler 构造的旋转 与 手动矩阵乘法构造的旋转 是否一致。
输出两行：
- 矩阵是否一致
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]], dtype=float)


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=float)


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]], dtype=float)


angles = np.deg2rad([10, 20, 30])


# SciPy: 小写 'xyz' 表示内禀旋转（关于随体轴），等价于“反序”的外禀组合
R_scipy = R.from_euler("xyz", angles, degrees=False).as_matrix()
R_manual = Rz(angles[2]) @ Ry(angles[1]) @ Rx(angles[0])

print(np.allclose(R_scipy, R_manual, atol=1e-12))

R_scipy = R.from_euler("XYZ", angles, degrees=False).as_matrix()
R_manual = Rx(angles[0]) @ Ry(angles[1]) @ Rz(angles[2])

print(np.allclose(R_scipy, R_manual, atol=1e-12))

R_scipy = R.from_euler("ZYX", angles, degrees=False).as_matrix()
R_manual = Rz(angles[0]) @ Ry(angles[1]) @ Rx(angles[2])
print(np.allclose(R_scipy, R_manual, atol=1e-12))

R_scipy = R.from_euler("zyx", angles, degrees=False).as_matrix()
R_manual = Rx(angles[2]) @ Ry(angles[1]) @ Rz(angles[0])
print(np.allclose(R_scipy, R_manual, atol=1e-12))

# 对于常用的ZYX欧拉角
roll = 0.1
pitch = -0.2
yaw = 1.5

R_scipy = R.from_euler("ZYX", [yaw, pitch, roll], degrees=False).as_matrix()
R_manual = Rz(yaw) @ Ry(pitch) @ Rx(roll)
print(np.allclose(R_scipy, R_manual, atol=1e-12))
