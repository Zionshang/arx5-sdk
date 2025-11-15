# code by boxjod 2025.1.13 copyright Box2AI Robotics 盒桥智能 版权所有

import math
import time
from glm import vec3, quat, angleAxis

from .joycon import JoyCon
from .gyro import GyroTrackingJoyCon
from .event import ButtonEventJoyCon
from .device import get_R_id, get_L_id

from scipy.spatial.transform import Rotation
import numpy as np
import threading
import logging

JOYCON_SERIAL_SUPPORT = "9c:54:"


class LowPassFilter:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.prev_value = 0.0

    def update(self, new_value):
        self.prev_value = self.alpha * new_value + (1 - self.alpha) * self.prev_value
        return self.prev_value


class AttitudeEstimator:
    def __init__(
        self,
        pitch_Threhold=math.pi / 2.0,
        roll_Threhold=math.pi / 2.02,
        yaw_Threhold=-1,
        common_rad=True,
        lerobot=False,
        pitch_down_double=False,
        lowpassfilter_alpha_rate=0.05,
    ):
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.dt = 0.01
        self.alpha = 0.55

        self.yaw_diff = 0.0
        self.pitch_rad_T = pitch_Threhold
        self.roll_rad_T = roll_Threhold
        self.yaw_rad_T = yaw_Threhold

        self.common_rad = common_rad
        self.lerobot = lerobot
        self.pitch_down_double = pitch_down_double

        self.direction_X = vec3(1, 0, 0)
        self.direction_Y = vec3(0, 1, 0)
        self.direction_Z = vec3(0, 0, 1)
        self.direction_Q = quat()

        self.lowpassfilter_alpha = 0.05 * lowpassfilter_alpha_rate  # lerobot-plus 0.1
        if self.lerobot:
            self.lowpassfilter_alpha = 0.08 * lowpassfilter_alpha_rate

        self.lpf_roll = LowPassFilter(alpha=self.lowpassfilter_alpha)  # lerobot real
        self.lpf_pitch = LowPassFilter(alpha=self.lowpassfilter_alpha)  # lerobot real

    def reset_yaw(self):
        self.direction_X = vec3(1, 0, 0)
        self.direction_Y = vec3(0, 1, 0)
        self.direction_Z = vec3(0, 0, 1)
        self.direction_Q = quat()

    def set_yaw_diff(self, data):
        self.yaw_diff = data

    def update(self, gyro_in_rad, accel_in_g):
        self.pitch = 0.0
        self.roll = 0.0

        ax, ay, az = accel_in_g
        ax = ax * math.pi
        ay = ay * math.pi
        az = az * math.pi

        gx, gy, gz = gyro_in_rad

        # Calculate the pitch and roll angles provided by the accelerometers
        roll_acc = math.atan2(ay, -az)
        pitch_acc = math.atan2(ax, math.sqrt(ay**2 + az**2))

        # Updating angles with gyroscope data
        self.pitch += gy * self.dt
        self.roll -= gx * self.dt

        # Complementary filters: weighted fusion of accelerometer and gyroscope data
        self.pitch = self.alpha * self.pitch + (1 - self.alpha) * pitch_acc
        self.roll = self.alpha * self.roll + (1 - self.alpha) * roll_acc

        # The final output roll and pitch angles are then low-pass filtered
        self.pitch = self.lpf_pitch.update(self.pitch)
        self.roll = self.lpf_roll.update(self.roll)

        # Yaw angle (updated by gyroscope)
        rotation = (
            angleAxis(gx * (-1 / 86), self.direction_X)
            * angleAxis(gy * (-1 / 86), self.direction_Y)
            * angleAxis(gz * (-1 / 86), self.direction_Z)
        )

        self.direction_X *= rotation
        self.direction_Y *= rotation
        self.direction_Z *= rotation
        self.direction_Q *= rotation

        self.yaw = self.direction_X[1]

        if self.common_rad:
            self.roll = self.roll * math.pi / 1.5
            self.pitch = self.pitch * math.pi / 1.5
            self.yaw = -self.yaw * math.pi / 1.5  # * 10.0

        else:
            self.yaw = -self.yaw * math.pi / 2

        if self.pitch_down_double:
            self.pitch = self.pitch * 3.0 if self.pitch < 0 else self.pitch
        if self.lerobot:
            self.roll = self.roll * math.pi / 2
            # self.yaw = -self.yaw * math.pi/1.5 # * 10.0
        self.yaw = self.yaw - self.yaw_diff

        if self.pitch_rad_T != -1:
            self.pitch = (
                self.pitch_rad_T
                if self.pitch > self.pitch_rad_T
                else (-self.pitch_rad_T if self.pitch < -self.pitch_rad_T else self.pitch)
            )

        if self.roll_rad_T != -1:
            self.roll = (
                self.roll_rad_T
                if self.roll > self.roll_rad_T
                else (-self.roll_rad_T if self.roll < -self.roll_rad_T else self.roll)
            )

        if self.yaw_rad_T != -1:
            self.yaw = (
                self.yaw_rad_T
                if self.yaw > self.yaw_rad_T
                else (-self.yaw_rad_T if self.yaw < -self.yaw_rad_T else self.yaw)
            )

        orientation = [self.roll, self.pitch, self.yaw]
        # Return roll angle, pitch angle, yaw angle (in radians)
        return orientation


class JoyconRobotics:
    def __init__(
        self,
        device: str = "right",
        gripper_state: float = 0,
        gripper_limit: list = [0.0, 1.0],
        gripper_speed: float = 1.0,
        translation_frame: str = "global",  # global or local
        limit_dof: bool = False,
        glimit: list = [
            [0.125, -0.4, 0.046, -3.1, -1.5, -1.57],
            [0.380, 0.4, 0.23, 3.1, 1.5, 1.57],
        ],
        home_position: list = [0.0, 0.0, 0.0],  # just use the position and yaw
        home_euler_ard: list = [0.0, 0.0, 0.0],  # adjust the orientation
        euler_reverse: list = [1, 1, 1],  # -1 reverse
        direction_reverse: list = [1, 1, 1],  # -1 reverse
        dof_speed: list = [1, 1, 1, 1, 1, 1],
        rotation_filter_alpha_rate=1,
        common_rad: bool = True,
        lerobot: bool = False,
        pitch_down_double: bool = False,
        without_rest_init: bool = False,
    ):

        if device == "right":
            self.joycon_id = get_R_id()
        elif device == "left":
            self.joycon_id = get_L_id()
        else:
            print("get a wrong device name of joycon")

        if not self.joycon_id or self.joycon_id[0] is None:
            raise RuntimeError("No Joy-Con detected !!")

        device_serial = self.joycon_id[2][:6]

        # init joycon
        self.joycon = JoyCon(*self.joycon_id)

        self.gyro = GyroTrackingJoyCon(*self.joycon_id)
        self.lerobot = lerobot
        self.pitch_down_double = pitch_down_double
        self.rotation_filter_alpha_rate = rotation_filter_alpha_rate
        self.orientation_sensor = AttitudeEstimator(
            common_rad=common_rad,
            lerobot=self.lerobot,
            pitch_down_double=self.pitch_down_double,
            lowpassfilter_alpha_rate=self.rotation_filter_alpha_rate,
        )
        self.button = ButtonEventJoyCon(*self.joycon_id, track_sticks=True)
        self.without_rest_init = without_rest_init
        # print(f"connect to {device} joycon successful.")

        print(f"\033[32mconnect to {device} joycon successful.\033[0m")
        if not self.without_rest_init:
            self.reset_joycon()

        print()
        # more information
        self.gripper_state = gripper_state  # Increase indicates open
        self.gripper_limit = gripper_limit.copy()
        self.gripper_speed = gripper_speed
        self.position = home_position.copy()
        self.orientation_rad = home_euler_ard.copy()
        self.yaw_diff = 0.0

        self.home_position = home_position.copy()
        self.posture = home_position.copy()

        self.translation_frame = translation_frame
        self.if_limit_dof = limit_dof
        self.dof_speed = dof_speed.copy()
        self.glimit = glimit
        self.home_euler_ard = home_euler_ard
        self.euler_reverse = euler_reverse
        self.direction_reverse = direction_reverse
        # Start the thread to read inputs

        self.reset_button = 0
        self.next_episode_button = 0
        self.restart_episode_button = 0
        self.button_control = 0

        if device_serial != JOYCON_SERIAL_SUPPORT and self.joycon_id != None:
            raise IOError("There is no joycon for robotics")

        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.solve_loop, daemon=True)
        self.thread.start()

    def disconnect(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.joycon._close()
        print("Joycon disconnected")

    def reset_joycon(self):
        print(f"\033[33mcalibrating(2 seconds)..., please place it horizontally on the desktop.\033[0m")

        self.gyro.calibrate()
        time.sleep(2)
        self.gyro.reset_orientation()
        self.orientation_sensor.reset_yaw()

        print(f"\033[32mJoycon calibrations is complete.\033[0m")

    def check_limits_position(self):
        for i in range(3):
            self.position[i] = (
                self.glimit[0][i]
                if self.position[i] < self.glimit[0][i]
                else (self.glimit[1][i] if self.position[i] > self.glimit[1][i] else self.position[i])
            )

    def check_limits_orientation(self):
        for i in range(3):
            self.orientation_rad[i] = (
                self.glimit[0][3 + i]
                if self.orientation_rad[i] < self.glimit[0][3 + i]
                else (
                    self.glimit[1][3 + i]
                    if self.orientation_rad[i] > self.glimit[1][3 + i]
                    else self.orientation_rad[i]
                )
            )

    def check_limits_gripper(self):
        if self.gripper_state < self.gripper_limit[0]:
            self.gripper_state = self.gripper_limit[0]
        elif self.gripper_state > self.gripper_limit[1]:
            self.gripper_state = self.gripper_limit[1]

    def update_translation(self):

        trans_step = np.array([0.0, 0.0, 0.0], dtype=float)
        x_step = 0.001 * self.dof_speed[0] * self.direction_reverse[0]
        y_step = 0.001 * self.dof_speed[1] * self.direction_reverse[1]
        z_step = 0.001 * self.dof_speed[2] * self.direction_reverse[2]

        # X
        joycon_stick_v = (
            self.joycon.get_stick_right_vertical() if self.joycon.is_right() else self.joycon.get_stick_left_vertical()
        )
        if joycon_stick_v > 4000:
            trans_step[0] += x_step
        elif joycon_stick_v < 1000:
            trans_step[0] -= x_step

        # Y
        joycon_stick_h = (
            self.joycon.get_stick_right_horizontal()
            if self.joycon.is_right()
            else self.joycon.get_stick_left_horizontal()
        )
        if joycon_stick_h > 4000:
            trans_step[1] -= y_step
        elif joycon_stick_h < 1000:
            trans_step[1] += y_step

        # Z
        joycon_button_up = self.joycon.get_button_r() if self.joycon.is_right() else self.joycon.get_button_l()
        joycon_button_down = self.joycon.get_button_zr() if self.joycon.is_right() else self.joycon.get_button_zl
        if joycon_button_up == 1:
            trans_step[2] += z_step
        if joycon_button_down == 1:
            trans_step[2] -= z_step

        # # X/B 细调 x 轴
        # joycon_button_xup = self.joycon.get_button_x() if self.joycon.is_right() else self.joycon.get_button_up()
        # joycon_button_xback = self.joycon.get_button_b() if self.joycon.is_right() else self.joycon.get_button_down()
        # if joycon_button_xup == 1:
        #     trans_step[0] += x_step
        # elif joycon_button_xback == 1:
        #     trans_step[0] -= x_step

        if self.translation_frame == "local":
            rot = Rotation.from_euler("xyz", self.orientation_rad, degrees=False).as_matrix()
            trans_step = rot.dot(trans_step)

        self.position[0] += float(trans_step[0])
        self.position[1] += float(trans_step[1])
        self.position[2] += float(trans_step[2])

        if self.if_limit_dof:
            self.check_limits_position()
        return self.position

    def update_gripper(self):
        # gripper
        joycon_button_gripper_open = (
            self.joycon.get_button_a() if self.joycon.is_right() else self.joycon.get_button_right()
        )
        joycon_button_gripper_close = (
            self.joycon.get_button_y() if self.joycon.is_right() else self.joycon.get_button_left()
        )
        if joycon_button_gripper_open == 1:
            self.gripper_state += 0.01 * self.gripper_speed
        if joycon_button_gripper_close == 1:
            self.gripper_state -= 0.01 * self.gripper_speed

        self.check_limits_gripper()
        return self.gripper_state

    def update_button_control(self):

        joycon_button_up = self.joycon.get_button_r()
        joycon_button_down = self.joycon.get_button_zr()
        if joycon_button_up == 1 and joycon_button_down == 1:
            self.button_control = 10

        # home
        joycon_button_home = (
            self.joycon.get_button_home() if self.joycon.is_right() else self.joycon.get_button_capture()
        )
        if joycon_button_home == 1:
            self.go_to_home()

        # calibrate yaw
        joycon_button_calibrate_yaw = (
            self.joycon.get_button_r_stick() if self.joycon.is_right() else self.joycon.get_button_l_stick()
        )
        if joycon_button_calibrate_yaw == 1:
            self.calibrate_yaw()

        return self.button_control

    def update_orientation(self):  # euler_rad, euler_deg, quaternion,
        self.orientation_rad = self.orientation_sensor.update(self.gyro.gyro_in_rad[0], self.gyro.accel_in_g[0])

        for i in range(3):  # deal with offset and reverse
            self.orientation_rad[i] = (self.orientation_rad[i] + self.home_euler_ard[i]) * self.euler_reverse[i]

        if self.if_limit_dof:
            self.check_limits_orientation()

        return self.orientation_rad

    def calibrate_yaw(self):
        if self.orientation_rad[2] > self.home_euler_ard[2] + 0.02:  # * self.dof_speed[5]:
            self.yaw_diff = self.yaw_diff + (0.01 * self.dof_speed[5])
        elif self.orientation_rad[2] < self.home_euler_ard[2] - 0.02:  # * self.dof_speed[5]:
            self.yaw_diff = self.yaw_diff - (0.01 * self.dof_speed[5])
        else:
            self.yaw_diff = self.yaw_diff

        # print(f'{self.yaw_diff=}')
        self.orientation_sensor.set_yaw_diff(self.yaw_diff)

        # print(f'{self.orientation_rad[2]=}')
        if self.orientation_rad[2] < (0.02 * self.dof_speed[5]) and self.orientation_rad[2] > (
            -0.02 * self.dof_speed[5]
        ):
            self.orientation_sensor.reset_yaw()  # gyro.reset_orientation()
            self.yaw_diff = 0.0
            self.orientation_sensor.set_yaw_diff(self.yaw_diff)

    def go_to_home(self):
        if self.position[0] > self.home_position[0] + 0.002:
            self.position[0] = self.position[0] - 0.001 * self.dof_speed[0] * 2.0
        elif self.position[0] < self.home_position[0] - 0.002:
            self.position[0] = self.position[0] + 0.001 * self.dof_speed[0] * 2.0
        else:
            self.position[0] = self.position[0]

        if self.position[1] > self.home_position[1] + 0.002:
            self.position[1] = self.position[1] - 0.001 * self.dof_speed[1] * 2.0
        elif self.position[1] < self.home_position[1] - 0.002:
            self.position[1] = self.position[1] + 0.001 * self.dof_speed[1] * 2.0
        else:
            self.position[1] = self.position[1]

        if self.position[2] > self.home_position[2] + 0.002:
            self.position[2] = self.position[2] - 0.001 * self.dof_speed[2] * 2.0
        elif self.position[2] < self.home_position[2] - 0.002:
            self.position[2] = self.position[2] + 0.001 * self.dof_speed[2] * 2.0
        else:
            self.position[2] = self.position[2]

        if self.if_limit_dof:
            self.check_limits_position()
        self.calibrate_yaw()

    def update(self):
        roll, pitch, yaw = self.update_orientation()
        self.position = self.update_translation()
        gripper = self.update_gripper()
        button_control = self.update_button_control()

        x, y, z = self.position
        self.posture = [x, y, z, roll, pitch, yaw]

        return self.posture, gripper, button_control

    def solve_loop(self):
        while self.running:
            try:
                self.update()
                # print("solve successful")
                time.sleep(0.01)
            except Exception as e:
                logging.error(f"Error solve_loop from device: {e}")
                time.sleep(1)  # Wait before retrying

    def get_control(self, out_format="euler_rad"):
        x, y, z = self.position
        if out_format == "euler_deg":
            roll, pitch, yaw = np.rad2deg(self.orientation_rad)
            self.posture = [x, y, z, roll, pitch, yaw]
        elif out_format == "quaternion":
            r4 = Rotation.from_euler("xyz", self.orientation_rad, degrees=False)
            qx, qy, qz, qw = r4.as_quat()
            self.posture = [x, y, z, qx, qy, qz, qw]
        else:
            roll, pitch, yaw = self.orientation_rad
            self.posture = [x, y, z, roll, pitch, yaw]

        return self.posture, self.gripper_state, self.button_control

    # More information
    def get_stick(self):
        stick_vertical = (
            self.joycon.get_stick_right_vertical() if self.joycon.is_right() else self.joycon.get_stick_left_vertical()
        )
        stick_horizontal = (
            self.joycon.get_stick_right_horizontal()
            if self.joycon.is_right()
            else self.joycon.get_stick_right_horizontal()
        )
        stick_button = self.joycon.get_button_r_stick() if self.joycon.is_right() else self.joycon.get_button_l_stick()

        return stick_vertical, stick_horizontal, stick_button

    def listen_button(self, button, show_all=False):
        # the button names:
        # right: r, zr, y, x, a, b, plus, r-stick, home, sr, sl
        # left: l, zl, left, up, right, down, minos, l-stick, capture, sr, sl

        # Use direct button state query instead of events to avoid competition with background thread
        button_mapping = {
            # Right Joy-Con buttons
            "r": lambda: self.joycon.get_button_r() if self.joycon.is_right() else 0,
            "zr": lambda: self.joycon.get_button_zr() if self.joycon.is_right() else 0,
            "y": lambda: self.joycon.get_button_y() if self.joycon.is_right() else 0,
            "x": lambda: self.joycon.get_button_x() if self.joycon.is_right() else 0,
            "a": lambda: self.joycon.get_button_a() if self.joycon.is_right() else 0,
            "b": lambda: self.joycon.get_button_b() if self.joycon.is_right() else 0,
            "plus": lambda: (self.joycon.get_button_plus() if self.joycon.is_right() else 0),
            "r-stick": lambda: (self.joycon.get_button_r_stick() if self.joycon.is_right() else 0),
            "home": lambda: (self.joycon.get_button_home() if self.joycon.is_right() else 0),
            # Left Joy-Con buttons
            "l": lambda: self.joycon.get_button_l() if self.joycon.is_left() else 0,
            "zl": lambda: self.joycon.get_button_zl() if self.joycon.is_left() else 0,
            "left": lambda: (self.joycon.get_button_left() if self.joycon.is_left() else 0),
            "up": lambda: self.joycon.get_button_up() if self.joycon.is_left() else 0,
            "right": lambda: (self.joycon.get_button_right() if self.joycon.is_left() else 0),
            "down": lambda: (self.joycon.get_button_down() if self.joycon.is_left() else 0),
            "minus": lambda: (self.joycon.get_button_minus() if self.joycon.is_left() else 0),
            "l-stick": lambda: (self.joycon.get_button_l_stick() if self.joycon.is_left() else 0),
            "capture": lambda: (self.joycon.get_button_capture() if self.joycon.is_left() else 0),
        }

        if show_all:
            # Show all button states
            for btn_name, btn_func in button_mapping.items():
                status = btn_func()
                if status:
                    print(f"{btn_name}: {status}")

        if button in button_mapping:
            return button_mapping[button]()
        else:
            print(f"Unknown button: {button}")
            return 0

    def set_position(self, set_position):
        # self.x, self.y, self.z = set_position
        self.position = set_position
        return

    def close_horizontal_stick(self):
        # mark horizontal stick as closed
        self._horizontal_stick_closed = True
        return

    def close_y(self):
        # disable Y movement coupling
        self.if_close_y = True
        return

    def open_horizontal(self):
        # mark horizontal stick as opened
        self._horizontal_stick_closed = False
        return

    def open_gripper(self):
        self.gripper_state = max(self.gripper_limit)
        return

    def close_gripper(self):
        self.gripper_state = min(self.gripper_limit)
        return

    def set_posture_limits(self, glimit):
        # glimit = [[x_min, y_min, z_min, roll_min, pitch_min, yaw_min]
        #           [x_max, y_max, z_max, roll_max, pitch_max, yaw_max]]
        # such as glimit = [[0.000, -0.4,  0.046, -3.1, -1.5, -1.5],
        #                   [0.430,  0.4,  0.23,  3.1,  1.5,  1.5]]
        self.glimit = glimit
        return

    def set_dof_speed(self, dof_speed):
        # glimit = [x_speed, y_speed, z_speed, _, _, yaw_speed]
        self.dof_speed = dof_speed
        return
