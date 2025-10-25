import time
from joyconrobotics import JoyconRobotics
from visualizer import PoseVisualizer

joyconrobotics_right = JoyconRobotics(
    device="right",
    translation_frame="local",
    direction_reverse=[1, 1, 1],
    euler_reverse=[-1, -1, 1],
    limit_dof=True,
    glimit=[[0.0, -0.5, -0.5, -1.3, -1.3, -1.3], [0.5, 0.5, 0.5, 1.3, 1.3, 1.3]],
    gripper_limit=[0.0, 0.8],
)

viz = PoseVisualizer(
    axis_len=0.3,
    world_axis_len=0.2,
    window_title="Joycon Pose",
    orientation_format="euler",
)

while True:  # continuously get data until user interrupts
    pose, gripper, control_button = joyconrobotics_right.get_control()
    if control_button == 10:
        break
    viz.update(pose)
    joyconrobotics_right.listen_button("x", True)
    time.sleep(0.01)

viz.close()
joyconrobotics_right.disconnect()
