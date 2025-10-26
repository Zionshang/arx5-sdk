#include "app/joint_controller.h"
#include "hardware/can_hardware.h"
#include <chrono>
#include <csignal>
#include <unistd.h>

using namespace arx;

RobotConfig g_robot_config = RobotConfigFactory::get_instance().get_config("L5");
ControllerConfig g_controller_config =
    ControllerConfigFactory::get_instance().get_config("joint_controller", g_robot_config.joint_dof);
std::shared_ptr<IHardwareInterface> g_hw = std::make_shared<CanHardware>("can0", g_robot_config);
Arx5JointController *arx5_joint_controller = new Arx5JointController(g_robot_config, g_controller_config, g_hw);

void signal_handler(int signal)
{
    std::cout << "SIGINT received" << std::endl;
    delete arx5_joint_controller;
    exit(signal);
}

int main()
{
    std::signal(SIGINT, signal_handler);
    int loop_cnt = 0;
    while (true)
    {
        JointState state = arx5_joint_controller->get_joint_state();
        EEFState eef_state = arx5_joint_controller->get_eef_state();
        Pose6d pose6 = eef_state.pose_6d;
        std::cout << "Gripper: " << state.gripper_pos << ", joint pos: " << state.pos.transpose()
                  << ", Pose: " << pose6.transpose() << std::endl;
        usleep(10000); // 10ms
        loop_cnt++;
        if (loop_cnt % 500 == 0)
        {
            arx5_joint_controller->reset_to_home();
            arx5_joint_controller->set_to_damping();
        }
    }
    return 0;
}