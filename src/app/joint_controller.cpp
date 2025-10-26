#include "app/joint_controller.h"
#include "utils.h"
#include <array>
#include <stdexcept>
#include <sys/syscall.h>
#include <sys/types.h>
using namespace arx;

Arx5JointController::Arx5JointController(RobotConfig robot_config, ControllerConfig controller_config,
                                         std::shared_ptr<IHardwareInterface> hw)
    : Arx5ControllerBase(robot_config, controller_config, std::move(hw))
{
}

Arx5JointController::Arx5JointController(std::string model, std::shared_ptr<IHardwareInterface> hw)
    : Arx5JointController::Arx5JointController(
          RobotConfigFactory::get_instance().get_config(model),
          ControllerConfigFactory::get_instance().get_config(
              "joint_controller", RobotConfigFactory::get_instance().get_config(model).joint_dof),
          std::move(hw))
{
}

void Arx5JointController::set_joint_cmd(JointState new_cmd)
{
    JointState current_joint_state = get_joint_state();
    double current_time = get_timestamp();
    if (new_cmd.timestamp == 0)
        new_cmd.timestamp = current_time + controller_config_.default_preview_time;

    std::lock_guard<std::mutex> guard(cmd_mutex_);
    if (abs(new_cmd.timestamp - current_time) < 1e-3)
        // If the new timestamp is close enough (<1ms) to the current time
        // Will override the entire interpolator object
        interpolator_.init_fixed(new_cmd);
    else
        interpolator_.override_waypoint(get_timestamp(), new_cmd);
}

void Arx5JointController::set_joint_traj(std::vector<JointState> new_traj)
{
    double start_time = get_timestamp();
    std::vector<JointState> joint_traj;
    double avg_window_s = 0.05;
    joint_traj.push_back(interpolator_.interpolate(start_time - 2 * avg_window_s));
    joint_traj.push_back(interpolator_.interpolate(start_time - avg_window_s));
    joint_traj.push_back(interpolator_.interpolate(start_time));

    double prev_timestamp = 0;
    for (auto joint_state : new_traj)
    {
        if (joint_state.timestamp <= start_time)
            continue;
        if (joint_state.timestamp == 0)
            throw std::invalid_argument("JointState timestamp must be set for all waypoints");
        if (joint_state.timestamp <= prev_timestamp)
            throw std::invalid_argument("JointState timestamps must be in ascending order");
        joint_traj.push_back(joint_state);
        prev_timestamp = joint_state.timestamp;
    }
    calc_joint_vel(joint_traj, avg_window_s);

    std::lock_guard<std::mutex> guard(cmd_mutex_);
    interpolator_.override_traj(get_timestamp(), joint_traj);
}

void Arx5JointController::recv_once()
{
    if (background_send_recv_running_)
    {
        logger_->warn("send_recv task is already running in background. recv_once() is ignored.");
        return;
    }
    recv_();
}

void Arx5JointController::send_recv_once()
{
    if (background_send_recv_running_)
    {
        logger_->warn("send_recv task is already running in background. send_recv_once is ignored.");
        return;
    }
    check_joint_state_sanity_();
    over_current_protection_();
    send_recv_();
}

void Arx5JointController::calibrate_gripper()
{
    bool prev_running = background_send_recv_running_;
    background_send_recv_running_ = false;
    sleep_us(1000);
    for (int i = 0; i < 10; ++i)
    {
        hw_->send_gripper_command(0, 0, 0, 0, 0);
        usleep(400);
    }
    logger_->info("Start calibrating gripper. Please fully close the gripper and press "
                  "enter to continue");
    std::cin.get();
    hw_->set_zero_at_current_gripper();
    usleep(400);
    for (int i = 0; i < 10; ++i)
    {
        hw_->send_gripper_command(0, 0, 0, 0, 0);
        usleep(400);
    }
    usleep(400);
    logger_->info("Finish setting zero point. Please fully open the gripper and press "
                  "enter to continue");
    std::cin.get();

    for (int i = 0; i < 10; ++i)
    {
        hw_->send_gripper_command(0, 0, 0, 0, 0);
        usleep(400);
    }
    // 抽象后不再直接读取原始电机读数；如需更新 gripper_open_readout，建议在具体硬件实现中提供日志或工具。
    if (prev_running)
    {
        background_send_recv_running_ = true;
    }
}

void Arx5JointController::calibrate_joint(int joint_id)
{
    bool prev_running = background_send_recv_running_;
    background_send_recv_running_ = false;
    sleep_us(1000);
    int motor_id = robot_config_.motor_id[joint_id];
    for (int i = 0; i < 10; ++i)
    {
        hw_->send_joint_command(joint_id, 0, 0, 0, 0, 0);
        usleep(400);
    }
    logger_->info("Start calibrating joint {}. Please move the joint to the home position and press enter to continue",
                  joint_id);
    std::cin.get();
    hw_->set_zero_at_current_joint(joint_id);
    usleep(400);
    for (int i = 0; i < 10; ++i)
    {
        hw_->send_joint_command(joint_id, 0, 0, 0, 0, 0);
        usleep(400);
    }
    usleep(400);
    logger_->info("Finish setting zero point for joint {}.", joint_id);
    if (prev_running)
    {
        background_send_recv_running_ = true;
    }
}
