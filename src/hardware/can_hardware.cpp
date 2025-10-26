#include "hardware/can_hardware.h"
#include <cassert>

namespace arx
{

CanHardware::CanHardware(const std::string &interface_name, const RobotConfig &robot_config)
    : can_(interface_name), robot_config_(robot_config)
{
}

JointState CanHardware::read_state()
{
    JointState joint_state{robot_config_.joint_dof};
    std::array<OD_Motor_Msg, 10> motor_msg = can_.get_motor_msg();

    // joints
    for (int i = 0; i < robot_config_.joint_dof; ++i)
    {
        const int mid = robot_config_.motor_id[i];
        joint_state.pos[i] = motor_msg[mid].angle_actual_rad;
        joint_state.vel[i] = motor_msg[mid].speed_actual_rad;
        // current(A) -> torque(Nm)
        if (robot_config_.motor_type[i] == MotorType::EC_A4310)
        {
            joint_state.torque[i] =
                motor_msg[mid].current_actual_float * kTorqueConst_EC_A4310 * kTorqueConst_EC_A4310; // 保持原逻辑
        }
        else if (robot_config_.motor_type[i] == MotorType::DM_J4310)
        {
            joint_state.torque[i] = motor_msg[mid].current_actual_float * kTorqueConst_DM_J4310;
        }
        else if (robot_config_.motor_type[i] == MotorType::DM_J4340)
        {
            joint_state.torque[i] = motor_msg[mid].current_actual_float * kTorqueConst_DM_J4340;
        }
        else
        {
            // 未知类型，按0处理
            joint_state.torque[i] = 0.0;
        }
    }

    // gripper: 将电机读数映射为开口宽度（m）
    {
        const int gid = robot_config_.gripper_motor_id;
        const double open_readout = robot_config_.gripper_open_readout; // 电机角度（rad）
        const double width = robot_config_.gripper_width;               // m
        joint_state.gripper_pos = motor_msg[gid].angle_actual_rad / open_readout * width;
        joint_state.gripper_vel = motor_msg[gid].speed_actual_rad / open_readout * width;
        joint_state.gripper_torque =
            motor_msg[gid].current_actual_float * kTorqueConst_DM_J4310; // gripper 默认 DM_J4310
    }

    // timestamp 由控制器层填充（更统一）
    return joint_state;
}

void CanHardware::send_joint_command(int joint_index, float kp, float kd, float pos, float vel, float torque)
{
    assert(joint_index >= 0 && joint_index < robot_config_.joint_dof);
    const int mid = robot_config_.motor_id[joint_index];
    if (robot_config_.motor_type[joint_index] == MotorType::EC_A4310)
    {
        can_.send_EC_motor_cmd(mid, kp, kd, pos, vel, torque / kTorqueConst_EC_A4310);
    }
    else if (robot_config_.motor_type[joint_index] == MotorType::DM_J4310)
    {

        can_.send_DM_motor_cmd(mid, kp, kd, pos, vel, torque / kTorqueConst_DM_J4310);
    }
    else if (robot_config_.motor_type[joint_index] == MotorType::DM_J4340)
    {
        can_.send_DM_motor_cmd(mid, kp, kd, pos, vel, torque / kTorqueConst_DM_J4340);
    }
    else
    {
        // 不支持的类型，忽略
    }
}

void CanHardware::send_gripper_command(float kp, float kd, float pos, float vel, float torque)
{
    // 将开口宽度(m)映射为电机角度（rad）
    const int gid = robot_config_.gripper_motor_id;
    const double open_readout = robot_config_.gripper_open_readout;
    const double width = robot_config_.gripper_width;
    const double motor_pos = (width > 0 ? (pos / width * open_readout) : 0.0);
    const double motor_vel = (width > 0 ? (vel / width * open_readout) : 0.0);

    float current = 0.0f;
    // 夹爪电机类型目前配置为 DM_J4310（原实现如此），如后续支持其他类型，在此扩展
    if (robot_config_.gripper_motor_type == MotorType::DM_J4310)
    {
        current = static_cast<float>(torque / kTorqueConst_DM_J4310);
        can_.send_DM_motor_cmd(gid, kp, kd, motor_pos, motor_vel, current);
    }
    else if (robot_config_.gripper_motor_type == MotorType::DM_J4340)
    {
        current = static_cast<float>(torque / kTorqueConst_DM_J4340);
        can_.send_DM_motor_cmd(gid, kp, kd, motor_pos, motor_vel, current);
    }
    else if (robot_config_.gripper_motor_type == MotorType::EC_A4310)
    {
        current = static_cast<float>(torque / kTorqueConst_EC_A4310);
        can_.send_EC_motor_cmd(gid, kp, kd, motor_pos, motor_vel, current);
    }
    else
    {
        // 不支持的类型，忽略
    }
}

void CanHardware::enable_joint(int joint_index)
{
    const auto mt = robot_config_.motor_type[joint_index];
    if (mt == MotorType::DM_J4310 || mt == MotorType::DM_J4340 || mt == MotorType::DM_J8009)
    {
        can_.enable_DM_motor(robot_config_.motor_id[joint_index]);
    }
    else if (mt == MotorType::EC_A4310)
    {
        // EC 电机此处默认不需要 enable；如需要可扩展
    }
}

void CanHardware::enable_gripper()
{
    if (robot_config_.gripper_motor_type == MotorType::DM_J4310 ||
        robot_config_.gripper_motor_type == MotorType::DM_J4340 ||
        robot_config_.gripper_motor_type == MotorType::DM_J8009)
    {
        can_.enable_DM_motor(robot_config_.gripper_motor_id);
    }
    else if (robot_config_.gripper_motor_type == MotorType::EC_A4310)
    {
        // EC 电机此处默认不需要 enable；如需要可扩展
    }
}

void CanHardware::set_zero_at_current_joint(int joint_index)
{
    const int mid = robot_config_.motor_id[joint_index];
    if (robot_config_.motor_type[joint_index] == MotorType::EC_A4310)
    {
        can_.can_cmd_init(mid, 0x03);
    }
    else
    {
        can_.reset_zero_readout(mid);
    }
}

void CanHardware::set_zero_at_current_gripper()
{
    const int gid = robot_config_.gripper_motor_id;
    if (robot_config_.gripper_motor_type == MotorType::EC_A4310)
    {
        can_.can_cmd_init(gid, 0x03);
    }
    else
    {
        can_.reset_zero_readout(gid);
    }
}

} // namespace arx
