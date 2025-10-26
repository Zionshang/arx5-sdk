#ifndef ARX5_HARDWARE_INTERFACE_H
#define ARX5_HARDWARE_INTERFACE_H

#include "app/common.h"
#include "app/config.h"
#include <array>
#include <memory>

namespace arx
{

// 通用硬件抽象接口：不暴露 EC/DM 或底层 motor_msg，全部以物理量表示
class IHardwareInterface
{
  public:
    virtual ~IHardwareInterface() = default;

    // 读取完整关节+夹爪状态（单位：rad, rad/s, Nm；夹爪：m, m/s, Nm）
    virtual JointState read_state() = 0;

    // 发送单个关节命令（单位：rad, rad/s, Nm）
    virtual void send_joint_command(int joint_index, float kp, float kd, float pos, float vel, float torque) = 0;

    // 发送夹爪命令（单位：m, m/s, Nm；内部自行做开口宽度<->电机角度映射）
    virtual void send_gripper_command(float kp, float kd, float pos, float vel, float torque) = 0;

    // 设备控制（与底层协议无关的通用语义）
    virtual void enable_joint(int joint_index) = 0;
    virtual void enable_gripper() = 0;

    // 在当前位置设置零点
    virtual void set_zero_at_current_joint(int joint_index) = 0;
    virtual void set_zero_at_current_gripper() = 0;
};

} // namespace arx

#endif // ARX5_HARDWARE_INTERFACE_H
