#ifndef ARX5_CAN_HARDWARE_H
#define ARX5_CAN_HARDWARE_H

#include "app/config.h"
#include "hardware/arx_can.h"
#include "hardware/hardware_interface.h"
#include <memory>

namespace arx
{

class CanHardware : public IHardwareInterface
{
  public:
    CanHardware(const std::string &interface_name, const RobotConfig &robot_config);
    ~CanHardware() override = default;

    JointState read_state() override;

    void send_joint_command(int joint_index, float kp, float kd, float pos, float vel, float torque) override;
    void send_gripper_command(float kp, float kd, float pos, float vel, float torque) override;

    void enable_joint(int joint_index) override;
    void enable_gripper() override;

    void set_zero_at_current_joint(int joint_index) override;
    void set_zero_at_current_gripper() override;

  private:
    ArxCan can_;
    RobotConfig robot_config_;

    // 力矩常数（Nm/A）
    static constexpr double kTorqueConst_EC_A4310 = 1.4;
    static constexpr double kTorqueConst_DM_J4310 = 0.424;
    static constexpr double kTorqueConst_DM_J4340 = 1.0;
};

} // namespace arx

#endif // ARX5_CAN_HARDWARE_H
