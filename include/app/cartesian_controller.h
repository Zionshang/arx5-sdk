#ifndef CARTESIAN_CONTROLLER_H
#define CARTESIAN_CONTROLLER_H

#include "app/common.h"
#include "app/config.h"
#include "app/controller_base.h"
#include "app/solver.h"
#include "hardware/hardware_interface.h"
#include "utils.h"

#include <chrono>
#include <memory>
#include <mutex>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h>

namespace arx
{
class Arx5CartesianController : public Arx5ControllerBase
{
  public:
    // New constructors that take an injected hardware interface
    Arx5CartesianController(RobotConfig robot_config, ControllerConfig controller_config,
                            std::shared_ptr<IHardwareInterface> hw);
    Arx5CartesianController(std::string model, std::shared_ptr<IHardwareInterface> hw);

    void set_eef_cmd(EEFState new_cmd);
    void set_eef_traj(std::vector<EEFState> new_traj);
    EEFState get_eef_cmd();

    std::tuple<int, Eigen::VectorXd> multi_trial_ik(Eigen::Matrix<double, 6, 1> target_pose_6d,
                                                    Eigen::VectorXd current_joint_pos, int additional_trial_num = 5);
};
} // namespace arx

#endif // CARTESIAN_CONTROLLER