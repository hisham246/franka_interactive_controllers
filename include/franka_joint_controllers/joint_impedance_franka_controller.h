// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#pragma once

#include <pinocchio/fwd.hpp>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <realtime_tools/realtime_publisher.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <std_msgs/Float64MultiArray.h>
#include <franka_utils/panda_trac_ik.h>
#include <Eigen/Dense>
  #include <sensor_msgs/JointState.h>

#include <franka_hw/franka_cartesian_command_interface.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <franka_hw/trigger_rate.h>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/spatial/explog.hpp>

namespace franka_interactive_controllers {

class JointImpedanceFrankaController : public controller_interface::MultiInterfaceController<
                                            franka_hw::FrankaModelInterface,
                                            hardware_interface::EffortJointInterface,
                                            franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  // Saturation
  std::array<double, 7> saturateTorqueRate(
      const std::array<double, 7>& tau_d_calculated,
      const std::array<double, 7>& tau_J_d);  // NOLINT (readability-identifier-naming)

  // std::unique_ptr<franka_hw::FrankaCartesianPoseHandle> cartesian_pose_handle_;
  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

  static constexpr double kDeltaTauMax{1.0};
  double radius_{0.1};
  double acceleration_time_{2.0};
  double vel_max_{0.05};
  double angle_{0.0};
  double vel_current_{0.0};
  double alpha_q_{0.2};
  std::array<double, 7> q_d_;
  std::array<double, 7> q_;
  std::array<double, 7> target_q_d_;
  std::array<double, 7> dq_d_;
  std::array<double, 7> dq_;
  double q_filt_ {0.4};  // amount off target to add to desired
  double epsilon_ = 0.0001;
  double max_abs_vel_ = 2.1;

  ros::Time start_time_;

  std::array<double, 7> limited_joint_cmds_;
  std::array<double, 7> last_commanded_pos_;
  std::array<double, 7> iters_;
  std::array<double, 7> joint_cmds_;

  Eigen::Vector3d position_d_;
  Eigen::Quaterniond orientation_d_;
  Eigen::Vector3d position_d_target_;
  Eigen::Quaterniond orientation_d_target_;
  std::mutex position_and_orientation_d_target_mutex_;

  // 7 by 4 matrix for coefficients for each franka panda joint
  std::array<std::array<double, 4>, 7> vel_catmull_coeffs_first_spline_;
  std::array<std::array<double, 4>, 7> vel_catmull_coeffs_second_spline_;
  std::array<double, 7> calc_max_pos_diffs;

  std::vector<double> k_gains_;
  std::vector<double> d_gains_;
  double coriolis_factor_{1.0};
  std::array<double, 7> q_filtered_;
  std::array<double, 7> dq_filtered_;
  // std::array<double, 16> initial_pose_;
  // std::array<double, 16> target_pose_;
  geometry_msgs::Pose target_pose_;

  franka_hw::TriggerRate rate_trigger_{1.0};
  std::array<double, 7> last_tau_d_{};

  // Desired pose subscriber
  ros::Subscriber sub_desired_pose_;
  ros::Subscriber sub_joint_positions_;
  void desiredPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  void jointPositionsCallback(const std_msgs::Float64MultiArray& msg);
  std::array<double, 7> runJointPositionController();

  // IK Integration
  franka_interactive_controllers::PandaTracIK panda_ik_service_;
  KDL::JntArray joints_result_;
  std::array<double, 7> ik_joint_targets_{};
  bool is_executing_cmd_ = false;

  std::array<double, 4> catmullRomSpline(const double &tau, const std::array<double,4> &points);
  double calcSplinePolynomial(const std::array<double,4> &coeffs, const double &x);
  void catmullRomSplineVelCmd(const double &norm_pos, const int &joint_num, const double &interval);
  bool isGoalReached();

  // Pinocchio
  pinocchio::Model pinocchio_model_;
  pinocchio::Data pinocchio_data_;
  pinocchio::FrameIndex ee_frame_id_;


  ros::Publisher desired_joints_pub_;

};

}  // namespace franka_interactive_controllers
