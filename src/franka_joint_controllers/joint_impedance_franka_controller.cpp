// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
// #include <franka_interactive_controllers/joint_impedance_example_controller.h>
#include <joint_impedance_franka_controller.h>

#include <array>
#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <Eigen/Dense>

#include <franka/robot_state.h>

namespace franka_interactive_controllers {

bool JointImpedanceFrankaController::init(hardware_interface::RobotHW* robot_hw,
                                           ros::NodeHandle& node_handle) {

  sub_desired_pose_ = node_handle.subscribe(
    "/joint_impedance_controller/desired_pose", 20, &JointImpedanceFrankaController::desiredPoseCallback, this,
    ros::TransportHints().reliable().tcpNoDelay());

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("JointImpedanceFrankaController: Could not read parameter arm_id");
    return false;
  }
  if (!node_handle.getParam("radius", radius_)) {
    ROS_INFO_STREAM(
        "JointImpedanceFrankaController: No parameter radius, defaulting to: " << radius_);
  }
  // if (std::fabs(radius_) < 0.005) {
  //   ROS_INFO_STREAM("JointImpedanceFrankaController: Set radius to small, defaulting to: " << 0.1);
  //   radius_ = 0.1;
  // }

  if (!node_handle.getParam("vel_max", vel_max_)) {
    ROS_INFO_STREAM(
        "JointImpedanceFrankaController: No parameter vel_max, defaulting to: " << vel_max_);
  }
  if (!node_handle.getParam("acceleration_time", acceleration_time_)) {
    ROS_INFO_STREAM(
        "JointImpedanceFrankaController: No parameter acceleration_time, defaulting to: "
        << acceleration_time_);
  }

  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "JointImpedanceFrankaController: Invalid or no joint_names parameters provided, aborting "
        "controller init!");
    return false;
  }

  if (!node_handle.getParam("k_gains", k_gains_) || k_gains_.size() != 7) {
    ROS_ERROR(
        "JointImpedanceFrankaController:  Invalid or no k_gain parameters provided, aborting "
        "controller init!");
    return false;
  }

  if (!node_handle.getParam("d_gains", d_gains_) || d_gains_.size() != 7) {
    ROS_ERROR(
        "JointImpedanceFrankaController:  Invalid or no d_gain parameters provided, aborting "
        "controller init!");
    return false;
  }

  double publish_rate(30.0);
  if (!node_handle.getParam("publish_rate", publish_rate)) {
    ROS_INFO_STREAM("JointImpedanceFrankaController: publish_rate not found. Defaulting to "
                    << publish_rate);
  }
  rate_trigger_ = franka_hw::TriggerRate(publish_rate);

  if (!node_handle.getParam("coriolis_factor", coriolis_factor_)) {
    ROS_INFO_STREAM("JointImpedanceFrankaController: coriolis_factor not found. Defaulting to "
                    << coriolis_factor_);
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "JointImpedanceFrankaController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "JointImpedanceFrankaController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  // auto* cartesian_pose_interface = robot_hw->get<franka_hw::FrankaPoseCartesianInterface>();
  // if (cartesian_pose_interface == nullptr) {
  //   ROS_ERROR_STREAM(
  //       "JointImpedanceFrankaController: Error getting cartesian pose interface from hardware");
  //   return false;
  // }
  // try {
  //   cartesian_pose_handle_ = std::make_unique<franka_hw::FrankaCartesianPoseHandle>(
  //       cartesian_pose_interface->getHandle(arm_id + "_robot"));
  // } catch (hardware_interface::HardwareInterfaceException& ex) {
  //   ROS_ERROR_STREAM(
  //       "JointImpedanceFrankaController: Exception getting cartesian pose handle from interface: "
  //       << ex.what());
  //   return false;
  // }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "JointImpedanceFrankaController: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "JointImpedanceFrankaController: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }


  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "JointImpedanceFrankaController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "JointImpedanceFrankaController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  std::fill(dq_filtered_.begin(), dq_filtered_.end(), 0);

  _panda_ik_service = franka_interactive_controllers::PandaTracIK();
  position_d_.setZero();
  orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  position_d_target_.setZero();
  orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;

  return true;
}

void JointImpedanceFrankaController::starting(const ros::Time& /*time*/) {


  // Get robot current/initial joint state
  franka::RobotState initial_state = state_handle_->getRobotState();
  // for (size_t i = 0; i < 7; i++) {
  //   ik_joint_targets_[i] = initial_state.q[i];
  // }

  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());

  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));

  // set desired point to current state
  position_d_           = initial_transform.translation();
  orientation_d_        = Eigen::Quaterniond(initial_transform.linear());
  position_d_target_    = initial_transform.translation();
  orientation_d_target_ = Eigen::Quaterniond(initial_transform.linear());


  for (size_t i = 0; i < 7; ++i) {
    q_[i] = joint_handles_[i].getPosition();
    q_d_[i] = joint_handles_[i].getPosition();
    target_q_d_[i] = joint_handles_[i].getPosition();
    dq_[i] = joint_handles_[i].getVelocity();
    dq_d_[i] = joint_handles_[i].getVelocity();
  }


  // target_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_d;

}

void JointImpedanceFrankaController::update(const ros::Time& /*time*/,
                                             const ros::Duration& period) {
  
  // cartesian_pose_handle_->setCommand(target_pose_d_);

  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis = model_handle_->getCoriolis();
  std::array<double, 7> gravity = model_handle_->getGravity();
  std::array<double, 7> current_tau;


  for (size_t i=0; i<7; i++){
    q_[i] = joint_handles_[i].getPosition();
    dq_[i] = joint_handles_[i].getVelocity();
  }

  for (size_t i=0; i<7; i++){
    dq_d_[i] = 0.0;
    current_tau[i] = joint_handles_[i].getEffort();
  }

  std::array<double, 7> tau_d_calculated;
  for (size_t i = 0; i < 7; i++) {
    tau_d_calculated[i] = coriolis_factor_ * coriolis[i] +
                          k_gains_[i] * (q_d_[i] - q_[i]) +
                          d_gains_[i] * (dq_d_[i] - dq_[i]);
  }

  std::array<double, 7> tau_d_saturated = saturateTorqueRate(tau_d_calculated, robot_state.tau_J_d);

  for (size_t i=0; i<7; i++){
    q_d_[i] = target_q_d_[i] * q_filt_ + q_d_[i] * (1 - q_filt_);
  }

  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(tau_d_saturated[i]);
  }

  // double alpha = 0.99;
  // for (size_t i = 0; i < 7; i++) {
  //   dq_filtered_[i] = (1 - alpha) * dq_filtered_[i] + alpha * robot_state.dq[i];
  // }

  // std::array<double, 7> desired_q{};
  // for (size_t i = 0; i < 7; i++) {
  //   desired_q[i] = ik_joint_targets_[i];  // Always use the last solved IK
  // }

  // // best and most recommended method for trajectory computation after
  // // inverse kinematics calculations
  // // normalize position, we need to do this separately for every joint
  // double current_pos, p_val;
  // for (int i=0; i<7; i++)
  // {
  //     current_pos = _position_joint_handles[i].getPosition();
  //     // norm position
  //     p_val = 2 - (2 * (abs(_joint_cmds[i] - current_pos) / abs(calc_max_pos_diffs[i])));
  //     // if p val is negative, treat it as 0
  //     p_val = std::max(p_val, 0.);
  //     catmullRomSplineVelCmd(p_val, i, interval_length);
  // }

  // geometry_msgs::Pose target_msg;
  // target_msg.position.x = position_d_target_.x();
  // target_msg.position.y = position_d_target_.y();
  // target_msg.position.z = position_d_target_.z();
  // target_msg.orientation.x = orientation_d_target_.x();
  // target_msg.orientation.y = orientation_d_target_.y();
  // target_msg.orientation.z = orientation_d_target_.z();
  // target_msg.orientation.w = orientation_d_target_.w();

  // KDL::JntArray ik_result = _panda_ik_service.perform_ik(target_msg);

  // if (_panda_ik_service.is_valid && ik_result.rows() == 7) {
  //   double alpha_q = 0.05;  // blending factor for smoothness
  //   for (size_t i = 0; i < 7; i++) {
  //     ik_joint_targets_[i] =
  //         alpha_q * ik_result(i) + (1.0 - alpha_q) * ik_joint_targets_[i];
  //   }
  // } else {
  //   ROS_WARN_THROTTLE(1.0, "IK failed to find a solution, keeping last valid joint targets.");
  // }

  // std::array<double, 7> tau_d_calculated;
  // for (size_t i = 0; i < 7; ++i) {
  //   tau_d_calculated[i] = coriolis_factor_ * coriolis[i] +
  //                         k_gains_[i] * (ik_joint_targets_[i] - robot_state.q[i]) +
  //                         d_gains_[i] * (0.0 - dq_filtered_[i]);
  // }


  // // Maximum torque difference with a sampling rate of 1 kHz. The maximum torque rate is
  // // 1000 * (1 / sampling_time).
  // std::array<double, 7> tau_d_saturated = saturateTorqueRate(tau_d_calculated, robot_state.tau_J_d);

  // for (size_t i = 0; i < 7; ++i) {
  //   joint_handles_[i].setCommand(tau_d_saturated[i]);
  // }

  // for (size_t i = 0; i < 7; ++i) {
  //   last_tau_d_[i] = tau_d_saturated[i] + gravity[i];
  // }

  // Eigen::Matrix4d pose_matrix = Eigen::Map<const Eigen::Matrix4d>(target_pose_.data());
  // ROS_INFO_STREAM(pose_matrix);

}

std::array<double, 7> JointImpedanceFrankaController::saturateTorqueRate(
    const std::array<double, 7>& tau_d_calculated,
    const std::array<double, 7>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  std::array<double, 7> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] = tau_J_d[i] + std::max(std::min(difference, kDeltaTauMax), -kDeltaTauMax);
  }
  return tau_d_saturated;
}

void JointImpedanceFrankaController::desiredPoseCallback(const geometry_msgs::Pose& msg) {

  std::lock_guard<std::mutex> position_d_target_mutex_lock(position_and_orientation_d_target_mutex_);

  position_d_target_ << msg.position.x, msg.position.y, msg.position.z;

  Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);

  orientation_d_target_.coeffs() << msg.orientation.x,
                                    msg.orientation.y,
                                    msg.orientation.z,
                                    msg.orientation.w;

  // Handle quaternion sign consistency to avoid jumps
  if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
    orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
  }

  KDL::JntArray ik_result = _panda_ik_service.perform_ik(msg);

  if (_panda_ik_service.is_valid && ik_result.rows() == 7) {
    // ROS_INFO("IK valid, updating joint targets");
    for (size_t i = 0; i < 7; i++) {
      target_q_d_[i] = ik_result(i);
    }
  } else {
    ROS_WARN("IK failed to find a solution");
  }
}

}  // namespace franka_interactive_controllers

PLUGINLIB_EXPORT_CLASS(franka_interactive_controllers::JointImpedanceFrankaController,
                       controller_interface::ControllerBase)
