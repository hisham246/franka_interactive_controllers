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
#include <franka_utils/franka_ik_He.hpp>

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

  panda_ik_service_ = franka_interactive_controllers::PandaTracIK();
  is_executing_cmd_ = false;

  return true;
}

void JointImpedanceFrankaController::starting(const ros::Time& /*time*/) {

  for (size_t i = 0; i < 7; ++i) {
    q_[i] = joint_handles_[i].getPosition();
    q_d_[i] = joint_handles_[i].getPosition();
    target_q_d_[i] = joint_handles_[i].getPosition();
    dq_[i] = joint_handles_[i].getVelocity();
    dq_d_[i] = joint_handles_[i].getVelocity();
  }
}

void JointImpedanceFrankaController::update(const ros::Time& /*time*/,
                                             const ros::Duration& period) {

  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis = model_handle_->getCoriolis();

  // Get current position and velocity of joints
  for (size_t i=0; i<7; i++){
    q_[i] = joint_handles_[i].getPosition();
    dq_[i] = joint_handles_[i].getVelocity();
  }
  
  // Get desired velocity based on desired joint poses and current joint torques for saturation
  for (size_t i=0; i<7; i++){
    dq_d_[i] = 0.0;
  }
  
  std::array<double, 7> tau_d_calculated;
  for (size_t i = 0; i < 7; i++) {
    tau_d_calculated[i] = coriolis_factor_ * coriolis[i] +
                          k_gains_[i] * (q_d_[i] - q_[i]) +
                          d_gains_[i] * (dq_d_[i] - dq_[i]);

  }

  // Maximum torque difference with a sampling rate of 1 kHz. The maximum torque rate is
  // 1000 * (1 / sampling_time).
  // std::array<double, 7> tau_d_saturated = saturateTorqueRate(tau_d_calculated, current_tau);
  // std::array<double, 7> tau_d_saturated = tau_d_calculated;
  // std::array<double, 7> tau_d_saturated = saturateTorqueRate(tau_d_calculated, robot_state.tau_J_d);


  for (size_t i=0; i<7; i++){
    q_d_[i] = target_q_d_[i] * q_filt_ + q_d_[i] * (1-q_filt_);
  }
    
  for (size_t i = 0; i < 7; i++) {
    joint_handles_[i].setCommand(tau_d_calculated[i]);
  }

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

void JointImpedanceFrankaController::desiredPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg)
{

  std::lock_guard<std::mutex> position_d_target_mutex_lock(position_and_orientation_d_target_mutex_);
  position_d_target_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  // ROS_INFO_STREAM("[CALLBACK] Desired ee position from DS: " << position_d_target_);
  
  Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
  orientation_d_target_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
      msg->pose.orientation.z, msg->pose.orientation.w;
  
  if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
    orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
  }

  ROS_INFO_STREAM("JointImpedanceFrankaController: Received desired pose for impedance controller.");

  // // Convert received pose to std::array<double,16>
  // if (msg.data.size() != 16) {
  //   ROS_ERROR("Received invalid pose: expected 16 elements for a 4x4 transformation matrix.");
  //   return;
  // }

  // std::array<double, 16> O_T_EE_array;
  // for (size_t i = 0; i < 16; i++) {
  //   O_T_EE_array[i] = msg.data[i];
  // }

  // 1) Convert PoseStamped to a 4x4 transformation matrix (Eigen)
  Eigen::Matrix3d R = orientation_d_target_.toRotationMatrix();
  Eigen::Vector3d p(position_d_target_);

  Eigen::Matrix4d O_T_EE;
  O_T_EE.setIdentity();
  O_T_EE.topLeftCorner<3, 3>() = R;
  O_T_EE.topRightCorner<3, 1>() = p;

  // Convert Eigen matrix to column-major std::array<double,16> for IK
  std::array<double, 16> O_T_EE_array;
  Eigen::Map<Eigen::Matrix<double, 4, 4>>(O_T_EE_array.data()) = O_T_EE;

  // Get the current joint configuration from the robot
  std::array<double, 7> q_actual_array;
  for (size_t i = 0; i < 7; i++) {
    q_actual_array[i] = q_[i];
  }

  // Preserve the current q7 for redundancy resolution
  double q7 = q_actual_array[6];

  // Run the IK solver (case-consistent for smooth motion)
  std::array<double, 7> q_solution = franka_IK_EE_CC(O_T_EE_array, q7, q_actual_array);

  // Check if IK returned a valid solution
  if (std::isnan(q_solution[0])) {
    ROS_WARN("IK failed to find a valid solution for the desired pose.");
    return;
  }

  // Update the target joint angles for the impedance controller
  for (size_t i = 0; i < 7; i++) {
    target_q_d_[i] = q_solution[i];
  }
}


}  // namespace franka_interactive_controllers

PLUGINLIB_EXPORT_CLASS(franka_interactive_controllers::JointImpedanceFrankaController,
                       controller_interface::ControllerBase)