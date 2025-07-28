// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
// #include <franka_interactive_controllers/joint_impedance_example_controller.h>
#include <pinocchio/fwd.hpp>
#include <joint_impedance_franka_controller.h>

#include <array>
#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <Eigen/Dense>

#include <franka/robot_state.h>
// #include <franka_utils/franka_ik_He.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/spatial/explog.hpp>


namespace franka_interactive_controllers {

bool JointImpedanceFrankaController::init(hardware_interface::RobotHW* robot_hw,
                                           ros::NodeHandle& node_handle) {

  sub_desired_pose_ = node_handle.subscribe(
    "/joint_impedance_controller/desired_pose", 20, &JointImpedanceFrankaController::desiredPoseCallback, this,
    ros::TransportHints().reliable().tcpNoDelay());

  desired_joints_pub_ = node_handle.advertise<sensor_msgs::JointState>("/joint_impedance_controller/desired_joint_positions", 10);


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

  // Initializing variables
  position_d_.setZero();
  orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  position_d_target_.setZero();
  orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;

  std::string urdf_path;
  if (!node_handle.getParam("pinocchio_urdf_path", urdf_path)) {
    ROS_ERROR("Pinocchio URDF path not specified!");
    return false;
  }


  // Build the Panda model
  pinocchio::urdf::buildModel(urdf_path, pinocchio_model_);
  pinocchio_data_ = pinocchio::Data(pinocchio_model_);
  ee_frame_id_ = pinocchio_model_.getFrameId("panda_hand_tcp");
  
  return true;
}

void JointImpedanceFrankaController::starting(const ros::Time& /*time*/) {

  franka::RobotState initial_state = state_handle_->getRobotState();

  // convert to eigen
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
}

void JointImpedanceFrankaController::update(const ros::Time& /*time*/,
                                             const ros::Duration& period) {

  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis = model_handle_->getCoriolis();

  // Get current position and velocity of joints
  for (size_t i=0; i<7; i++){
    q_[i] = robot_state.q[i];
    dq_[i] = robot_state.dq[i];
  }
  
  // Get desired velocity based on desired joint poses and current joint torques for saturation
  for (size_t i=0; i<7; i++){
    dq_d_[i] = 0.0;
  }
  
  double alpha = 0.99;
  for (size_t i = 0; i < 7; i++) {
    dq_filtered_[i] = (1 - alpha) * dq_filtered_[i] + alpha * dq_[i];
  }

  for (size_t i = 0; i < 7; i++) {
    q_filtered_[i] = (1 - q_filt_) * q_filtered_[i] + q_filt_ * q_d_[i];
  }

  std::array<double, 7> tau_d_calculated;
  for (size_t i = 0; i < 7; i++) {
    tau_d_calculated[i] = coriolis_factor_ * coriolis[i] +
                          k_gains_[i] * (q_filtered_[i] - q_[i]) +
                          d_gains_[i] * (dq_d_[i] - dq_filtered_[i]);
                        }
  

  // Maximum torque difference with a sampling rate of 1 kHz. The maximum torque rate is
  // 1000 * (1 / sampling_time).
  // std::array<double, 7> tau_d_saturated = saturateTorqueRate(tau_d_calculated, current_tau);
  // std::array<double, 7> tau_d_saturated = tau_d_calculated;
  std::array<double, 7> tau_d_saturated = saturateTorqueRate(tau_d_calculated, robot_state.tau_J_d);


  // for (size_t i=0; i<7; i++){
  //   q_d_[i] = target_q_d_[i] * q_filt_ + q_d_[i] * (1 - q_filt_);
  // }
    
  for (size_t i = 0; i < 7; i++) {
    joint_handles_[i].setCommand(tau_d_saturated[i]);
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

  Eigen::Quaterniond quat(
    orientation_d_target_.w(),
    orientation_d_target_.x(),
    orientation_d_target_.y(),
    orientation_d_target_.z());
  quat.normalize();

  pinocchio::SE3 oMdes(
      orientation_d_target_.toRotationMatrix(),
      position_d_target_
  );

  // Use current joint positions as initial guess
  Eigen::VectorXd q(pinocchio_model_.nq);
  for (size_t i = 0; i < 7; i++) {
    q[i] = q_[i];
  }

  // IK parameters
  const double eps = 1e-4;
  const int IT_MAX = 500;
  const double DT = 1e-1;
  const double damp = 1e-6;

  pinocchio::Data::Matrix6x J(6, pinocchio_model_.nv);
  J.setZero();

  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  Vector6d err;
  Eigen::VectorXd v(pinocchio_model_.nv);

  bool success = false;
  for (int i = 0;; i++) {
    pinocchio::forwardKinematics(pinocchio_model_, pinocchio_data_, q);
    pinocchio::updateFramePlacements(pinocchio_model_, pinocchio_data_);

    const pinocchio::SE3 &current_pose = pinocchio_data_.oMf[ee_frame_id_];
    // err = pinocchio::log6(oMdes.actInv(current_pose)).toVector();
    const pinocchio::SE3 iMd = current_pose.actInv(oMdes);
    err = pinocchio::log6(iMd).toVector();
    // const pinocchio::SE3 iMd = pinocchio_data_.oMi[ee_frame_id_].actInv(oMdes);
    // ROS_INFO_STREAM("Current pose" << iMd);
    // err = pinocchio::log6(iMd).toVector();
    // ROS_INFO_STREAM("Error vector: " << err);

    if (err.norm() < eps)
    {
      success = true;
      break;
    }
    if (i >= IT_MAX)
    {
      success = false;
      break;
    }

    pinocchio::computeFrameJacobian(pinocchio_model_, pinocchio_data_, q, ee_frame_id_, pinocchio::LOCAL, J);


    pinocchio::Data::Matrix6 Jlog;
    pinocchio::Jlog6(iMd.inverse(), Jlog);
    J = -Jlog * J;
    pinocchio::Data::Matrix6 JJt;
    JJt.noalias() = J * J.transpose();
    JJt.diagonal().array() += damp;
    v.noalias() = - J.transpose() * JJt.ldlt().solve(err);
    q = pinocchio::integrate(pinocchio_model_, q, v * DT);
  }

  if (!success) {
    ROS_WARN("Pinocchio IK did not converge to the desired pose.");
    return;
  }

  // Publish desired joint positions
  sensor_msgs::JointState joint_state_msg;
  joint_state_msg.header.stamp = ros::Time::now();
  joint_state_msg.name = {"panda_joint1", "panda_joint2", "panda_joint3", 
                          "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"};
  joint_state_msg.position.resize(7);

  for (size_t i = 0; i < 7; i++) {
    joint_state_msg.position[i] = q[i];
  }

  desired_joints_pub_.publish(joint_state_msg);

  // Update the impedance controller's target joint configuration
  for (size_t i = 0; i < 7; i++) {
    q_d_[i] = q[i];
  }
}  // namespace franka_interactive_controllers

}

PLUGINLIB_EXPORT_CLASS(franka_interactive_controllers::JointImpedanceFrankaController,
                       controller_interface::ControllerBase)

  // ------------ Analytical IK code ----------------                     
  // ROS_INFO_STREAM("JointImpedanceFrankaController: Received desired pose for impedance controller.");

  // // Convert received pose to std::array<double,16>
  // if (msg.data.size() != 16) {
  //   ROS_ERROR("Received invalid pose: expected 16 elements for a 4x4 transformation matrix.");
  //   return;
  // }

  // std::array<double, 16> O_T_EE_array;
  // for (size_t i = 0; i < 16; i++) {
  //   O_T_EE_array[i] = msg.data[i];
  // }

  // // Convert PoseStamped to a 4x4 transformation matrix (Eigen)
  // Eigen::Matrix3d R = orientation_d_target_.toRotationMatrix();
  // Eigen::Vector3d p(position_d_target_);

  // Eigen::Matrix4d O_T_EE;
  // O_T_EE.setIdentity();
  // O_T_EE.topLeftCorner<3, 3>() = R;
  // O_T_EE.topRightCorner<3, 1>() = p;

  // // Convert Eigen matrix to column-major std::array<double,16> for IK
  // std::array<double, 16> O_T_EE_array;
  // Eigen::Map<Eigen::Matrix<double, 4, 4>>(O_T_EE_array.data()) = O_T_EE;

  // // Get the current joint configuration from the robot
  // std::array<double, 7> q_actual_array;
  // for (size_t i = 0; i < 7; i++) {
  //   q_actual_array[i] = q_[i];
  // }

  // // Preserve the current q7 for redundancy resolution
  // double q7 = q_actual_array[6];

  // // Run the IK solver (case-consistent for smooth motion)
  // std::array<double, 7> q_solution = franka_IK_EE_CC(O_T_EE_array, q7, q_actual_array);

  // // Check if IK returned a valid solution
  // if (std::isnan(q_solution[0])) {
  //   ROS_WARN("IK failed to find a valid solution for the desired pose.");
  //   return;
  // }

  // // Update the target joint angles for the impedance controller
  // for (size_t i = 0; i < 7; i++) {
  //   target_q_d_[i] = q_solution[i];
  // }