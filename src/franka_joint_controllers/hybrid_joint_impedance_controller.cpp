// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
// #include <franka_interactive_controllers/joint_impedance_example_controller.h>
#include <pinocchio/fwd.hpp>
#include <hybrid_joint_impedance_controller.h>

#include <array>
#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <controller_manager_msgs/SwitchController.h>
#include <Eigen/Dense>

#include <franka/robot_state.h>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/spatial/explog.hpp>

namespace franka_interactive_controllers {

bool HybridJointImpedanceController::init(hardware_interface::RobotHW* robot_hw,
                                           ros::NodeHandle& node_handle) {

  sub_desired_pose_ = node_handle.subscribe(
    "/hybrid_joint_impedance_controller/desired_pose", 20, &HybridJointImpedanceController::desiredPoseCallback, this,
    ros::TransportHints().reliable().tcpNoDelay());

  desired_joints_pub_ = node_handle.advertise<sensor_msgs::JointState>("/hybrid_joint_impedance_controller/desired_joint_positions", 10);


  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("HybridJointImpedanceController: Could not read parameter arm_id");
    return false;
  }

  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "HybridJointImpedanceController: Invalid or no joint_names parameters provided, aborting "
        "controller init!");
    return false;
  }

  if (!node_handle.getParam("k_gains", k_gains_) || k_gains_.size() != 7) {
    ROS_ERROR(
        "HybridJointImpedanceController:  Invalid or no k_gain parameters provided, aborting "
        "controller init!");
    return false;
  }

  if (!node_handle.getParam("d_gains", d_gains_) || d_gains_.size() != 7) {
    ROS_ERROR(
        "HybridJointImpedanceController:  Invalid or no d_gain parameters provided, aborting "
        "controller init!");
    return false;
  }

  if (!node_handle.getParam("coriolis_factor", coriolis_factor_)) {
    ROS_INFO_STREAM("HybridJointImpedanceController: coriolis_factor not found. Defaulting to "
                    << coriolis_factor_);
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "HybridJointImpedanceController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "HybridJointImpedanceController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "HybridJointImpedanceController: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "HybridJointImpedanceController: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }


  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "HybridJointImpedanceController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "HybridJointImpedanceController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }


  violation_check_timer_ = node_handle.createTimer(ros::Duration(0.05),
    &HybridJointImpedanceController::violationCheckCallback, this);

  // Getting Dynamic Reconfigure objects
  dynamic_reconfigure_compliance_param_node_ =
      ros::NodeHandle(node_handle.getNamespace() + "/dynamic_reconfigure_compliance_param_node");

  dynamic_server_compliance_param_ = std::make_unique<
      dynamic_reconfigure::Server<franka_interactive_controllers::compliance_param_hybridConfig>>(
      dynamic_reconfigure_compliance_param_node_);

  dynamic_server_compliance_param_->setCallback(
      boost::bind(&HybridJointImpedanceController::complianceParamCallback, this, _1, _2));


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

void HybridJointImpedanceController::starting(const ros::Time& /*time*/) {

  franka::RobotState initial_state = state_handle_->getRobotState();
  // get jacobian
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);

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

void HybridJointImpedanceController::update(const ros::Time& /*time*/,
                                             const ros::Duration& period) {

  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis = model_handle_->getCoriolis();
  std::array<double, 42> jacobian_array =
    model_handle_->getZeroJacobian(franka::Frame::kEndEffector);

  // Convert jacobian to Eigen matrix 
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());

  Eigen::MatrixXd J_T(7, 6);
  J_T = jacobian.transpose();


  // Hybrid impedance gains
  Eigen::Matrix<double, 7, 7> Kq  = Eigen::Map<const Eigen::Matrix<double, 7, 1>>(k_gains_.data()).asDiagonal();
  Eigen::Matrix<double, 7, 7> Kqd = Eigen::Map<const Eigen::Matrix<double, 7, 1>>(d_gains_.data()).asDiagonal();
  Eigen::Matrix<double, 6, 6> Kx = cartesian_stiffness_;
  Eigen::Matrix<double, 6, 6> Kxd = cartesian_damping_;

  Eigen::Matrix<double, 7, 7> Kp = J_T * Kx * jacobian + Kq;
  Eigen::Matrix<double, 7, 7> Kd = J_T * Kxd * jacobian + Kqd;


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

  Eigen::Matrix<double, 7, 1> q_error;
  Eigen::Matrix<double, 7, 1> dq_error;

  // Eigen::Matrix<double, 7, 1> dq_error;
  // for (size_t i = 0; i < 7; ++i) {
  //   q_error[i] = q_d_[i] - q_[i];
  //   dq_error[i] = robot_state.dq_d[i] - dq_filtered_[i];
  // }

  for (size_t i = 0; i < 7; ++i) {
    q_error[i] = q_d_[i] - q_[i];
    dq_error[i] = dq_d_[i] - dq_filtered_[i];
  }

  Eigen::Matrix<double, 7, 1> tau_d_calculated_eigen = 
    coriolis_factor_ * Eigen::Map<const Eigen::Matrix<double, 7, 1>>(coriolis.data()) 
    + Kp * q_error
    + Kd * dq_error;

  std::array<double, 7> tau_d_calculated;
  Eigen::VectorXd::Map(tau_d_calculated.data(), 7) = tau_d_calculated_eigen;

//   for (size_t i = 0; i < 7; i++) {
//     q_filtered_[i] = (1 - q_filt_) * q_filtered_[i] + q_filt_ * q_d_[i];
//   }

//   std::array<double, 7> tau_d_calculated;
//   for (size_t i = 0; i < 7; i++) {
//     tau_d_calculated[i] = coriolis_factor_ * coriolis[i] +
//                           k_gains_[i] * (q_filtered_[i] - q_[i]) +
//                           d_gains_[i] * (dq_d_[i] - dq_filtered_[i]);
//                         }
  

  // Maximum torque difference with a sampling rate of 1 kHz. The maximum torque rate is
  // 1000 * (1 / sampling_time).
  std::array<double, 7> tau_d_saturated = saturateTorqueRate(tau_d_calculated, robot_state.tau_J_d);

  for (size_t i=0; i<7; i++){
    tau_d_calculated[i] = 0.0;
  }

  // for (size_t i=0; i<7; i++){
  //   q_d_[i] = target_q_d_[i] * q_filt_ + q_d_[i] * (1 - q_filt_);
  // }
    
  for (size_t i = 0; i < 7; i++) {
    joint_handles_[i].setCommand(tau_d_calculated[i]);
  }

  // cartesian_stiffness_ =
  //     filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * cartesian_stiffness_;
  // cartesian_damping_ =
  //     filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * cartesian_damping_;

  cartesian_stiffness_  = cartesian_stiffness_target_;
  cartesian_damping_    = cartesian_damping_target_;

  std::lock_guard<std::mutex> position_d_target_mutex_lock(
    position_and_orientation_d_target_mutex_);
  position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
  orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);

  Eigen::AngleAxisd aa(orientation_d_);
  Eigen::Vector3d rotvec = aa.angle() * aa.axis();
  double table_height_threshold = 0.005; // 5 mm above table (adjust as needed)
  enforceTableCollisionConstraint(position_d_, rotvec, table_height_threshold);
}

std::array<double, 7> HybridJointImpedanceController::saturateTorqueRate(
    const std::array<double, 7>& tau_d_calculated,
    const std::array<double, 7>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  std::array<double, 7> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] = tau_J_d[i] + std::max(std::min(difference, kDeltaTauMax), -kDeltaTauMax);
  }
  return tau_d_saturated;
}

void HybridJointImpedanceController::desiredPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg)
{

  std::lock_guard<std::mutex> position_d_target_mutex_lock(position_and_orientation_d_target_mutex_);
  position_d_target_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;

  // Eigen::Vector3d target_position(position_d_target_);
  
  Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
  orientation_d_target_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
      msg->pose.orientation.z, msg->pose.orientation.w;
  
  if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
    orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
  }

  // -------- Inverse Kinematics using Pinocchio --------
  Eigen::Quaterniond quat(
    orientation_d_.w(),
    orientation_d_.x(),
    orientation_d_.y(),
    orientation_d_.z());
  quat.normalize();

  pinocchio::SE3 oMdes(
      orientation_d_.toRotationMatrix(),
      position_d_
  );

  // pinocchio::SE3 oMdes(
  //     quat.toRotationMatrix(),
  //     position_d_
  // );


  // Use current joint positions as initial guess
  Eigen::VectorXd q(pinocchio_model_.nq);
  for (size_t i = 0; i < 7; i++) {
    q[i] = q_[i];
  }

  // IK parameters
  const double eps = 1e-4;
  const int IT_MAX = 1000;
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
    const pinocchio::SE3 iMd = current_pose.actInv(oMdes);
    err = pinocchio::log6(iMd).toVector();

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
}  

void HybridJointImpedanceController::complianceParamCallback(
    franka_interactive_controllers::compliance_param_hybridConfig& config,
    uint32_t /*level*/) {
  // Set individual translational stiffness for each axis
  cartesian_stiffness_target_.setIdentity();
  cartesian_stiffness_target_(0, 0) = config.translational_stiffness_x;
  cartesian_stiffness_target_(1, 1) = config.translational_stiffness_y;
  cartesian_stiffness_target_(2, 2) = config.translational_stiffness_z;

  // Set individual rotational stiffness for each axis
  cartesian_stiffness_target_(3, 3) = config.rotational_stiffness_x;
  cartesian_stiffness_target_(4, 4) = config.rotational_stiffness_y;
  cartesian_stiffness_target_(5, 5) = config.rotational_stiffness_z;

  // Set damping to ensure critical damping ratio
  cartesian_damping_target_.setIdentity();
  cartesian_damping_target_(0, 0) = 2.0 * sqrt(config.translational_stiffness_x);
  cartesian_damping_target_(1, 1) = 2.0 * sqrt(config.translational_stiffness_y);
  cartesian_damping_target_(2, 2) = 2.0 * sqrt(config.translational_stiffness_z);
  cartesian_damping_target_(3, 3) = 2.0 * sqrt(config.rotational_stiffness_x);
  cartesian_damping_target_(4, 4) = 2.0 * sqrt(config.rotational_stiffness_y);
  cartesian_damping_target_(5, 5) = 2.0 * sqrt(config.rotational_stiffness_z);
}

void HybridJointImpedanceController::enforceTableCollisionConstraint(
    Eigen::Vector3d& pos, const Eigen::Vector3d& rotvec, double height_threshold)
{
  const double gripper_width = 0.1;               // Fixed for safety
  const double finger_thickness = 55.5 / 1000.0;  // Side pad thickness (m)
  // const double gripper_height = 0.06;             // Height of gripper above fingertips (m)

  std::vector<Eigen::Vector3d> keypoints;
  for (int dx : {-1, 1}) {
    for (int dy : {-1, 1}) {
      keypoints.emplace_back(
          dx * gripper_width / 2.0,
          dy * finger_thickness / 2.0,
          0.0
      );
    }
  }

  Eigen::Matrix3d rot = Eigen::AngleAxisd(rotvec.norm(), rotvec.normalized()).toRotationMatrix();
  double min_z = std::numeric_limits<double>::infinity();
  for (const auto& pt : keypoints) {
    Eigen::Vector3d world_pt = rot * pt + pos;  // Transform from fingertip frame to world
    min_z = std::min(min_z, world_pt.z());
  }

  double delta = std::max(height_threshold - min_z, 0.0);
  pos.z() += delta;
}

void HybridJointImpedanceController::violationCheckCallback(const ros::TimerEvent&) {
  if (violation_triggered_) {
    ROS_ERROR_STREAM("Safety violation detected. Requesting controller stop...");

    ros::NodeHandle nh;
    ros::ServiceClient switch_client = nh.serviceClient<controller_manager_msgs::SwitchController>(
        "/controller_manager/switch_controller");

    controller_manager_msgs::SwitchController switch_srv;
    switch_srv.request.stop_controllers.push_back("hybrid_joint_impedance_controller");
    switch_srv.request.strictness = controller_manager_msgs::SwitchController::Request::STRICT;

    if (switch_client.call(switch_srv)) {
      if (switch_srv.response.ok) {
        ROS_WARN("HybridJointImpedanceController stopped successfully.");
      } else {
        ROS_ERROR("Controller stop request was sent but failed.");
      }
    } else {
      ROS_ERROR("Failed to call controller_manager/switch_controller service.");
    }

    // Prevent repeated attempts
    violation_triggered_ = false;
  }
} 

} // namespace franka_interactive_controllers

PLUGINLIB_EXPORT_CLASS(franka_interactive_controllers::HybridJointImpedanceController,
                       controller_interface::ControllerBase)