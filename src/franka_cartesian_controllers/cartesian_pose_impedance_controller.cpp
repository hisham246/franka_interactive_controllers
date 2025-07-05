// This code was derived from franka_example controllers
// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license.
// Current development and modification of this code by Nadia Figueroa (MIT) 2021.

#include <cartesian_pose_impedance_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <pseudo_inversion.h>
#include <hardware_interface/joint_command_interface.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/WrenchStamped.h>

namespace franka_interactive_controllers {

bool CartesianPoseImpedanceController::init(hardware_interface::RobotHW* robot_hw,
                                               ros::NodeHandle& node_handle) {
  std::vector<double> cartesian_stiffness_vector;
  std::vector<double> cartesian_damping_vector;

  sub_desired_pose_ = node_handle.subscribe(
      "/cartesian_impedance_controller/desired_pose", 20, &CartesianPoseImpedanceController::desiredPoseCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

  // Getting ROSParams
  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_STREAM("CartesianPoseImpedanceController: Could not read parameter arm_id");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "CartesianPoseImpedanceController: Invalid or no joint_names parameters provided, "
        "aborting controller init!");
    return false;
  }

  // Getting libranka control interfaces
  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianPoseImpedanceController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "CartesianPoseImpedanceController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianPoseImpedanceController: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "CartesianPoseImpedanceController: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianPoseImpedanceController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "CartesianPoseImpedanceController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  // Getting Dynamic Reconfigure objects
  dynamic_reconfigure_compliance_param_node_ =
      ros::NodeHandle(node_handle.getNamespace() + "/dynamic_reconfigure_compliance_param_node");

  dynamic_server_compliance_param_ = std::make_unique<
      dynamic_reconfigure::Server<franka_interactive_controllers::compliance_full_paramConfig>>(
      dynamic_reconfigure_compliance_param_node_);

  dynamic_server_compliance_param_->setCallback(
      boost::bind(&CartesianPoseImpedanceController::complianceParamCallback, this, _1, _2));


  // Initializing variables
  position_d_.setZero();
  orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  position_d_target_.setZero();
  orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  cartesian_stiffness_.setZero();
  cartesian_damping_.setZero();

  ///////////////////////////////////////////////////////////////////////////
  ////////////////  Parameter Initialization from YAML FILES!!!     /////////
  ///////////////////////////////////////////////////////////////////////////

   // Initialize stiffness and damping gains
  cartesian_stiffness_target_.setIdentity();
  cartesian_damping_target_.setIdentity();
  std::vector<double> cartesian_stiffness_target_yaml;
  if (!node_handle.getParam("cartesian_stiffness_target", cartesian_stiffness_target_yaml) || cartesian_stiffness_target_yaml.size() != 6) {
    ROS_ERROR(
      "CartesianPoseImpedanceController: Invalid or no cartesian_stiffness_target_yaml parameters provided, "
      "aborting controller init!");
    return false;
  }
  for (int i = 0; i < 6; i ++) {
    cartesian_stiffness_target_(i,i) = cartesian_stiffness_target_yaml[i];
  }
  // Damping ratio = 1

  // default_cart_stiffness_target_ << 300, 300, 300, 50, 50, 50;
  // Eigen::VectorXd xi(6);
  // xi << 0.5, 0.5, 0.5, 0.1, 0.1, 0.1;
  // for (int i = 0; i < 6; i ++) {
  //   if (cartesian_stiffness_target_yaml[i] == 0.0)
  //     cartesian_damping_target_(i,i) = xi[i] * 2.0 * sqrt(default_cart_stiffness_target_[i]);
  //   else
  //     cartesian_damping_target_(i,i) = xi[i] * 2.0 * sqrt(cartesian_stiffness_target_yaml[i]);
  // }

  // ROS_INFO_STREAM("cartesian_stiffness_target_: " << std::endl <<  cartesian_stiffness_target_);
  // ROS_INFO_STREAM("cartesian_damping_target_: " << std::endl <<  cartesian_damping_target_);

  // if (!node_handle.getParam("nullspace_stiffness", nullspace_stiffness_target_) || nullspace_stiffness_target_ <= 0) {
  //   ROS_ERROR(
  //     "CartesianPoseImpedanceController: Invalid or no nullspace_stiffness parameters provided, "
  //     "aborting controller init!");
  //   return false;
  // }
  // ROS_INFO_STREAM("nullspace_stiffness_target_: " << std::endl <<  nullspace_stiffness_target_);

  // // Initialize variables for tool compensation from yaml config file
  // activate_tool_compensation_ = true;
  // tool_compensation_force_.setZero();
  // std::vector<double> external_tool_compensation;
  // // tool_compensation_force_ << 0.46, -0.17, -1.64, 0, 0, 0;  //read from yaml
  // if (!node_handle.getParam("external_tool_compensation", external_tool_compensation) || external_tool_compensation.size() != 6) {
  //     ROS_ERROR(
  //         "CartesianPoseImpedanceController: Invalid or no external_tool_compensation parameters provided, "
  //         "aborting controller init!");
  //     return false;
  //   }
  // for (size_t i = 0; i < 6; ++i) 
  //   tool_compensation_force_[i] = external_tool_compensation.at(i);
  // ROS_INFO_STREAM("External tool compensation force: " << std::endl << tool_compensation_force_);

  // // Initialize variables for nullspace control from yaml config file
  // q_d_nullspace_.setZero();
  // std::vector<double> q_nullspace;
  // if (node_handle.getParam("q_nullspace", q_nullspace)) {
  //   q_d_nullspace_initialized_ = true;
  //   if (q_nullspace.size() != 7) {
  //     ROS_ERROR(
  //       "CartesianPoseImpedanceController: Invalid or no q_nullspace parameters provided, "
  //       "aborting controller init!");
  //     return false;
  //   }
  //   for (size_t i = 0; i < 7; ++i) 
  //     q_d_nullspace_[i] = q_nullspace.at(i);
  //   ROS_INFO_STREAM("Desired nullspace position (from YAML): " << std::endl << q_d_nullspace_);
  // }
   // Added: Publish torques
  tau_d_pub_ = node_handle.advertise<std_msgs::Float64MultiArray>(node_handle.getNamespace() + "/tau_d", 10);
  franka_EE_wrench_pub = node_handle.advertise<geometry_msgs::WrenchStamped>(node_handle.getNamespace() + "/franka_ee_wrench", 20);
  franka_wrench_pub = node_handle.advertise<geometry_msgs::WrenchStamped>(node_handle.getNamespace() + "/franka_wrench", 20);
  return true;
}

void CartesianPoseImpedanceController::starting(const ros::Time& /*time*/) {

  // Get robot current/initial joint state
  franka::RobotState initial_state = state_handle_->getRobotState();
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());

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

  if (!q_d_nullspace_initialized_) {
    q_d_nullspace_ = q_initial;
    q_d_nullspace_initialized_ = true;
    ROS_INFO_STREAM("Desired nullspace position (from q_initial): " << std::endl << q_d_nullspace_);
  }
}

void CartesianPoseImpedanceController::update(const ros::Time& /*time*/,
                                                 const ros::Duration& period) {

  // if(period.toSec() > 0.001) {
  //   ROS_WARN_STREAM("lost it" << period.toSec());
  // }
  // get state variables
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 49> mass_array = model_handle_->getMass();
  std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  std::array<double, 7> gravity_array = model_handle_->getGravity();

  // convert to Eigen
  Eigen::Map<Eigen::Matrix<double, 7, 7>> inertia(mass_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> gravity(gravity_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(  // NOLINT (readability-identifier-naming)
      robot_state.tau_J_d.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.linear());

  Eigen::Map<const Eigen::Matrix<double, 7, 1> > tau_ext(robot_state.tau_ext_hat_filtered.data());

  // load from the robot the updated wrench
  Eigen::Map<Eigen::Matrix<double, 6, 1>> wrench_v(robot_state.O_F_ext_hat_K.data());

  Eigen::Vector3d ext_wrench_t, ext_wrench_r;
  Eigen::VectorXd ext_wrench(6);
  Eigen::MatrixXd Jpinv(7, 6);
  Eigen::MatrixXd JpinvT(6, 7);

  Jpinv = jacobian.completeOrthogonalDecomposition().pseudoInverse();
  JpinvT = (jacobian.transpose()).completeOrthogonalDecomposition().pseudoInverse();
  double th_F_msr = 2.;
  ext_wrench = (JpinvT * tau_ext).transpose();

  for (int j = 0; j < 3; ++j){
      ext_wrench_t(j) = ext_wrench(j);
      ext_wrench_r(j) = ext_wrench(j+3);

  }

  // if (fabs(ext_wrench_t(2))>th_F_msr)
  // {
  //     if (ext_wrench_t(2)>0.)
  //         ext_wrench_t(2) = ext_wrench_t(2) - th_F_msr;
  //     else
  //         ext_wrench_t(2) = ext_wrench_t(2) + th_F_msr;
  // }
  // else
  //     ext_wrench_t(2) = 0.;


  // wrench publisher
  geometry_msgs::WrenchStamped wrench_msg;
  wrench_msg.wrench.force.x = ext_wrench_t(0);
  wrench_msg.wrench.force.y = ext_wrench_t(1);
  wrench_msg.wrench.force.z = ext_wrench_t(2);
  wrench_msg.wrench.torque.x = ext_wrench_r(0);
  wrench_msg.wrench.torque.y = ext_wrench_r(1);
  wrench_msg.wrench.torque.z = ext_wrench_r(2);


  geometry_msgs::WrenchStamped wrench_m;
  wrench_m.wrench.force.x = wrench_v(0);
  wrench_m.wrench.force.y = wrench_v(1);
  wrench_m.wrench.force.z = wrench_v(2);
  wrench_m.wrench.torque.x = wrench_v(4);
  wrench_m.wrench.torque.y = wrench_v(5);
  wrench_m.wrench.torque.z = wrench_v(6);

  franka_EE_wrench_pub.publish(wrench_msg);
  franka_wrench_pub.publish(wrench_m);

  //////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////              COMPUTING TASK CONTROL TORQUE           //////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////

  // compute control
  // allocate variables
  Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7), tau_tool(7);


  // ROS_INFO_STREAM ("Doing Cartesian Impedance Control");            
  // compute error to desired pose
  // position error
  Eigen::Matrix<double, 6, 1> error;
  error.head(3) << position - position_d_;

  // orientation error
  if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  // "difference" quaternion
  Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  // Transform to base frame
  error.tail(3) << -transform.linear() * error.tail(3);

  // Cartesian PD control with damping ratio = 1
  Eigen::Matrix<double, 6, 1> velocity;
  velocity << jacobian * dq;
  Eigen::VectorXd     F_ee_des_;
  F_ee_des_.resize(6);
  F_ee_des_ << -cartesian_stiffness_ * error - cartesian_damping_ * velocity;
  tau_task << jacobian.transpose() * F_ee_des_;
  // ROS_WARN_STREAM_THROTTLE(0.5, "Current Velocity Norm:" << velocity.head(3).norm());
  // ROS_WARN_STREAM_THROTTLE(0.5, "Classic Linear Control Force:" << F_ee_des_.head(3).norm());
  // ROS_WARN_STREAM_THROTTLE(0.5, "Classic Angular Control Force :" << F_ee_des_.tail(3).norm());

  //////////////////////////////////////////////////////////////////////////////////////////////////


  // pseudoinverse for nullspace handling
  // kinematic pseudoinverse
  Eigen::MatrixXd jacobian_transpose_pinv;
  pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

  // nullspace PD control with damping ratio = 1
  // ROS_WARN_STREAM_THROTTLE(0.5, "Nullspace stiffness:" << nullspace_stiffness_); 
  tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
                    jacobian.transpose() * jacobian_transpose_pinv) *
                       (nullspace_stiffness_ * (q_d_nullspace_ - q) -
                        (2.0 * sqrt(nullspace_stiffness_)) * dq);

  // ROS_WARN_STREAM_THROTTLE(0.5, "Nullspace torques:" << tau_nullspace.transpose());    
  // double tau_nullspace_0 = tau_nullspace(0);
  // tau_nullspace.setZero();
  // tau_nullspace[0] = tau_nullspace_0; 


  // Desired torque
  tau_d << tau_task + tau_nullspace + coriolis;
  // ROS_WARN_STREAM_THROTTLE(0.5, "Desired control torque:" << tau_d.transpose());

  // Saturate torque rate to avoid discontinuities
  tau_d << saturateTorqueRate(tau_d, tau_J_d);

  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(tau_d(i));
  }

  publishDesiredTorques(tau_d);

  //////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////

  // update parameters changed online either through dynamic reconfigure or through the interactive
  // target by fi1ltering
  // cartesian_stiffness_  = cartesian_stiffness_target_;
  // cartesian_damping_    = cartesian_damping_target_;
  // nullspace_stiffness_  = nullspace_stiffness_target_;
  // position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
  // orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);

  cartesian_stiffness_ =
      filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * cartesian_stiffness_;
  cartesian_damping_ =
      filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * cartesian_damping_;
  nullspace_stiffness_ =
      filter_params_ * nullspace_stiffness_target_ + (1.0 - filter_params_) * nullspace_stiffness_;
  std::lock_guard<std::mutex> position_d_target_mutex_lock(
      position_and_orientation_d_target_mutex_);
  position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
  orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);
}

Eigen::Matrix<double, 7, 1> CartesianPoseImpedanceController::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] =
        tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}

void CartesianPoseImpedanceController::complianceParamCallback(
    franka_interactive_controllers::compliance_full_paramConfig& config,
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

  nullspace_stiffness_target_ = config.nullspace_stiffness;
}

void CartesianPoseImpedanceController::desiredPoseCallback(
    const geometry_msgs::PoseStampedConstPtr& msg) {

  std::lock_guard<std::mutex> position_d_target_mutex_lock(position_and_orientation_d_target_mutex_);
  position_d_target_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  // ROS_INFO_STREAM("[CALLBACK] Desired ee position from DS: " << position_d_target_);
  
  Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
  orientation_d_target_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
      msg->pose.orientation.z, msg->pose.orientation.w;
  
  if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
    orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
  }
}

void CartesianPoseImpedanceController::publishDesiredTorques(const Eigen::Matrix<double, 7, 1>& tau_d) {
  std_msgs::Float64MultiArray msg;
  for (int i = 0; i < tau_d.size(); ++i) {
    msg.data.push_back(tau_d(i));
  }
  tau_d_pub_.publish(msg);
}

}  // namespace franka_interactive_controllers

PLUGINLIB_EXPORT_CLASS(franka_interactive_controllers::CartesianPoseImpedanceController,
                       controller_interface::ControllerBase)