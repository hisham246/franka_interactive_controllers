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

  joints_result_.resize(7);
  // set joint positions to current position, to prevent movement
  for (int i = 0; i < 7; i++)
  {
      joints_result_(i) = joint_handles_[i].getPosition();
      last_commanded_pos_[i] = joints_result_(i);
      iters_[i] = 0;
  }

}

void JointImpedanceFrankaController::update(const ros::Time& /*time*/,
                                             const ros::Duration& period) {
  
  // cartesian_pose_handle_->setCommand(target_pose_d_);

  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis = model_handle_->getCoriolis();
  std::array<double, 7> gravity = model_handle_->getGravity();
  // std::array<double, 7> current_tau;

  double interval_length = period.toSec();
  // get joint commands in radians from inverse kinematics
  for (int i = 0; i < 7; i++)
  {
      target_q_d_[i] = joints_result_(i);
  }
  // if goal is reached and robot is originally executing command, we are done
  if (isGoalReached() && is_executing_cmd_)
  {
      is_executing_cmd_ = false;
  }

  double current_pos, p_val;
  for (int i=0; i<7; i++)
  {
      current_pos = joint_handles_[i].getPosition();
      // norm position
      p_val = 2 - (2 * (abs(joint_cmds_[i] - current_pos) / abs(calc_max_pos_diffs[i])));
      // if p val is negative, treat it as 0
      p_val = std::max(p_val, 0.);
      catmullRomSplineVelCmd(p_val, i, interval_length);

      q_d_[i] = limited_joint_cmds_[i];
  }


  for (size_t i=0; i<7; i++){
    q_[i] = joint_handles_[i].getPosition();
    dq_[i] = joint_handles_[i].getVelocity();
    dq_d_[i] = (limited_joint_cmds_[i] - q_d_[i]) / period.toSec();  
  }


  // Compute impedance control torques
  std::array<double, 7> tau_d_calculated;
  for (size_t i = 0; i < 7; i++) {
    tau_d_calculated[i] = coriolis_factor_ * coriolis[i] +
                          k_gains_[i] * (q_d_[i] - q_[i]) +
                          d_gains_[i] * (dq_d_[i] - dq_[i]);
  }

  std::array<double, 7> tau_d_saturated = saturateTorqueRate(tau_d_calculated, robot_state.tau_J_d);

  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(tau_d_saturated[i]);
  }

  for (int i = 0; i < 7; i++) {
    last_commanded_pos_[i] = limited_joint_cmds_[i];
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

void JointImpedanceFrankaController::desiredPoseCallback(const geometry_msgs::Pose &target_pose) {

   if (is_executing_cmd_) 
    {
        ROS_ERROR("JointImpedanceFrankaController: Still executing command!");
        return;
        // panda is still executing command, cannot publish yet to this topic.
    }
    target_pose_.orientation.w = target_pose.orientation.w;
    target_pose_.orientation.x = target_pose.orientation.x;
    target_pose_.orientation.y = target_pose.orientation.y;
    target_pose_.orientation.z = target_pose.orientation.z;

    target_pose_.position.x = target_pose.position.x;
    target_pose_.position.y = target_pose.position.y;
    target_pose_.position.z = target_pose.position.z;

    // use tracik to get joint positions from target pose
    KDL::JntArray ik_result = panda_ik_service_.perform_ik(target_pose_);
    joints_result_ = (panda_ik_service_.is_valid) ? ik_result : joints_result_; 
    
    if (joints_result_.rows() != 7)
    {   
        ROS_ERROR("JointImpedanceFrankaController: Wrong Amount of Rows Received From TRACIK");
        return;
    }
    // for catmull rom spline, break up into two splines
    // spline 1 - increases to max provided velocity
    // spline 2 - decreases down to 0 smoothly
    std::array<double,4> velocity_points_first_spline;
    std::array<double,4> velocity_points_second_spline;
    double max_vel;
    for (int i=0; i<7; i++)
    {
        // get current position
        double cur_pos = joint_handles_[i].getPosition();
        // difference between current position and desired position from ik
        calc_max_pos_diffs[i] = joints_result_(i) - cur_pos;
        // if calc_max_poss_diff is negative, flip sign of max vel
        max_vel = calc_max_pos_diffs[i] < 0 ? -max_abs_vel_ : max_abs_vel_;
        int sign = calc_max_pos_diffs[i] < 0 ? -1 : 1;
        // get p_i-2, p_i-1, p_i, p+i+1 for catmull rom spline
        
        velocity_points_first_spline = {0, 0.3*sign, max_vel, max_vel};
        velocity_points_second_spline = {max_vel, max_vel, 0, 0};

        // tau of 0.3
        
        // velocity as function of position
        vel_catmull_coeffs_first_spline_[i] = catmullRomSpline(0.3, velocity_points_first_spline);
        vel_catmull_coeffs_second_spline_[i] = catmullRomSpline(0.3, velocity_points_second_spline);

    }
    is_executing_cmd_ = true;
    for (int i = 0; i < 7; i++)
    {
        iters_[i] = 0;
    }
    start_time_ = ros::Time::now();
}

std::array<double, 4> JointImpedanceFrankaController::catmullRomSpline(const double &tau, const std::array<double,4> &points)
{
    // catmullRomSpline calculation for any 4 generic points
    // result array for 4 coefficients of cubic polynomial
    std::array<double, 4> coeffs;
    // 4 by 4 matrix for calculating coefficients
    std::array<std::array<double, 4>, 4> catmullMat = {{{0, 1, 0, 0}, {-tau, 0, tau, 0},
                                                        {2*tau, tau-3, 3 - (2*tau), -tau}, 
                                                        {-tau,2-tau,tau-2,tau}}};
    // calculation of coefficients
    for (int i=0; i<4; i++)
    {
        coeffs[i] = (points[0]*catmullMat[i][0]) + (points[1]*catmullMat[i][1]) 
                    + (points[2]*catmullMat[i][2]) + (points[3]*catmullMat[i][3]);
    }
    return coeffs;
}

double JointImpedanceFrankaController::calcSplinePolynomial(const std::array<double,4> &coeffs, const double &x)
{
    // function that calculates third degree polynomial given input x and 
    // 4 coefficients
    double output = 0.;   
    int power = 0;
    for (int i=0; i<4; i++)
    {
        output+=(coeffs[i]*(pow(x, power)));
        power++;
    }
    return output;
}

void JointImpedanceFrankaController::catmullRomSplineVelCmd(const double &norm_pos, const int &joint_num,
                                                    const double &interval)
{
    // individual joints
    // velocity is expressed as a function of position (position is normalized)
    if (norm_pos <= 2 && is_executing_cmd_)
    {
        // valid position and is executing a command
        double vel;
        if (norm_pos < 1)
        {
            // use first spline to get velocity
            vel = calcSplinePolynomial(vel_catmull_coeffs_first_spline_[joint_num], norm_pos);
        }
        else 
        {
            // use second spline to get velocity
            vel = calcSplinePolynomial(vel_catmull_coeffs_second_spline_[joint_num], norm_pos - 1);
        }
        
        // calculate velocity step
        limited_joint_cmds_[joint_num] = last_commanded_pos_[joint_num] + (vel * interval);
    }
    else
    {
        for (int i=0; i<7; i++)
        {
            limited_joint_cmds_[i] = joint_cmds_[i];
        }
        
        is_executing_cmd_ = false;
    }
    
}

bool JointImpedanceFrankaController::isGoalReached()
{
    // sees if goal is reached given a distance threshold epsilon
    // l2 norm is used for overall distance
    double err = 0;
    for (int i = 0; i < 7; i++)
    {
        err += pow(joint_handles_[i].getPosition() - joints_result_(i), 2);
    }
    err = sqrt(err);
    return err <= epsilon_;

}

}  // namespace franka_interactive_controllers

PLUGINLIB_EXPORT_CLASS(franka_interactive_controllers::JointImpedanceFrankaController,
                       controller_interface::ControllerBase)



// The following code is commented out from the update as it is not part of the current implementation

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