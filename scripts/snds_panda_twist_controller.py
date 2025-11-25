#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import rospy
import threading
from geometry_msgs.msg import Twist
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from snds.learn_nn_ds import NL_DS
from franka_msgs.msg import FrankaState
from scipy.spatial.transform import Rotation as R
import torch


class PoseListener:
    def __init__(self, topic='/franka_state_controller/franka_states'):
        self.pose = None
        self.ready = threading.Event()
        self.sub = rospy.Subscriber(topic, FrankaState, self.callback)

    def callback(self, msg: FrankaState):
        # Convert O_T_EE (4x4 column-major transform)
        T = np.array(msg.O_T_EE).reshape(4, 4, order='F')
        # Position
        pos = T[:3, 3]
        # Rotation as rotation vector (Rodrigues form)
        rotvec = R.from_matrix(T[:3, :3]).as_rotvec()
        # Store in the expected [x, y, z, rx, ry, rz] format
        self.pose = np.concatenate([pos, rotvec])
        self.ready.set()

    def wait_for_pose(self, timeout=5.0):
        if not self.ready.wait(timeout):
            raise TimeoutError("Timed out waiting for initial EE pose.")
        return self.pose


def main():
    rospy.init_node("snds_panda_controller")

    # model_dir = "/home/hisham246/uwaterloo/DS-Learning/stable-imitation-policy/res/nlds_policy/snds/test-snds-24-11-20-29.pt"  # Your saved model directory
    model_dir = "/home/hisham246/uwaterloo/DS-Learning/stable-imitation-policy/res/nlds_policy"
    model_name = "test-snds-24-11-20-29"
    network = "snds"  # Network type used during training
    goal_pos = np.array([0.502043495, 0.113929932, 0.100071396], dtype=np.float32)  # Set the desired goal position
    cmd_topic = "/passiveDS/desired_twist"  # TwistStamped topic
    rate_hz = 100.0  # Control rate in Hz
    vel_scale = 1.0  # Scaling factor for the velocity

    # Load SNDS policy
    data_dim = 3  # We trained on 3D Cartesian position → 3D velocity

    nl_ds = NL_DS(
        network=network,
        data_dim=data_dim,
        gpu=False,  # Set to True if using GPU
        eps=0.01,
        alpha=0.01,
        relaxed=True  # Set as per training configuration
    )

    rospy.loginfo("Loading SNDS model '%s' from '%s'", model_name, model_dir)
    # nl_ds.__nn_module.load_state_dict(torch.load(model_dir))
    nl_ds.load(model_name=model_name, dir=model_dir)
    rospy.loginfo("Model loaded successfully.")

    # ROS interfaces
    pub_twist = rospy.Publisher(cmd_topic, Twist, queue_size=1)

    # Initialize PoseListener to subscribe to FrankaState topic
    pose_listener = PoseListener()

    rate = rospy.Rate(rate_hz)

    # Control loop
    while not rospy.is_shutdown():
        # Wait for and get the current end-effector pose
        pose = pose_listener.wait_for_pose()

        # Extract position (x, y, z) and ignore rotation (rx, ry, rz)
        ee_pos = pose[:3]

        # State is position error wrt goal: x = p_current - p_goal
        x = ee_pos - goal_pos  # (3,)
        x_input = x.reshape(1, -1)  # (1, 3)

        # SNDS policy: x -> x_dot (desired Cartesian velocity)
        v = nl_ds.predict(x_input)[0]  # (3,)

        # Optionally scale & clamp for safety
        v_cmd = vel_scale * v
        max_lin = 0.3  # m/s max linear speed
        speed = np.linalg.norm(v_cmd)
        if speed > max_lin and speed > 1e-6:
            v_cmd = v_cmd * (max_lin / speed)

        twist_msg = Twist()
        twist_msg.linear.x = float(v_cmd[0])
        twist_msg.linear.y = float(v_cmd[1])
        twist_msg.linear.z = float(v_cmd[2])

        # For now, no rotational component from SNDS (position-only policy)
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = 0.0

        # Publish the Twist message
        pub_twist.publish(twist_msg)

        rate.sleep()


if __name__ == "__main__":
    main()