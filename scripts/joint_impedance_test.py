# !/usr/bin/env python3

import rospy
import numpy as np
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import math

# Global variables
current_position = None
current_quaternion = None

def franka_state_callback(msg):
    global current_position, current_quaternion
    # O_T_EE is 16 elements (4x4 matrix, column-major)
    T = np.array(msg.O_T_EE).reshape(4, 4, order='F')

    # Extract position
    position = T[:3, 3]

    # Extract quaternion (w, x, y, z) from rotation matrix
    rot = R.from_matrix(T[:3, :3])
    quat = rot.as_quat()  # (x, y, z, w)

    current_position = position
    current_quaternion = quat

def publish_circular_trajectory(radius=0.5, duration=10.0, freq=100):
    """Publishes a circular trajectory in the XY plane, keeping Z and orientation fixed."""
    rospy.loginfo("Starting circular trajectory...")

    pub = rospy.Publisher("/joint_impedance_controller/desired_pose", PoseStamped, queue_size=1)
    rate = rospy.Rate(freq)

    center_x, center_y, center_z = current_position
    quat = current_quaternion
    start_time = rospy.Time.now().to_sec()

    while not rospy.is_shutdown():
        t = rospy.Time.now().to_sec() - start_time
        if t > duration:
            rospy.loginfo("Circular trajectory completed.")
            break

        # Circle in XY plane
        x = center_x + radius * math.cos(2 * math.pi * t / duration)
        y = center_y + radius * math.sin(2 * math.pi * t / duration)
        z = center_z

        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "panda_link0"

        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        pub.publish(pose_msg)
        rate.sleep()

def publish_forward_trajectory(distance=0.05, duration=5.0, freq=100, axis='x'):
    """
    Moves the EE forward slowly in a straight line along the chosen axis.
    axis: 'x', 'y', or 'z'
    """
    rospy.loginfo(f"Starting forward trajectory along {axis}-axis...")

    pub = rospy.Publisher("/joint_impedance_controller/desired_pose", PoseStamped, queue_size=1)
    rate = rospy.Rate(freq)

    start_pos = current_position.copy()
    quat = current_quaternion
    start_time = rospy.Time.now().to_sec()

    while not rospy.is_shutdown():
        t = rospy.Time.now().to_sec() - start_time
        if t > duration:
            rospy.loginfo("Forward trajectory completed.")
            break

        # Linear interpolation from 0 to distance
        progress = t / duration
        delta = distance * progress

        x, y, z = start_pos
        if axis == 'x':
            x = start_pos[0] + delta
        elif axis == 'y':
            y = start_pos[1] + delta
        elif axis == 'z':
            z = start_pos[2] + delta

        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "panda_link0"

        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        pub.publish(pose_msg)
        rate.sleep()

def main():
    rospy.init_node("circular_trajectory_publisher", anonymous=True)
    rospy.Subscriber("/franka_state_controller/franka_states", FrankaState, franka_state_callback)

    rospy.loginfo("Waiting for the first valid robot state...")
    while not rospy.is_shutdown() and current_position is None:
        rospy.sleep(0.1)

    rospy.loginfo(f"Current EE Pose:\n Position: {current_position}\n Quaternion: {current_quaternion}")
    # publish_circular_trajectory(radius=0.05, duration=10.0, freq=100)
    publish_forward_trajectory(distance=0.05, duration=5.0, freq=100, axis='x')

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass