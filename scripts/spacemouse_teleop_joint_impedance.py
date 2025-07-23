# !/usr/bin/env python3

import rospy
import time
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
import threading
import sys
import os

from dynamic_reconfigure.client import Client as DynamicReconfigureClient

sys.path.append(os.path.join(os.path.dirname(__file__)))

from policy_utils.spacemouse_shared_memory import Spacemouse
from policy_utils.keystroke_counter import KeystrokeCounter, KeyCode
from policy_utils.precise_sleep import precise_wait


class InitialPoseListener:
    def __init__(self, topic='/franka_state_controller/ee_pose'):
        self.pose = None
        self.ready = threading.Event()
        self.sub = rospy.Subscriber(topic, Pose, self.callback)

    def callback(self, msg):
        pos = msg.position
        quat = msg.orientation
        rotvec = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_rotvec()
        self.pose = np.array([pos.x, pos.y, pos.z, *rotvec])
        self.ready.set()

    def wait_for_pose(self, timeout=5.0):
        if not self.ready.wait(timeout):
            raise TimeoutError("Timed out waiting for initial EE pose.")
        return self.pose


def publish_pose(publisher, pose):
    """Publish a 4x4 homogeneous transform as Float64MultiArray (column-major for Franka)."""
    pos = pose[:3]
    rotvec = pose[3:]
    quat = R.from_rotvec(rotvec).as_quat()
    rotm = R.from_quat(quat).as_matrix()

    # Build homogeneous transform
    T = np.eye(4)
    T[:3, :3] = rotm
    T[:3, 3] = pos

    msg = Float64MultiArray()
    msg.data = T.T.ravel().tolist()
    print(f"Publishing pose: {msg.data}")
    publisher.publish(msg)


def main():
    rospy.init_node('teleop_spacemouse_ros_node')

    # Parameters
    frequency = rospy.get_param('~frequency', 10.0)
    max_pos_speed = rospy.get_param('~max_pos_speed', 0.25)
    max_rot_speed = rospy.get_param('~max_rot_speed', 0.6)
    gripper_speed = rospy.get_param('~gripper_speed', 0.05)
    max_gripper_width = 0.1
    gripper_width = 0.08

    dt = 1.0 / frequency
    command_latency = dt / 2

    # Publisher for Cartesian pose
    pose_pub = rospy.Publisher('/joint_impedance_controller/desired_pose', Float64MultiArray, queue_size=1)
    rospy.loginfo("ROS publisher ready.")


    # Wait for initial robot pose
    rospy.loginfo("Waiting for /franka_state_controller/ee_pose...")
    pose_listener = InitialPoseListener()
    target_pose = pose_listener.wait_for_pose()
    rospy.loginfo(f"Initial pose received: {target_pose}")

    with SharedMemoryManager() as shm_manager, \
         Spacemouse(shm_manager=shm_manager) as sm, \
         KeystrokeCounter() as key_counter:

        rospy.loginfo("SpaceMouse ready. Press 'q' to quit.")

        t_start = time.monotonic()
        iter_idx = 0
        stop = False

        while not rospy.is_shutdown() and not stop:
            t_cycle_end = t_start + (iter_idx + 1) * dt
            t_sample = t_cycle_end - command_latency

            press_events = key_counter.get_press_events()
            for key_stroke in press_events:
                if key_stroke == KeyCode(char='q'):
                    stop = True

            precise_wait(t_sample)

            sm_state = sm.get_motion_state_transformed()
            print(f"[DEBUG] Raw SpaceMouse state: {sm_state}")
            dpos = sm_state[:3] * (max_pos_speed / frequency)
            drot_xyz = sm_state[3:] * (max_rot_speed / frequency)
            drot = st.Rotation.from_euler('xyz', drot_xyz)

            target_pose[:3] += dpos
            target_pose[3:] = (drot * st.Rotation.from_rotvec(target_pose[3:])).as_rotvec()
            target_pose[2] = max(target_pose[2], 0.05)

            print(f"[DEBUG] dpos: {dpos}, drot_xyz: {drot_xyz}")
            print(f"[DEBUG] Updated target pose: {target_pose}")

            dwidth = 0
            if sm.is_button_pressed(0):
                dwidth = -gripper_speed / frequency
            if sm.is_button_pressed(1):
                dwidth = gripper_speed / frequency
            gripper_width = np.clip(gripper_width + dwidth, 0.0, max_gripper_width)

            publish_pose(pose_pub, target_pose)

            print(f"[Iter {iter_idx}] Pose: {target_pose[:3]}, Gripper width: {gripper_width:.3f}", end='\r')
            precise_wait(t_cycle_end)
            iter_idx += 1


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass