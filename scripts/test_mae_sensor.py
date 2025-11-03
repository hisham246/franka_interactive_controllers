#!/usr/bin/env python3

import rospy
import csv
import os
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
from scipy.spatial.transform import Rotation as R
import threading
from datetime import datetime

# initial_pose = [0.5]


class EEPoseListener:
    """Subscribe to EE pose and keep the latest Pose."""

    def __init__(self, topic='/franka_state_controller/ee_pose'):
        self._lock = threading.Lock()
        self._pose = None
        self._ready = threading.Event()
        self._sub = rospy.Subscriber(topic, Pose, self._callback)

    def _callback(self, msg):
        with self._lock:
            self._pose = msg
        self._ready.set()

    def wait_for_initial_pose(self, timeout=5.0):
        """Block until first pose is received or timeout."""
        if not self._ready.wait(timeout):
            raise TimeoutError("Timed out waiting for initial EE pose.")
        with self._lock:
            return self._pose

    def get_latest_pose(self):
        """Return the latest Pose (or None if not yet received)."""
        with self._lock:
            return self._pose


def pose_to_array(pose_msg):
    """
    Convert a geometry_msgs/Pose to [x, y, z, rx, ry, rz] (rotvec).
    """
    pos = pose_msg.position
    quat = pose_msg.orientation
    rotvec = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_rotvec()
    return np.array([pos.x, pos.y, pos.z, *rotvec])


def make_pose_stamped_from_pose(pose, frame_id="panda_link0"):
    """Wrap a geometry_msgs/Pose into PoseStamped with current time."""
    ps = PoseStamped()
    ps.header.stamp = rospy.Time.now()
    ps.header.frame_id = frame_id
    ps.pose = pose
    return ps


def main():
    rospy.init_node("panda_z_displacement_logger")

    # Parameters
    state_topic   = rospy.get_param("~state_topic", "/franka_state_controller/ee_pose")
    command_topic = rospy.get_param("~command_topic", "/cartesian_pose_impedance_controller/desired_pose")
    frame_id      = rospy.get_param("~frame_id", "panda_link0")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    duration      = rospy.get_param("~duration", 60.0)       # seconds
    frequency     = rospy.get_param("~frequency", 100.0)      # Hz
    z_speed       = rospy.get_param("~z_speed", 0.01)        # m/s (positive = up)
    output_csv    = rospy.get_param("~output_csv", f"ee_z_displacement_log_{timestamp}.csv")

    rospy.loginfo(f"Will move along +Z at {z_speed} m/s for {duration} s.")

    # Set up EE pose listener
    ee_listener = EEPoseListener(topic=state_topic)
    rospy.loginfo(f"Waiting for initial EE pose on {state_topic}...")
    initial_pose = ee_listener.wait_for_initial_pose(timeout=10.0)
    rospy.loginfo("Initial EE pose received.")

    # Copy initial pose to build command pose
    cmd_pose = Pose()
    cmd_pose.position.x = initial_pose.position.x
    cmd_pose.position.y = initial_pose.position.y
    cmd_pose.position.z = initial_pose.position.z
    cmd_pose.orientation = initial_pose.orientation  # keep orientation fixed

    # Publisher for desired pose
    pose_pub = rospy.Publisher(command_topic, PoseStamped, queue_size=1)
    rospy.loginfo(f"Publishing desired poses on {command_topic}")

    # Prepare CSV logging
    output_csv = os.path.abspath(output_csv)
    rospy.loginfo(f"Logging EE pose to: {output_csv}")
    csv_file = open(output_csv, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    # CSV header
    csv_writer.writerow(["t_rel_s", "x", "y", "z", "rx", "ry", "rz"])

    rate = rospy.Rate(frequency)
    start_time = rospy.Time.now().to_sec()

    try:
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            t_rel = now - start_time

            # if t_rel < duration:
            #     # Compute desired Z based on elapsed time (linear motion)
            #     cmd_pose.position.z = 0.05 * np.sin(t_rel) - 0.08
            #     # initial_pose.position.z - z_speed * t_rel
            # else:
            #     rospy.loginfo("Duration reached, stopping motion.")
            #     break

            # Publish desired pose
            # pose_stamped = make_pose_stamped_from_pose(cmd_pose, frame_id=frame_id)
            # pose_pub.publish(pose_stamped)

            # Get latest actual EE pose and log it
            latest_pose = ee_listener.get_latest_pose()
            if latest_pose is not None:
                arr = pose_to_array(latest_pose)
                csv_writer.writerow([t_rel, *arr])

            rate.sleep()

        # Optionally, hold final pose for a bit
        rospy.loginfo("Holding final pose for 1 second.")
        hold_end = rospy.Time.now().to_sec() + 1.0
        while not rospy.is_shutdown() and rospy.Time.now().to_sec() < hold_end:
            pose_stamped = make_pose_stamped_from_pose(cmd_pose, frame_id=frame_id)
            pose_pub.publish(pose_stamped)
            rate.sleep()

    finally:
        csv_file.close()
        rospy.loginfo("CSV file closed. Node exiting.")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
