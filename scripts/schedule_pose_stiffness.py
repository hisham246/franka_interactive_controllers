#!/usr/bin/env python3
import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped, Quaternion
from franka_msgs.msg import FrankaState
from tf.transformations import quaternion_from_matrix
from dynamic_reconfigure.client import Client as DynamicReconfigureClient


class TrajectoryReplayer:
    def __init__(self):
        # You can turn this into a param if you like
        self.csv_path = rospy.get_param(
            "~csv_path",
            "/home/hisham246/uwaterloo/panda_ws/src/franka_interactive_controllers/robot_demos/stiffness/demo_01_pos_vel_stiffness.csv",
        )
        # self.csv_path = rospy.get_param(
        #     "~csv_path",
        #     "/home/robohub/hisham/panda_ws/src/franka_interactive_controllers/robot_demos/stiffness/demo_01_pos_vel_stiffness.csv",
        # )
        
        self.rate_hz = rospy.get_param("~rate_hz", 1200.0)  # playback frequency
        self.wait_before_start = rospy.get_param("~wait_before_start", 3.0)  # seconds

        # Frame id for PoseStamped
        self.frame_id = rospy.get_param("~frame_id", "panda_link0")

        # Topic from which to read the current Franka state once
        self.ee_state_topic = rospy.get_param(
            "~ee_state_topic", "/franka_state_controller/franka_states"
        )

        # Dynamic reconfigure server
        self.dyn_server_name = rospy.get_param(
            "~dyn_server_name",
            "/hybrid_joint_impedance_controller/dynamic_reconfigure_compliance_param_node",
        )

        self.kx_param_name = rospy.get_param("~kx_param_name", "translational_stiffness_x")
        self.ky_param_name = rospy.get_param("~ky_param_name", "translational_stiffness_y")
        self.kz_param_name = rospy.get_param("~kz_param_name", "translational_stiffness_z")

        if not self.csv_path:
            rospy.logerr("Parameter ~csv_path is required!")
            raise RuntimeError("Missing ~csv_path parameter")

        rospy.loginfo(f"Loading trajectory CSV from: {self.csv_path}")
        self.positions, self.stiffness = self._load_positions_from_csv(self.csv_path)
        # rospy.loginfo(f"Loaded {self.positions.shape[0]} trajectory points.")

        # Publisher to your hybrid joint impedance controller
        # IMPORTANT: advertise as PoseStamped
        self.pub = rospy.Publisher(
            "/hybrid_joint_impedance_controller/desired_pose",
            PoseStamped,
            queue_size=10,
        )

        # Dynamic reconfigure client for stiffness updates
        rospy.loginfo(f"Creating dynamic_reconfigure client for {self.dyn_server_name}")
        self.dyn_client = DynamicReconfigureClient(self.dyn_server_name, timeout=5.0)

        # Grab current orientation once from FrankaState
        self.initial_position, self.initial_orientation = self._get_initial_pose_from_franka_state()
        rospy.loginfo(
            "Initial orientation (from O_T_EE): "
            f"[{self.initial_orientation.x}, "
            f"{self.initial_orientation.y}, "
            f"{self.initial_orientation.z}, "
            f"{self.initial_orientation.w}]"
        )

    def _load_positions_from_csv(self, csv_path):
        """
        Expect CSV with header:
        x,y,z,vx,vy,vz,Kx,Ky,Kz

        We only use x,y,z for now.
        """
        # Skip header line ("x,y,z,...")
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

        # if data.ndim == 1:
        #     # Single row -> make it (1, N)
        #     data = data[None, :]

        # if data.shape[1] < 3:
        #     raise ValueError(
        #         f"Expected at least 3 columns (x,y,z), got shape {data.shape}"
        #     )

        positions = data[:, 0:3]  # x,y,z
        stiffness = data[:, 6:9]
        return positions, stiffness

    def _get_initial_orientation_from_franka_state(self):
        """
        Wait for one FrankaState message and extract the EE orientation
        from the O_T_EE 4x4 homogeneous transform.

        O_T_EE is a 16-element array in row-major order:
            [ R00 R01 R02 p0
              R10 R11 R12 p1
              R20 R21 R22 p2
              0   0   0   1  ]
        """
        rospy.loginfo(
            f"Waiting for Franka state on topic '{self.ee_state_topic}'..."
        )
        msg = rospy.wait_for_message(self.ee_state_topic, FrankaState)
        rospy.loginfo("Received FrankaState message.")

        # Convert O_T_EE (list of 16 floats) to 4x4 matrix
        T = np.array(msg.O_T_EE).reshape(4, 4)

        # Use tf.transformations to get quaternion from the 4x4 transform
        q = quaternion_from_matrix(T)  # [x, y, z, w]

        q_msg = Quaternion()
        q_msg.x = q[0]
        q_msg.y = q[1]
        q_msg.z = q[2]
        q_msg.w = q[3]

        return q_msg
    
    def _get_initial_pose_from_franka_state(self):
        """
        Wait for one FrankaState message and extract the EE pose
        (position + orientation) from the O_T_EE 4x4 homogeneous transform.
        """
        rospy.loginfo(
            f"Waiting for Franka state on topic '{self.ee_state_topic}'..."
        )
        msg = rospy.wait_for_message(self.ee_state_topic, FrankaState)
        rospy.loginfo("Received FrankaState message.")

        # Convert O_T_EE (list of 16 floats) to 4x4 matrix
        T = np.array(msg.O_T_EE).reshape(4, 4)

        # Position is the translation part
        p = T[0:3, 3].copy()

        # Use tf.transformations to get quaternion from the 4x4 transform
        q = quaternion_from_matrix(T)  # [x, y, z, w]

        q_msg = Quaternion()
        q_msg.x = q[0]
        q_msg.y = q[1]
        q_msg.z = q[2]
        q_msg.w = q[3]

        return p, q_msg
    
    def _slow_transition_to_first_point(self, rate):
        """Slowly move from current position to first trajectory point."""
        transition_time = rospy.get_param("~transition_time", 3.0)  # seconds
        num_steps = int(transition_time * self.rate_hz)
        
        # Interpolate from initial position to first point
        for i in range(num_steps):
            if rospy.is_shutdown():
                break
            
            alpha = float(i) / num_steps  # 0 to 1
            interp_pos = self.initial_position + alpha * (self.positions[0] - self.initial_position)
            
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = self.frame_id
            pose_msg.pose.position.x = float(interp_pos[0])
            pose_msg.pose.position.y = float(interp_pos[1])
            pose_msg.pose.position.z = float(interp_pos[2])
            pose_msg.pose.orientation = self.initial_orientation
            
            self.pub.publish(pose_msg)
            rate.sleep()

    def _update_stiffness(self, k_vec):
        """
        Send a dynamic_reconfigure request using the stiffness vector [Kx, Ky, Kz].
        """
        cfg = {
            self.kx_param_name: float(k_vec[0]),
            self.ky_param_name: float(k_vec[1]),
            self.kz_param_name: float(k_vec[2]),
        }
        try:
            self.dyn_client.update_configuration(cfg)
        except Exception as e:
            # You can make this rospy.logerr once if you like; keeping it warn to not spam
            rospy.logwarn(f"Failed to update stiffness via dynamic_reconfigure: {e}")

    def run(self):
        rospy.loginfo(
            f"Waiting {self.wait_before_start} seconds before starting playback..."
        )
        rospy.sleep(self.wait_before_start)

        rate = rospy.Rate(self.rate_hz)

        # self._slow_transition_to_first_point(rate)

        # # 1) Stabilize at current pose with soft stiffness
        # self._stabilize_at_current_pose(rate)

        # # 2) Safe motion to first trajectory point
        # self._go_to_start_pose_safely(rate)

        # # 3) Ramp stiffness from soft -> first sample stiffness
        # self._ramp_stiffness_to_first_sample(rate)

        rospy.loginfo("Starting trajectory replay (one shot).")

        # Use the same orientation for the whole trajectory
        ori = self.initial_orientation

        for idx, p in enumerate(self.positions):
            if rospy.is_shutdown():
                break

            # Update stiffness for this step
            k_vec = self.stiffness[idx]
            self._update_stiffness(k_vec)

            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = self.frame_id

            pose_msg.pose.position.x = float(p[0])
            pose_msg.pose.position.y = float(p[1])
            pose_msg.pose.position.z = float(p[2])

            pose_msg.pose.orientation.x = ori.x
            pose_msg.pose.orientation.y = ori.y
            pose_msg.pose.orientation.z = ori.z
            pose_msg.pose.orientation.w = ori.w

            self.pub.publish(pose_msg)
            rate.sleep()

        # Hold the last pose for a bit so the controller stabilizes
        rospy.loginfo("Finished trajectory replay. Holding final pose briefly.")
        final_pose = PoseStamped()
        final_pose.header.frame_id = self.frame_id
        final_pose.pose.position.x = float(self.positions[-1, 0])
        final_pose.pose.position.y = float(self.positions[-1, 1])
        final_pose.pose.position.z = float(self.positions[-1, 2])
        final_pose.pose.orientation = ori

        hold_time = rospy.get_param("~hold_final_pose_time", 2.0)
        end_time = rospy.Time.now() + rospy.Duration.from_sec(hold_time)
        while not rospy.is_shutdown() and rospy.Time.now() < end_time:
            final_pose.header.stamp = rospy.Time.now()
            self.pub.publish(final_pose)
            rate.sleep()

        rospy.loginfo("Trajectory replay node finished.")


def main():
    rospy.init_node("trajectory_replayer")
    try:
        node = TrajectoryReplayer()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()