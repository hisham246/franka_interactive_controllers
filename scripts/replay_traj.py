#!/usr/bin/env python3
import rospy
import pandas as pd
import numpy as np
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import time as pytime

class PoseReplayer:
    def __init__(self,
                 csv_path,
                 topic="/hybrid_joint_impedance_controller/desired_pose",
                 frame_id="panda_link0",
                 rate=100.0,
                 repeat=False,
                 mode="absolute",      # "absolute" or "relative"
                 row_index=None,       # None = stream all; int = single-row test
                 hold_seconds=1.0):    # how long to hold a single row (if row_index is set)
        self.csv_path = csv_path
        self.topic = topic
        self.frame_id = frame_id
        self.rate = rate
        self.repeat = repeat
        self.mode = mode
        self.row_index = row_index
        self.hold_seconds = hold_seconds

        # live robot state (from O_T_EE)
        self.current_pos = None     # np.array(3,)
        self.current_quat = None    # np.array(4,) [x,y,z,w]

        self.pose_pub = rospy.Publisher(self.topic, PoseStamped, queue_size=10)

        # Load CSV: must have x,y,z,rx,ry,rz and optional time
        self.df = pd.read_csv(self.csv_path)
        self.has_time = "time" in self.df.columns

        # Cache the first CSV pose (for relative mode)
        self.csv0_pos = self.df.loc[self.df.index[0], ["ee_pos_0","ee_pos_1","ee_pos_2"]].values.astype(float)
        self.csv0_rv  = self.df.loc[self.df.index[0], ["ee_rot_0","ee_rot_1","ee_rot_2"]].values.astype(float)
        self.csv0_quat = R.from_rotvec(self.csv0_rv).as_quat()

        # Subscriber for live state (like your circle node)
        rospy.Subscriber("/franka_state_controller/franka_states", FrankaState, self._state_cb)

    def _state_cb(self, msg: FrankaState):
        # O_T_EE (column-major 4x4)
        T = np.array(msg.O_T_EE).reshape(4,4, order='F')
        self.current_pos  = T[:3, 3].copy()
        self.current_quat = R.from_matrix(T[:3, :3]).as_quat()  # [x,y,z,w]

    def _wait_for_state(self):
        rospy.loginfo("Waiting for first valid /franka_state_controller/franka_states...")
        r = rospy.Rate(100)
        while not rospy.is_shutdown() and (self.current_pos is None or self.current_quat is None):
            r.sleep()
        rospy.loginfo("Got live EE pose.")
        rospy.loginfo(f"  pos: {self.current_pos}")
        rospy.loginfo(f"  quat[x,y,z,w]: {self.current_quat}")

    # def update_desired_ee_pose(self, pose_xyz_rxryrz):
    #     """
    #     EXACT same style you used before: publish PoseStamped
    #     converting rotvec -> quaternion inside this function.
    #     pose_xyz_rxryrz = [x,y,z, rx,ry,rz]
    #     """
    #     msg = PoseStamped()
    #     msg.header.stamp = rospy.Time.now()
    #     msg.header.frame_id = self.frame_id

    #     x, y, z, rx, ry, rz = pose_xyz_rxryrz
    #     msg.pose.position.x = x
    #     msg.pose.position.y = y
    #     msg.pose.position.z = z

    #     quat = R.from_rotvec([rx, ry, rz]).as_quat()  # [x,y,z,w]
    #     msg.pose.orientation.x = quat[0]
    #     msg.pose.orientation.y = quat[1]
    #     msg.pose.orientation.z = quat[2]
    #     msg.pose.orientation.w = quat[3]

    #     self.pose_pub.publish(msg)

    def update_desired_ee_pose(self, pose_xyz_rxryrz):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id

        x, y, z = pose_xyz_rxryrz[:3]
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z

        # --- FIXED ORIENTATION ---
        if self.current_quat is not None:
            # keep robot's current orientation
            quat = self.current_quat
        else:
            # fallback: identity quaternion
            quat = [0.0, 0.0, 0.0, 1.0]

        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        # -------------------------

        self.pose_pub.publish(msg)

    def _absolute_pose_from_row(self, row):
        """CSV is assumed already in frame_id; use directly."""
        xyz = row[["ee_pos_0","ee_pos_1","ee_pos_2"]].values.astype(float)
        rv  = row[["ee_rot_0","ee_rot_1","ee_rot_2"]].values.astype(float)
        return np.concatenate([xyz, rv])

    def _relative_pose_from_row(self, row):
        """
        Re-anchor CSV trajectory to the current live pose:
        desired_pos = current_pos + (csv_pos - csv0_pos)
        q_rel = q_csv * inv(q_csv0)
        desired_quat = q_rel * current_quat
        Then convert to rotvec for publishing (to use your exact update function).
        """
        assert self.current_pos is not None and self.current_quat is not None
        csv_pos = row[["ee_pos_0","ee_pos_1","ee_pos_2"]].values.astype(float)
        csv_rv  = row[["ee_rot_0","ee_rot_1","ee_rot_2"]].values.astype(float)
        q_csv   = R.from_rotvec(csv_rv).as_quat()

        # position
        dpos = csv_pos - self.csv0_pos
        out_pos = self.current_pos + dpos

        # orientation
        q_rel = R.from_quat(q_csv) * R.from_quat(self.csv0_quat).inv()
        q_out = (q_rel * R.from_quat(self.current_quat)).as_quat()
        rv_out = R.from_quat(q_out).as_rotvec()

        return np.concatenate([out_pos, rv_out])

    def _row_to_pose(self, row):
        if self.mode.lower() == "relative":
            return self._relative_pose_from_row(row)
        else:
            return self._absolute_pose_from_row(row)

    def run(self):
        self._wait_for_state()

        rate = rospy.Rate(self.rate)
        rospy.loginfo(f"Replayer started: mode={self.mode}, csv={self.csv_path}")

        while not rospy.is_shutdown():
            if self.row_index is not None:
                # Single-row test, hold for hold_seconds at rate
                idx = self.row_index
                if idx < 0:
                    idx = len(self.df) + idx  # support -1, -2, ...
                row = self.df.iloc[idx]
                pose_xyzrv = self._row_to_pose(row)

                t_end = rospy.Time.now().to_sec() + float(self.hold_seconds)
                rospy.loginfo(f"Holding row {self.row_index} for {self.hold_seconds}s at {self.rate} Hz")
                while not rospy.is_shutdown() and rospy.Time.now().to_sec() < t_end:
                    self.update_desired_ee_pose(pose_xyzrv)
                    rate.sleep()

            else:
                # Stream whole CSV (time-synchronized if 'time' present)
                start_wall = pytime.time()
                t0 = self.df["time"].iloc[0] if self.has_time else 0.0

                for _, row in self.df.iterrows():
                    if rospy.is_shutdown():
                        break
                    pose_xyzrv = self._row_to_pose(row)

                    if self.has_time:
                        target_dt = row["time"] - t0
                        now = pytime.time()
                        sleep_s = (start_wall + target_dt) - now
                        if sleep_s > 0:
                            rospy.sleep(sleep_s)
                    else:
                        rate.sleep()

                    self.update_desired_ee_pose(pose_xyzrv)

            if not self.repeat:
                break

            rospy.loginfo("Replay finished; looping again due to repeat=True")

def main():
    rospy.init_node("pose_csv_replayer_with_state")

    csv_path = rospy.get_param("~csv_path", "/home/hisham246/uwaterloo/surface_wiping_unet/policy_actions_20250828_231539.csv")
    topic    = rospy.get_param("~topic", "/hybrid_joint_impedance_controller/desired_pose")
    frame_id = rospy.get_param("~frame_id", "panda_link0")
    rate     = float(rospy.get_param("~rate", 10.0))
    repeat   = bool(rospy.get_param("~repeat", False))

    # "absolute" uses CSV poses as-is; "relative" anchors the CSV to current EE pose
    mode = rospy.get_param("~mode", "absolute")  # or "relative"

    # If you want a single-row test: set ~row_index, else leave unset or None to stream all
    row_index = rospy.get_param("~row_index", None)  # e.g., -1 for last row, 0 for first
    if isinstance(row_index, str) and row_index.strip() == "":
        row_index = None
    if row_index is not None:
        row_index = int(row_index)

    hold_sec = float(rospy.get_param("~hold_seconds", 1.0))

    replayer = PoseReplayer(csv_path, topic, frame_id, rate, repeat, mode, row_index, hold_sec)
    replayer.run()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
