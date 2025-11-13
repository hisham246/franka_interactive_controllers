#!/usr/bin/env python3
"""
High-frequency pose + Cartesian velocity recorder for Franka Emika Panda robot.
- Subscribes to the EE pose topic (Pose) and FrankaState.
- Saves Cartesian pose and O_dP_EE_d / O_dP_EE_c (desired/commanded EE twist) with timestamps.
"""

import rospy
import numpy as np
from geometry_msgs.msg import Pose
from franka_msgs.msg import FrankaState
from sensor_msgs.msg import JointState
from datetime import datetime
import os
import threading


class PoseRecorder:
    def __init__(self):
        rospy.init_node('pose_recorder', anonymous=True)
        
        # Hard-coded parameters (adjust as you like)
        self.output_dir = "/home/hisham246/uwaterloo/panda_ws/src/franka_interactive_controllers/robot_demos"
        self.output_filename = "franka_ee_pose"
        self.buffer_size = 10000  # Number of samples to buffer
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            rospy.loginfo(f"Created output directory: {self.output_dir}")
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(
            self.output_dir, 
            f"{self.output_filename}_{timestamp}.csv"
        )
        
        # Data buffer (thread-safe)
        self.data_buffer = []
        self.buffer_lock = threading.Lock()
        self.recording = True
        
        # Statistics
        self.sample_count = 0
        self.start_time = None

        # Latest Franka state fields we care about
        self.latest_O_dP_EE_d = None  # 6D desired twist
        self.latest_O_dP_EE_c = None  # 6D commanded twist
        
        # Initialize CSV file with header
        self._initialize_file()
        
        # Subscribe to Franka end-effector pose topic (Pose)
        self.pose_sub = rospy.Subscriber(
            '/franka_state_controller/ee_pose',
            Pose,
            self.pose_callback,
            queue_size=1000,
            tcp_nodelay=True
        )

        # Subscribe to FrankaState for O_dP_EE_d / O_dP_EE_c
        self.state_sub = rospy.Subscriber(
            '/franka_state_controller/franka_states',  # adjust topic name if needed
            FrankaState,
            self.state_callback,
            queue_size=1000,
            tcp_nodelay=True
        )

        # Subscribe to gripper joint states (lower frequency)
        self.gripper_sub = rospy.Subscriber(
            '/franka_gripper/joint_states',
            JointState,
            self.gripper_callback,
            queue_size=100,
            tcp_nodelay=True
        )
        
        # Start background thread for writing data
        self.write_thread = threading.Thread(target=self._write_loop)
        self.write_thread.daemon = True
        self.write_thread.start()
        
        rospy.loginfo(f"Pose recorder initialized. Saving to: {self.filepath}")
        rospy.loginfo("Subscribed to: /franka_state_controller/ee_pose (Pose)")
        rospy.loginfo("Subscribed to: /franka_state_controller/franka_state (FrankaState)")
        rospy.loginfo("Subscribed to: /franka_gripper/joint_states (JointState)")
        rospy.loginfo("Recording started...")
    
    def _initialize_file(self):
        """Initialize CSV file with header."""
        with open(self.filepath, 'w') as f:
            # Time
            header = "ros_time_sec,ros_time_nsec,"
            # Pose
            header += "pos_x,pos_y,pos_z,"
            header += "quat_x,quat_y,quat_z,quat_w,"
            # Gripper
            header += "gripper_pos_f1,gripper_pos_f2,"
            # O_dP_EE_d (desired EE twist)
            header += "O_dP_EE_d_vx,O_dP_EE_d_vy,O_dP_EE_d_vz,"
            header += "O_dP_EE_d_wx,O_dP_EE_d_wy,O_dP_EE_d_wz,"
            # O_dP_EE_c (commanded EE twist)
            header += "O_dP_EE_c_vx,O_dP_EE_c_vy,O_dP_EE_c_vz,"
            header += "O_dP_EE_c_wx,O_dP_EE_c_wy,O_dP_EE_c_wz\n"
            f.write(header)

    def state_callback(self, msg):
        """Callback for FrankaState messages, store latest twists."""
        # These are float64[6] arrays in FrankaState
        self.latest_O_dP_EE_d = np.array(msg.O_dP_EE_d, dtype=float)
        self.latest_O_dP_EE_c = np.array(msg.O_dP_EE_c, dtype=float)

    def gripper_callback(self, msg):
        """Callback for gripper joint states; store latest finger positions."""
        if len(msg.position) >= 2:
            # position[0] = finger 1, position[1] = finger 2
            self.latest_gripper_pos = np.array(msg.position[:2], dtype=float)
        else:
            # If something weird happens, store NaNs
            self.latest_gripper_pos = np.full(2, np.nan)

    def pose_callback(self, msg):
        """High-frequency callback for pose messages (primary trigger for logging)."""
        if not self.recording:
            return
        
        # Get system time for accurate timestamping
        now = rospy.Time.now()
        
        if self.start_time is None:
            self.start_time = now
        
        # Pose
        pos = msg.position
        quat = msg.orientation

        # Use latest FrankaState if available, otherwise NaNs
        if self.latest_O_dP_EE_d is not None:
            O_dP_EE_d = self.latest_O_dP_EE_d
        else:
            O_dP_EE_d = np.full(6, np.nan)

        if self.latest_O_dP_EE_c is not None:
            O_dP_EE_c = self.latest_O_dP_EE_c
        else:
            O_dP_EE_c = np.full(6, np.nan)

        # Use latest gripper state (sample-and-hold), otherwise NaNs
        if self.latest_gripper_pos is not None:
            gpos = self.latest_gripper_pos
        else:
            gpos = np.full(2, np.nan)
        
        # Extract data row
        data_row = [
            float(now.secs),
            float(now.nsecs),
            float(pos.x),
            float(pos.y),
            float(pos.z),
            float(quat.x),
            float(quat.y),
            float(quat.z),
            float(quat.w),
            # Gripper
            float(gpos[0]),
            float(gpos[1]),
            # desired twist
            float(O_dP_EE_d[0]),
            float(O_dP_EE_d[1]),
            float(O_dP_EE_d[2]),
            float(O_dP_EE_d[3]),
            float(O_dP_EE_d[4]),
            float(O_dP_EE_d[5]),
            # commanded twist
            float(O_dP_EE_c[0]),
            float(O_dP_EE_c[1]),
            float(O_dP_EE_c[2]),
            float(O_dP_EE_c[3]),
            float(O_dP_EE_c[4]),
            float(O_dP_EE_c[5]),
        ]
        
        # Add to buffer (thread-safe)
        with self.buffer_lock:
            self.data_buffer.append(data_row)
            self.sample_count += 1
            
            # If buffer is full, write immediately
            if len(self.data_buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def _flush_buffer(self):
        """Write buffer contents to file (must be called with lock held)."""
        if not self.data_buffer:
            return
        
        # Convert to numpy array for efficient writing
        data_array = np.array(self.data_buffer, dtype=float)
        
        # Append to CSV
        with open(self.filepath, 'a') as f:
            np.savetxt(f, data_array, delimiter=',', fmt='%.9f')
        
        # Clear buffer
        self.data_buffer.clear()
    
    def _write_loop(self):
        """Background thread for periodic buffer flushing."""
        rate = rospy.Rate(10)  # Flush buffer 10 times per second
        
        while not rospy.is_shutdown() and self.recording:
            with self.buffer_lock:
                if self.data_buffer:
                    self._flush_buffer()
            rate.sleep()
    
    def get_statistics(self):
        """Get recording statistics."""
        if self.start_time is None:
            return None
        
        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        avg_freq = self.sample_count / elapsed if elapsed > 0 else 0
        
        return {
            'sample_count': self.sample_count,
            'elapsed_time': elapsed,
            'average_frequency': avg_freq,
            'filepath': self.filepath
        }
    
    def stop_recording(self):
        """Stop recording and flush remaining data."""
        rospy.loginfo("Stopping recording...")
        self.recording = False
        
        # Final flush
        with self.buffer_lock:
            self._flush_buffer()
        
        # Print statistics
        stats = self.get_statistics()
        if stats:
            rospy.loginfo("=" * 60)
            rospy.loginfo("Recording Statistics:")
            rospy.loginfo(f"  Total samples: {stats['sample_count']}")
            rospy.loginfo(f"  Elapsed time: {stats['elapsed_time']:.2f} seconds")
            rospy.loginfo(f"  Average frequency: {stats['average_frequency']:.2f} Hz")
            rospy.loginfo(f"  Data saved to: {stats['filepath']}")
            rospy.loginfo("=" * 60)
    
    def run(self):
        """Main run loop."""
        rospy.on_shutdown(self.stop_recording)
        rospy.spin()


if __name__ == '__main__':
    try:
        recorder = PoseRecorder()
        recorder.run()
    except rospy.ROSInterruptException:
        pass