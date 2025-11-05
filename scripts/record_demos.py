#!/usr/bin/env python3
"""
High-frequency pose recorder for Franka Emika Panda robot.
Subscribes to the EE pose topic and saves Cartesian poses with timestamps.
"""

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from datetime import datetime
import os
import threading


class PoseRecorder:
    def __init__(self):
        rospy.init_node('pose_recorder', anonymous=True)
        
        # Get parameters
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
        
        # Initialize CSV file with header
        self._initialize_file()
        
        # Subscribe to Franka end-effector pose topic
        self.pose_sub = rospy.Subscriber(
            '/franka_state_controller/ee_pose',
            PoseStamped,
            self.pose_callback,
            queue_size=1000,  # Large queue to handle 1kHz
            tcp_nodelay=True
        )
        
        # Start background thread for writing data
        self.write_thread = threading.Thread(target=self._write_loop)
        self.write_thread.daemon = True
        self.write_thread.start()
        
        rospy.loginfo(f"Pose recorder initialized. Saving to: {self.filepath}")
        rospy.loginfo("Subscribed to: /franka_state_controller/ee_pose")
        rospy.loginfo("Recording started...")
    
    def _initialize_file(self):
        """Initialize CSV file with header."""
        with open(self.filepath, 'w') as f:
            header = "timestamp_sec,timestamp_nsec,ros_time_sec,ros_time_nsec,"
            header += "pos_x,pos_y,pos_z,"
            header += "quat_x,quat_y,quat_z,quat_w\n"
            f.write(header)
    
    def pose_callback(self, msg):
        """High-frequency callback for pose messages."""
        if not self.recording:
            return
        
        # Get system time for accurate timestamping
        now = rospy.Time.now()
        
        if self.start_time is None:
            self.start_time = now
        
        # Extract data
        data_row = [
            now.secs,
            now.nsecs,
            msg.header.stamp.secs,
            msg.header.stamp.nsecs,
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
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
        data_array = np.array(self.data_buffer)
        
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