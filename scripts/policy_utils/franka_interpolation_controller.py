import rospy
import numpy as np
import enum
import time
from policy_utils.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from policy_utils.shared_memory_ring_buffer import SharedMemoryRingBuffer
from policy_utils.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from policy_utils.precise_sleep import precise_wait
from multiprocessing.managers import SharedMemoryManager
import pathlib
import multiprocessing as mp
import traceback
from geometry_msgs.msg import PoseStamped, Pose, Twist
from sensor_msgs.msg import JointState
from franka_msgs.msg import FrankaState
from dynamic_reconfigure.client import Client as DynClient
from scipy.spatial.transform import Rotation as R
import rosbag
from datetime import datetime
import csv

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

class FrankaROSInterface:
    def __init__(self,
                 pose_topic='/hybrid_joint_impedance_controller/desired_pose',
                 impedance_config_ns='/hybrid_joint_impedance_controller/dynamic_reconfigure_compliance_param_node',
                 ee_state_topic='/franka_state_controller/ee_pose',
                 joint_state_topic='/joint_states',
                 joint_state_desired_topic='/joint_states_desired',
                 franka_state_topic='/franka_state_controller/franka_states',
                 filtered_pose_topic='/hybrid_joint_impedance_controller/filtered_pose'):
        
        self.pose_pub = rospy.Publisher(pose_topic, PoseStamped, queue_size=1)
        self.dyn_client = DynClient(impedance_config_ns)

        # State variables - actual
        self.ee_pose = None
        self.ee_velocity = None  # Will be computed from pose history
        self.filtered_ee_pose = None
        self.joint_positions = None
        self.joint_velocities = None
        self.joint_torques = None
        
        # State variables - desired
        self.desired_ee_pose = None
        self.desired_ee_velocity = None  # From O_dP_EE_d in FrankaState
        self.desired_joint_positions = None
        self.desired_joint_velocities = None
        self.desired_joint_torques = None
        
        # For computing actual EE velocity from pose history
        self.prev_ee_pose = None
        self.prev_ee_timestamp = None
        
        # Subscribers
        rospy.Subscriber(ee_state_topic, Pose, self._ee_pose_callback)
        rospy.Subscriber(joint_state_topic, JointState, self._joint_state_callback)
        rospy.Subscriber(joint_state_desired_topic, JointState, self._joint_state_desired_callback)
        rospy.Subscriber(franka_state_topic, FrankaState, self._franka_state_callback)
        rospy.Subscriber(filtered_pose_topic, PoseStamped, self._filtered_ee_pose_callback)


    def _ee_pose_callback(self, msg):
        pos = msg.position
        quat = msg.orientation
        rotvec = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_rotvec()
        current_pose = np.concatenate([np.array([pos.x, pos.y, pos.z]), np.array([*rotvec])])
        
        # Compute actual EE velocity from pose differentiation (raw, unfiltered)
        current_time = rospy.Time.now()
        if self.prev_ee_pose is not None and self.prev_ee_timestamp is not None:
            dt = (current_time - self.prev_ee_timestamp).to_sec()
            if dt > 0:
                pose_diff = current_pose - self.prev_ee_pose
                self.ee_velocity = pose_diff / dt
        
        # Update pose and history
        self.ee_pose = current_pose
        self.prev_ee_pose = current_pose.copy()
        self.prev_ee_timestamp = current_time

    def _filtered_ee_pose_callback(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        rotvec = R.from_quat([q.x, q.y, q.z, q.w]).as_rotvec()
        self.filtered_ee_pose = np.array([p.x, p.y, p.z, *rotvec], dtype=np.float64)

    def _joint_state_callback(self, msg):
        """Actual joint states from /joint_states"""
        if len(msg.position) >= 7:
            self.joint_positions = np.array(msg.position[:7])
            self.joint_velocities = np.array(msg.velocity[:7]) if len(msg.velocity) >= 7 else np.zeros(7)
            self.joint_torques = np.array(msg.effort[:7]) if len(msg.effort) >= 7 else np.zeros(7)

    def _joint_state_desired_callback(self, msg):
        """Desired joint states from /joint_states_desired"""
        if len(msg.position) >= 7:
            self.desired_joint_positions = np.array(msg.position[:7])
            self.desired_joint_velocities = np.array(msg.velocity[:7]) if len(msg.velocity) >= 7 else np.zeros(7)
            self.desired_joint_torques = np.array(msg.effort[:7]) if len(msg.effort) >= 7 else np.zeros(7)

    def _franka_state_callback(self, msg):
        """Extract additional data from comprehensive Franka state"""
        # Extract desired EE velocity from O_dP_EE_d (most accurate source)
        if len(msg.O_dP_EE_d) >= 6:
            self.desired_ee_velocity = np.array(msg.O_dP_EE_d[:6])

    # Getter methods
    def get_ee_pose(self):
        while self.ee_pose is None and not rospy.is_shutdown():
            rospy.sleep(0.01)
        return self.ee_pose.copy() if self.ee_pose is not None else np.zeros(6)
    
    def get_filtered_ee_pose(self):
        return self.filtered_ee_pose.copy() if self.filtered_ee_pose is not None else np.zeros(6)

    def get_ee_velocity(self):
        """Alias for get_actual_ee_velocity for backward compatibility"""
        return self.get_actual_ee_velocity()

    def get_joint_positions(self):
        while self.joint_positions is None and not rospy.is_shutdown():
            rospy.sleep(0.01)
        return self.joint_positions.copy() if self.joint_positions is not None else np.zeros(7)
    
    def get_joint_velocities(self):
        return self.joint_velocities.copy() if self.joint_velocities is not None else np.zeros(7)
    
    def get_joint_torques(self):
        return self.joint_torques.copy() if self.joint_torques is not None else np.zeros(7)

    # Desired state getters
    def get_desired_ee_pose(self):
        return self.desired_ee_pose.copy() if self.desired_ee_pose is not None else np.zeros(6)
        
    def get_desired_ee_velocity(self):
        """Get desired EE velocity from FrankaState O_dP_EE_d"""
        return self.desired_ee_velocity.copy() if self.desired_ee_velocity is not None else np.zeros(6)

    def get_actual_ee_velocity(self):
        """Get actual EE velocity computed from pose differentiation"""
        return self.ee_velocity.copy() if self.ee_velocity is not None else np.zeros(6)
        
    def get_desired_joint_positions(self):
        return self.desired_joint_positions.copy() if self.desired_joint_positions is not None else np.zeros(7)
        
    def get_desired_joint_velocities(self):
        return self.desired_joint_velocities.copy() if self.desired_joint_velocities is not None else np.zeros(7)
        
    def get_desired_joint_torques(self):
        return self.desired_joint_torques.copy() if self.desired_joint_torques is not None else np.zeros(7)


    def update_desired_ee_pose(self, pose: np.ndarray):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'panda_link0'
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[:3]
        quat = R.from_rotvec(pose[3:]).as_quat()
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quat
        self.pose_pub.publish(msg)
        
        # Store desired pose for logging
        self.desired_ee_pose = pose.copy()

    def set_desired_ee_velocity(self, velocity: np.ndarray):
        self.desired_ee_velocity = velocity.copy()

    def update_stiffness_gains(self, Kx: np.ndarray):
        config = {
            'translational_stiffness_x': Kx[0],
            'translational_stiffness_y': Kx[1],
            'translational_stiffness_z': Kx[2],
        }
        self.dyn_client.update_configuration(config)
    
    def terminate_policy(self):
        rospy.loginfo("[FrankaROSInterface] Terminating policy...")

class FrankaDataLogger:
    def __init__(self, output_dir=None, episode_id=None, use_rosbag=True, use_csv=True):
        """
        Initialize the Franka data logger
        
        Args:
            output_dir: Directory to save logs (default: ./franka_logs)
            episode_id: Episode identifier (default: timestamp)
            use_rosbag: Whether to log to rosbag format
            use_csv: Whether to log to CSV format
        """
        self.output_dir = pathlib.Path(output_dir) if output_dir else pathlib.Path("./franka_logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_id = episode_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.use_rosbag = use_rosbag
        self.use_csv = use_csv
        
        # Initialize rosbag
        if self.use_rosbag:
            bag_path = self.output_dir / f"franka_data_{self.episode_id}.bag"
            self.bag = rosbag.Bag(str(bag_path), 'w')
            rospy.loginfo(f"Logging to rosbag: {bag_path}")
        
        # Initialize CSV
        if self.use_csv:
            csv_path = self.output_dir / f"franka_data_{self.episode_id}.csv"
            self.csv_file = open(csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            rospy.loginfo(f"Logging to CSV: {csv_path}")
            
            # Write CSV header
            self._write_csv_header()
    
    def _write_csv_header(self):
        """Write CSV header with all data fields"""
        header = [
            'timestamp',
            # Desired EE pose (6D: x,y,z,rx,ry,rz)
            'des_ee_x', 'des_ee_y', 'des_ee_z', 'des_ee_rx', 'des_ee_ry', 'des_ee_rz',
            # Actual EE pose
            'act_ee_x', 'act_ee_y', 'act_ee_z', 'act_ee_rx', 'act_ee_ry', 'act_ee_rz',
            # Filtered EE pose
            'filt_ee_x','filt_ee_y','filt_ee_z','filt_ee_rx','filt_ee_ry','filt_ee_rz',
            # Desired EE velocity (6D: vx,vy,vz,wx,wy,wz)
            'des_ee_vx', 'des_ee_vy', 'des_ee_vz', 'des_ee_wx', 'des_ee_wy', 'des_ee_wz',
            # Actual EE velocity  
            'act_ee_vx', 'act_ee_vy', 'act_ee_vz', 'act_ee_wx', 'act_ee_wy', 'act_ee_wz',
            # Desired joint positions (7 joints)
            'des_q1', 'des_q2', 'des_q3', 'des_q4', 'des_q5', 'des_q6', 'des_q7',
            # Actual joint positions
            'act_q1', 'act_q2', 'act_q3', 'act_q4', 'act_q5', 'act_q6', 'act_q7',
            # Desired joint velocities
            'des_dq1', 'des_dq2', 'des_dq3', 'des_dq4', 'des_dq5', 'des_dq6', 'des_dq7',
            # Actual joint velocities
            'act_dq1', 'act_dq2', 'act_dq3', 'act_dq4', 'act_dq5', 'act_dq6', 'act_dq7',
            # Desired joint torques
            'des_tau1', 'des_tau2', 'des_tau3', 'des_tau4', 'des_tau5', 'des_tau6', 'des_tau7',
            # Actual joint torques
            'act_tau1', 'act_tau2', 'act_tau3', 'act_tau4', 'act_tau5', 'act_tau6', 'act_tau7'
        ]
        self.csv_writer.writerow(header)
    
    def log_robot_state(self, robot_interface):
        """
        Log current robot state from FrankaROSInterface
        
        Args:
            robot_interface: Instance of FrankaROSInterface
        """
        timestamp = rospy.Time.now()
        
        # Get all state data from robot interface
        desired_ee_pose = robot_interface.get_desired_ee_pose()
        actual_ee_pose = robot_interface.get_ee_pose()

        filtered_ee_pose = robot_interface.get_filtered_ee_pose()

        desired_ee_vel = robot_interface.get_desired_ee_velocity()
        actual_ee_vel = robot_interface.get_actual_ee_velocity()
        
        desired_joint_pos = robot_interface.get_desired_joint_positions()
        actual_joint_pos = robot_interface.get_joint_positions()
        
        desired_joint_vel = robot_interface.get_desired_joint_velocities()
        actual_joint_vel = robot_interface.get_joint_velocities()
        
        desired_joint_torque = robot_interface.get_desired_joint_torques()
        actual_joint_torque = robot_interface.get_joint_torques()
        
        # Log to rosbag
        if self.use_rosbag:
            self._log_to_rosbag(timestamp, 
                               desired_ee_pose, actual_ee_pose, filtered_ee_pose,
                               desired_ee_vel, actual_ee_vel,
                               desired_joint_pos, actual_joint_pos, 
                               desired_joint_vel, actual_joint_vel,
                               desired_joint_torque, actual_joint_torque)
        
        # Log to CSV
        if self.use_csv:
            self._log_to_csv(timestamp.to_sec(),
                            desired_ee_pose, actual_ee_pose, filtered_ee_pose,
                            desired_ee_vel, actual_ee_vel,
                            desired_joint_pos, actual_joint_pos,
                            desired_joint_vel, actual_joint_vel,
                            desired_joint_torque, actual_joint_torque)
    
    def _log_to_csv(self, timestamp, des_ee_pose, act_ee_pose, filt_ee_pose, des_ee_vel, act_ee_vel,
                    des_joint_pos, act_joint_pos, des_joint_vel, act_joint_vel,
                    des_joint_torque, act_joint_torque):
        """Log data to CSV file"""
        row = ([timestamp] + 
               list(des_ee_pose) + list(act_ee_pose) + list(filt_ee_pose) +
               list(des_ee_vel) + list(act_ee_vel) +
               list(des_joint_pos) + list(act_joint_pos) +
               list(des_joint_vel) + list(act_joint_vel) +
               list(des_joint_torque) + list(act_joint_torque))
        
        self.csv_writer.writerow(row)
        self.csv_file.flush()  # Ensure data is written immediately
    
    def _log_to_rosbag(self, timestamp, des_ee_pose, act_ee_pose, filt_ee_pose, des_ee_vel, act_ee_vel,
                      des_joint_pos, act_joint_pos, des_joint_vel, act_joint_vel,
                      des_joint_torque, act_joint_torque):
        """Log data to rosbag"""
        
        # Log desired EE pose
        des_pose_msg = self._create_pose_msg(timestamp, des_ee_pose)
        self.bag.write('/franka_logger/desired_ee_pose', des_pose_msg, timestamp)
        
        # Log actual EE pose  
        act_pose_msg = self._create_pose_msg(timestamp, act_ee_pose)
        self.bag.write('/franka_logger/actual_ee_pose', act_pose_msg, timestamp)

        # Log filtered EE pose
        filt_pose_msg = self._create_pose_msg(timestamp, filt_ee_pose)
        self.bag.write('/franka_logger/filtered_ee_pose', filt_pose_msg, timestamp)

        # Log EE velocities
        des_vel_msg = self._create_twist_msg(des_ee_vel)
        self.bag.write('/franka_logger/desired_ee_velocity', des_vel_msg, timestamp)
        
        act_vel_msg = self._create_twist_msg(act_ee_vel)
        self.bag.write('/franka_logger/actual_ee_velocity', act_vel_msg, timestamp)
        
        # Log joint states
        des_joint_msg = self._create_joint_state_msg(timestamp, des_joint_pos, des_joint_vel, des_joint_torque)
        self.bag.write('/franka_logger/desired_joint_states', des_joint_msg, timestamp)
        
        act_joint_msg = self._create_joint_state_msg(timestamp, act_joint_pos, act_joint_vel, act_joint_torque)
        self.bag.write('/franka_logger/actual_joint_states', act_joint_msg, timestamp)
    
    def _create_pose_msg(self, timestamp, pose):
        """Create PoseStamped message from pose array"""
        msg = PoseStamped()
        msg.header.stamp = timestamp
        msg.header.frame_id = 'panda_link0'
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[:3]
        
        # Convert rotation vector to quaternion
        if np.linalg.norm(pose[3:]) > 0:
            quat = R.from_rotvec(pose[3:]).as_quat()
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quat
        else:
            # Identity quaternion
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = 0, 0, 0, 1
        
        return msg
    
    def _create_twist_msg(self, velocity):
        """Create Twist message from velocity array"""
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = velocity[:3]
        msg.angular.x, msg.angular.y, msg.angular.z = velocity[3:]
        return msg
    
    def _create_joint_state_msg(self, timestamp, positions, velocities, torques):
        """Create JointState message"""
        msg = JointState()
        msg.header.stamp = timestamp
        msg.name = [f'panda_joint{i+1}' for i in range(7)]
        msg.position = positions.tolist()
        msg.velocity = velocities.tolist()
        msg.effort = torques.tolist()
        return msg
    
    def close(self):
        """Close all log files"""
        if self.use_rosbag and hasattr(self, 'bag'):
            self.bag.close()
            rospy.loginfo("Rosbag closed")
        
        if self.use_csv and hasattr(self, 'csv_file'):
            self.csv_file.close()
            rospy.loginfo("CSV file closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class FrankaVariableImpedanceController(mp.Process):
    def __init__(self,
        shm_manager: SharedMemoryManager,
        # robot_interface=None,
        frequency=1000,
        launch_timeout=3,
        joints_init=None,
        joints_init_duration=None,
        soft_real_time=False,
        verbose=False,
        get_max_k=None,
        receive_latency=0.0,
        output_dir=None,
        episode_id=None,
        max_pos_speed=0.25,
        max_rot_speed=0.6
        ):

        super().__init__(name="FrankaVariableImpedanceController")
        # self.daemon = True
        # self.robot_interface = robot_interface
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_duration = joints_init_duration
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose
        self.output_dir = pathlib.Path(output_dir) if output_dir is not None else None
        self.episode_id = episode_id
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            # 'target_stiffness': np.zeros((3,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }

        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )


        receive_keys = [
            ('ActualTCPPose', 'get_ee_pose')
            # ('ActualQ', 'get_joint_positions'),
        ]
        
        example = dict()
        for key, func_name in receive_keys:
            if 'joint' in func_name:
                example[key] = np.zeros(7)
            elif 'ee_pose' in func_name:
                example[key] = np.zeros(6)
        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FrankaVariableImpedanceController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        # print("[DEBUG] STOP called from:")
        traceback.print_stack()
        message = {'cmd': Command.STOP.value}
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def servoL(self, pose, duration=0.1):
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)
        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,    
            # 'target_stiffness': stiffness,
            'duration': duration
        }

        # print("[DEBUG] Sending servoL command:", message)
        self.input_queue.put(message)


    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            # 'target_stiffness': stiffness,
            'target_time': target_time
        }
        self.input_queue.put(message)


    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def run(self):

        rospy.init_node('franka_interpolation_controller')
        robot = FrankaROSInterface()

        logger = None

        try:
            # stiffness_update_rate = 10  # Hz
            # stiffness_update_dt = 1.0 / stiffness_update_rate
            # last_stiffness_update_time = time.monotonic()
            # target_stiffness = None

            # Initialize logger
            if self.output_dir:
                logger = FrankaDataLogger(
                    output_dir=self.output_dir, 
                    episode_id=self.episode_id,
                    use_rosbag=False, 
                    use_csv=True
                )

            dt = 1. / self.frequency
            curr_pose = robot.get_ee_pose()
            print("[Robot] Got EE pose:", curr_pose)
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            t_start = time.monotonic()
            iter_idx = 0

            # rate = rospy.Rate(self.frequency)
            keep_running = True

            while not rospy.is_shutdown() and keep_running:
                t_now = time.monotonic()
                
                # Compute interpolated pose and publish
                ee_pose = pose_interp(t_now)
                # print("[Robot] Interpolated EE pose:", ee_pose)
                robot.update_desired_ee_pose(ee_pose)

                # print("Updated the following pose", ee_pose)

                # Log data with error handling
                if logger:
                    try:
                        logger.log_robot_state(robot)
                    except Exception as e:
                        rospy.logwarn_throttle(1.0, f"Logging error: {e}")

                # # Lower frequency stiffness update
                # if target_stiffness is not None and t_now - last_stiffness_update_time >= stiffness_update_dt:
                #     robot.update_stiffness_gains(target_stiffness)
                #     last_stiffness_update_time = t_now

                # Read robot state and push to buffer
                state = dict()
                for key, func_name in self.receive_keys:
                    state[key] = getattr(robot, func_name)()

                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # Process commands
                try:
                    commands = self.input_queue.get_k(1)
                    # print("[Robot] Received commands:", commands)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    # print("[Robot] No command received.")
                    n_cmd = 0

                # print(f"Number of commands received: {n_cmd}")
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        # if i == 0 and iter_idx < 10:
                        #     print("[WARN] Early STOP detected, possibly uninitialized. Skipping.")
                        #     continue
                        print("[Robot] Stopped")
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + (1. / self.frequency)
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time
                        )
                        last_waypoint_time = t_insert
                        # print("Interpolated pose:", pose_interp)

                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        # target_stiffness = command['target_stiffness']
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + (1. / self.frequency)
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                        # print("Interpolated pose:", pose_interp)

                    else:
                        keep_running = False
                        break

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

        finally:
            # mandatory cleanup
            # terminate
            if logger:
                logger.close()
            robot.terminate_policy()
            del robot
            self.ready_event.set()


# class FrankaROSInterface:
#     def __init__(self,
#                  pose_topic='/hybrid_joint_impedance_controller/desired_pose',
#                  impedance_config_ns='/hybrid_joint_impedance_controller/dynamic_reconfigure_compliance_param_node',
#                  ee_state_topic='/franka_state_controller/ee_pose',
#                  joint_state_topic='/joint_states'):
        
#         self.pose_pub = rospy.Publisher(pose_topic, PoseStamped, queue_size=1)
#         self.dyn_client = DynClient(impedance_config_ns)

#         self.ee_pose = None
#         self.joint_positions = None
#         rospy.Subscriber(ee_state_topic, Pose, self._ee_pose_callback)
#         rospy.Subscriber(joint_state_topic, JointState, self._joint_state_callback)

#     def _ee_pose_callback(self, msg):
#         pos = msg.position
#         quat = msg.orientation
#         rotvec = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_rotvec()
#         self.ee_pose = np.concatenate([np.array([pos.x, pos.y, pos.z]), np.array([*rotvec])])

#     def _joint_state_callback(self, msg):
#         self.joint_positions = np.array(msg.position)


#     def get_ee_pose(self):
#         while self.ee_pose is None and not rospy.is_shutdown():
#             rospy.sleep(0.01)
#         return self.ee_pose

#     def get_joint_positions(self):
#         while self.joint_positions is None and not rospy.is_shutdown():
#             rospy.sleep(0.01)
#         return self.joint_positions

#     def update_desired_ee_pose(self, pose: np.ndarray):
#         msg = PoseStamped()
#         msg.header.stamp = rospy.Time.now()
#         msg.header.frame_id = 'panda_link0'
#         msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[:3]
#         # print("Desired pose:", pose)
#         quat = R.from_rotvec(pose[3:]).as_quat()
#         msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quat
#         # print("Publishing desired pose:", msg)
#         self.pose_pub.publish(msg)

#     def update_stiffness_gains(self, Kx: np.ndarray):
#         config = {
#             'translational_stiffness_x': Kx[0],
#             'translational_stiffness_y': Kx[1],
#             'translational_stiffness_z': Kx[2],
#         }
#         self.dyn_client.update_configuration(config)
    
#     def terminate_policy(self):
#         rospy.loginfo("[FrankaROSInterface] Terminating policy...")