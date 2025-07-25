import rospy
import numpy as np
import enum
import time
import os
from policy_utils.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from policy_utils.shared_memory_ring_buffer import SharedMemoryRingBuffer
from policy_utils.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from policy_utils.precise_sleep import precise_wait
from multiprocessing.managers import SharedMemoryManager
import pathlib
import multiprocessing as mp
import traceback
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
from dynamic_reconfigure.client import Client as DynClient
from scipy.spatial.transform import Rotation as R

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

class FrankaROSInterface:
    def __init__(self,
                 pose_topic='/cartesian_pose_impedance_controller/desired_pose',
                 impedance_config_ns='/cartesian_pose_impedance_controller/dynamic_reconfigure_compliance_param_node',
                 ee_state_topic='/franka_state_controller/ee_pose',
                 joint_state_topic='/joint_states'):
        
        self.pose_pub = rospy.Publisher(pose_topic, PoseStamped, queue_size=1)
        self.dyn_client = DynClient(impedance_config_ns)

        self.ee_pose = None
        self.joint_positions = None
        rospy.Subscriber(ee_state_topic, Pose, self._ee_pose_callback)
        rospy.Subscriber(joint_state_topic, JointState, self._joint_state_callback)

    def _ee_pose_callback(self, msg):
        pos = msg.position
        quat = msg.orientation
        rotvec = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_rotvec()
        self.ee_pose = np.concatenate([np.array([pos.x, pos.y, pos.z]), np.array([*rotvec])])

    def _joint_state_callback(self, msg):
        self.joint_positions = np.array(msg.position)


    def get_ee_pose(self):
        while self.ee_pose is None and not rospy.is_shutdown():
            rospy.sleep(0.01)
        return self.ee_pose

    def get_joint_positions(self):
        while self.joint_positions is None and not rospy.is_shutdown():
            rospy.sleep(0.01)
        return self.joint_positions

    def update_desired_ee_pose(self, pose: np.ndarray):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'panda_link0'
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[:3]
        # print("Desired pose:", pose)
        quat = R.from_rotvec(pose[3:]).as_quat()
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quat
        # print("Publishing desired pose:", msg)
        self.pose_pub.publish(msg)

    def update_stiffness_gains(self, Kx: np.ndarray):
        config = {
            'translational_stiffness_x': Kx[0],
            'translational_stiffness_y': Kx[1],
            'translational_stiffness_z': Kx[2],
        }
        self.dyn_client.update_configuration(config)
    
    def terminate_policy(self):
        rospy.loginfo("[FrankaROSInterface] Terminating policy...")

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

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'target_stiffness': np.zeros((3,), dtype=np.float64),
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
        print("[DEBUG] STOP called from:")
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

    def servoL(self, pose, stiffness, duration=0.1):
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)
        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,    
            'target_stiffness': stiffness,
            'duration': duration
        }

        print("[DEBUG] Sending servoL command:", message)
        self.input_queue.put(message)


    def schedule_waypoint(self, pose, stiffness, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_stiffness': stiffness,
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

        try:
            stiffness_update_rate = 10  # Hz
            stiffness_update_dt = 1.0 / stiffness_update_rate
            last_stiffness_update_time = time.monotonic()
            target_stiffness = None

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

                # Lower frequency stiffness update
                if target_stiffness is not None and t_now - last_stiffness_update_time >= stiffness_update_dt:
                    robot.update_stiffness_gains(target_stiffness)
                    last_stiffness_update_time = t_now

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
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_stiffness = command['target_stiffness']
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + (1. / self.frequency)
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            # max_pos_speed=3.5,
                            # max_rot_speed=3.5,                            
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time

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
            robot.terminate_policy()
            del robot
            self.ready_event.set()