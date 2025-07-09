import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose
from std_srvs.srv import Empty
from dynamic_reconfigure.client import Client as DynClient
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R
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


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    SET_IMPEDANCE = 3

class FrankaROSInterface:
    def __init__(self,
                 pose_topic='/cartesian_impedance_controller/desired_pose',
                 impedance_config_ns='/cartesian_pose_impedance_controller/dynamic_reconfigure_compliance_param_node',
                 ee_state_topic='/franka_state_controller/ee_pose',
                 joint_state_topic='/joint_states'):
        
        # Publishers
        self.pose_pub = rospy.Publisher(pose_topic, PoseStamped, queue_size=1)

        # Dynamic reconfigure client for impedance
        self.dyn_client = DynClient(impedance_config_ns)

        # Subscribers
        self.ee_pose = None
        self.joint_positions = None
        rospy.Subscriber(ee_state_topic, Pose, self._ee_pose_callback)
        rospy.Subscriber(joint_state_topic, JointState, self._joint_state_callback)

    def _ee_pose_callback(self, msg):
        pos = msg.position
        quat = msg.orientation
        rotvec = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_rotvec()
        self.ee_pose = np.array([pos.x, pos.y, pos.z, *rotvec])

    def _joint_state_callback(self, msg):
        self.joint_positions = np.array(msg.position)
        self.joint_velocities = np.array(msg.velocity)

    def get_ee_pose(self):
        while self.ee_pose is None and not rospy.is_shutdown():
            rospy.sleep(0.01)
        return self.ee_pose

    def get_joint_positions(self):
        while self.joint_positions is None and not rospy.is_shutdown():
            rospy.sleep(0.01)
        return self.joint_positions

    def get_robot_state(self):
        return {
            'ee_pose': self.get_ee_pose(),
            'joint_positions': self.get_joint_positions(),
        }

    def update_desired_ee_pose(self, pose: np.ndarray):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'panda_link0'
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[:3]
        quat = R.from_rotvec(pose[3:]).as_quat()
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quat
        self.pose_pub.publish(msg)

    def update_impedance_gains(self, Kx: np.ndarray, Kxd: np.ndarray):
        config = {
            'translational_stiffness_x': Kx[0],
            'translational_stiffness_y': Kx[1],
            'translational_stiffness_z': Kx[2],
            'rotational_stiffness_x': Kx[3],
            'rotational_stiffness_y': Kx[4],
            'rotational_stiffness_z': Kx[5],
            'translational_damping_x': Kxd[0],
            'translational_damping_y': Kxd[1],
            'translational_damping_z': Kxd[2],
            'rotational_damping_x': Kxd[3],
            'rotational_damping_y': Kxd[4],
            'rotational_damping_z': Kxd[5],
        }
        self.dyn_client.update_configuration(config)

    def terminate_policy(self):
        rospy.loginfo("[FrankaROSInterface] Terminating policy...")


class FrankaVariableImpedanceController(mp.Process):
    def __init__(self,
        shm_manager: SharedMemoryManager,
        frequency=1000,
        launch_timeout=3,
        joints_init=None,
        joints_init_duration=None,
        soft_real_time=False,
        verbose=False,
        get_max_k=None,
        receive_latency=0.0,
        output_dir=None,
        episode_id=None
        ):

        super().__init__(name="FrankaVariableImpedanceController")
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
            'Kx': np.zeros((6,), dtype=np.float64),
            'Kxd': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        receive_keys = [
            ('ActualTCPPose', 'get_ee_pose'),
            ('ActualQ', 'get_joint_positions'),
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
        self.input_queue.put({
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        })

    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)
        self.input_queue.put({
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        })

    def set_impedance(self, Kx, Kxd):
        assert self.is_alive()
        assert Kx.shape == (6,) and Kxd.shape == (6,)
        self.input_queue.put({
            'cmd': Command.SET_IMPEDANCE.value,
            'Kx': Kx,
            'Kxd': Kxd
        })

    def get_state(self, k=None, out=None):
        return self.ring_buffer.get(out=out) if k is None else self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def run(self):
        rospy.init_node("franka_variable_impedance_controller", anonymous=True)
        # if self.soft_real_time:
        #     os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))

        robot = FrankaROSInterface()

        while not rospy.is_shutdown() and (robot.ee_pose is None or robot.joint_positions is None):
            print("[Robot] Waiting for messages...")
            rospy.sleep(0.1)

        try:
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

            rate = rospy.Rate(self.frequency)
            keep_running = True

            while not rospy.is_shutdown() and keep_running:
                t_now = time.monotonic()
                
                # Compute interpolated pose and publish
                ee_pose = pose_interp(t_now)
                robot.update_desired_ee_pose(ee_pose)
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
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = {key: val[i] for key, val in commands.items()}
                    cmd = command['cmd']
                    if cmd == Command.STOP.value:
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
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + (1. / self.frequency)
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.SET_IMPEDANCE.value:
                        self.curr_Kx = command['Kx']
                        self.curr_Kxd = command['Kxd']
                        robot.update_impedance_gains(self.curr_Kx, self.curr_Kxd)
                    else:
                        keep_running = False
                        break

                t_wait_until = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_until, time_func=time.monotonic)
                iter_idx += 1


        finally:
            # mandatory cleanup
            # terminate
            robot.terminate_policy()
            del robot
            self.ready_event.set()