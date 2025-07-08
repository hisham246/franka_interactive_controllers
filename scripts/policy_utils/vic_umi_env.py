import pathlib
import numpy as np
import time
import shutil
import math
import cv2
from multiprocessing.managers import SharedMemoryManager
from policy_utils.franka_interpolation_controller import FrankaVariableImpedanceController
from policy_utils.franka_hand_controller import FrankaHandController
from policy_utils.multi_uvc_camera import MultiUvcCamera
from policy_utils.video_recorder import VideoRecorder
from policy_utils.timestamp_accumulator import (
    TimestampActionAccumulator,
    ObsAccumulator
)
from policy_utils.cv_util import (
    draw_predefined_mask,
    get_mirror_crop_slices
)
from policy_utils.replay_buffer import ReplayBuffer
from policy_utils.cv2_util import (
    get_image_transform, optimal_row_cols)
from policy_utils.usb_util import reset_all_avermedia_devices, get_sorted_v4l_paths
from policy_utils.interpolation_util import get_interp1d, PoseInterpolator

class VicUmiEnv:
    def __init__(self, 
            # required params
            output_dir,
            gripper_ip,
            gripper_port=4242,
            # env params
            frequency=10,
            # obs
            obs_image_resolution=(224,224),
            max_obs_buffer_size=60,
            obs_float32=False,
            camera_reorder=None,
            no_mirror=False,
            fisheye_converter=None,
            mirror_crop=False,
            mirror_swap=False,
            # timing
            align_camera_idx=0,
            # this latency compensates receive_timestamp
            # all in seconds
            camera_obs_latency=0.125,
            robot_obs_latency=0.0001,
            gripper_obs_latency=0.01,
            robot_action_latency=0.1,
            gripper_action_latency=0.1,
            # all in steps (relative to frequency)
            camera_down_sample_steps=1,
            robot_down_sample_steps=1,
            gripper_down_sample_steps=1,
            # all in steps (relative to frequency)
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            # action
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            multi_cam_vis_resolution=(960, 960),
            # shared memory
            shm_manager=None
            ):
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        # Find and reset all Elgato capture cards.
        # Required to workaround a firmware bug.
        reset_all_avermedia_devices()

        # Wait for all v4l cameras to be back online
        time.sleep(0.1)
        v4l_paths = get_sorted_v4l_paths()
        if camera_reorder is not None:
            paths = [v4l_paths[i] for i in camera_reorder]
            v4l_paths = paths
        
        # compute resolution for vis
        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(v4l_paths),
            in_wh_ratio=4/3,
            max_resolution=multi_cam_vis_resolution
        )

        resolution = list()
        capture_fps = list()
        cap_buffer_size = list()
        video_recorder = list()
        transform = list()
        vis_transform = list()
        for idx, path in enumerate(v4l_paths):
            if 'Cam_Link_4K' in path:
                res = (3840, 2160)
                fps = 30
                buf = 3
                bit_rate = 6000*1000
                def tf4k(data, input_res=res):
                    img = data['color']
                    f = get_image_transform(
                        input_res=input_res,
                        output_res=obs_image_resolution, 
                        # obs output rgb
                        bgr_to_rgb=True)
                    img = f(img)
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data['color'] = img
                    return data
                transform.append(tf4k)
            else:
                res = (1920, 1080)
                fps = 60
                buf = 1
                bit_rate = 3000*1000
                stack_crop = (idx==0) and mirror_crop
                is_mirror = None
                if mirror_swap:
                    mirror_mask = np.ones((224,224,3),dtype=np.uint8)
                    mirror_mask = draw_predefined_mask(
                        mirror_mask, color=(0,0,0), mirror=True, gripper=False, finger=False)
                    is_mirror = (mirror_mask[...,0] == 0)

                def tf(data, input_res=res, stack_crop=stack_crop, is_mirror=is_mirror):
                    img = data['color']
                    if fisheye_converter is None:
                        crop_img = None
                        if stack_crop:
                            slices = get_mirror_crop_slices(img.shape[:2], left=False)
                            crop = img[slices]
                            crop_img = cv2.resize(crop, obs_image_resolution)
                            crop_img = crop_img[:,::-1,::-1] # bgr to rgb
                        f = get_image_transform(
                            input_res=input_res,
                            output_res=obs_image_resolution, 
                            # obs output rgb
                            bgr_to_rgb=True)
                        img = np.ascontiguousarray(f(img))
                        if is_mirror is not None:
                            img[is_mirror] = img[:,::-1,:][is_mirror]
                        img = draw_predefined_mask(img, color=(0,0,0), 
                            mirror=no_mirror, gripper=True, finger=False, use_aa=True)
                        if crop_img is not None:
                            img = np.concatenate([img, crop_img], axis=-1)
                    else:
                        img = fisheye_converter.forward(img)
                        img = img[...,::-1]
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data['color'] = img
                    return data
                transform.append(tf)

            resolution.append(res)
            capture_fps.append(fps)
            cap_buffer_size.append(buf)
            video_recorder.append(VideoRecorder.create_hevc_nvenc(
                fps=fps,
                input_pix_fmt='bgr24',
                bit_rate=bit_rate
            ))

            def vis_tf(data, input_res=res):
                img = data['color']
                f = get_image_transform(
                    input_res=input_res,
                    output_res=(rw,rh),
                    bgr_to_rgb=False
                )
                img = f(img)
                data['color'] = img
                return data
            vis_transform.append(vis_tf)

        camera = MultiUvcCamera(
            dev_video_paths=v4l_paths,
            shm_manager=shm_manager,
            resolution=resolution,
            capture_fps=capture_fps,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            get_max_k=max_obs_buffer_size,
            receive_latency=camera_obs_latency,
            cap_buffer_size=cap_buffer_size,
            transform=transform,
            vis_transform=vis_transform,
            video_recorder=video_recorder,
            verbose=False
        )

        self.replay_buffer = replay_buffer
        self.episode_id_counter = self.replay_buffer.n_episodes
        
        robot = FrankaVariableImpedanceController(
            shm_manager=shm_manager,
            frequency=1000,
            verbose=False,
            receive_latency=robot_obs_latency,
            output_dir=output_dir,
            episode_id=self.episode_id_counter,
        )
        
        gripper = FrankaHandController(
            host=gripper_ip,
            port=gripper_port,
            speed=0.05,
            force=20.0,
            update_rate=frequency
        )

        self.camera = camera
        self.robot = robot
        self.gripper = gripper
        self.frequency = frequency
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.mirror_crop = mirror_crop
        # timing
        self.align_camera_idx = align_camera_idx
        self.camera_obs_latency = camera_obs_latency
        self.robot_obs_latency = robot_obs_latency
        self.gripper_obs_latency = gripper_obs_latency
        self.robot_action_latency = robot_action_latency
        self.gripper_action_latency = gripper_action_latency
        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.gripper_down_sample_steps = gripper_down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.gripper_obs_horizon = gripper_obs_horizon
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        # temp memory buffers
        self.last_camera_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None

        self.start_time = None
            
    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.camera.is_ready and self.robot.is_ready
    
    def start(self, wait=True):
        self.camera.start(wait=False)
        self.gripper.start(wait=False)
        self.robot.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.end_episode()
        self.robot.stop(wait=False)
        self.gripper.stop(wait=False)
        self.camera.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.camera.start_wait()
        self.gripper.start_wait()
        self.robot.start_wait()
    
    def stop_wait(self):
        self.robot.stop_wait()
        self.gripper.stop_wait()
        self.camera.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb): 
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        """
        Timestamp alignment policy
        'current' time is the last timestamp of align_camera_idx
        All other cameras, find corresponding frame with the nearest timestamp
        All low-dim observations, interpolate with respect to 'current' time
        """

        "observation dict"
        assert self.is_ready

        # get data
        # 60 Hz, camera_calibrated_timestamp
        k = math.ceil(
            self.camera_obs_horizon * self.camera_down_sample_steps \
            * (60 / self.frequency))
        
        self.last_camera_data = self.camera.get(
            k=k, 
            out=self.last_camera_data)

        last_robot_data = self.robot.get_all_state()

        print(last_robot_data)

        # 30 hz, gripper_receive_timestamp
        # last_gripper_data = self.gripper.get_all_state()
        last_gripper_data = 0.05
        last_timestamp = self.last_camera_data[0]['timestamp'][-1]
        dt = 1 / self.frequency

        # align camera obs timestamps
        camera_obs_timestamps = last_timestamp - (
            np.arange(self.camera_obs_horizon)[::-1] * self.camera_down_sample_steps * dt)

        this_timestamps = self.last_camera_data[0]['timestamp']
        this_idxs = [np.argmin(np.abs(this_timestamps - t)) for t in camera_obs_timestamps]

        camera_obs = dict()
        if self.mirror_crop:
            camera_obs['camera0_rgb'] = self.last_camera_data[0]['color'][...,:3][this_idxs]
            camera_obs['camera0_rgb_mirror_crop'] = self.last_camera_data[0]['color'][...,3:][this_idxs]
        else:
            camera_obs['camera0_rgb'] = self.last_camera_data[0]['color'][this_idxs]

        # align robot obs
        robot_obs_timestamps = last_timestamp - (
            np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt)
        robot_pose_interpolator = PoseInterpolator(
            t=last_robot_data['robot_timestamp'], 
            x=last_robot_data['ActualTCPPose'])
        robot_pose = robot_pose_interpolator(robot_obs_timestamps)
        robot_obs = {
            'robot0_eef_pos': robot_pose[...,:3],
            'robot0_eef_rot_axis_angle': robot_pose[...,3:]
        }

        # align gripper obs
        # gripper_obs_timestamps = last_timestamp - (
        #     np.arange(self.gripper_obs_horizon)[::-1] * self.gripper_down_sample_steps * dt)
        # gripper_interpolator = get_interp1d(
        #     t=last_gripper_data['gripper_timestamp'],
        #     x=last_gripper_data['gripper_position'][...,None]
        # )
        # gripper_interpolator = get_interp1d(
        #     t=np.array(last_gripper_data['gripper_timestamp']),
        #     x=np.array(last_gripper_data['gripper_position'])[..., None]
        # )

        x = np.array(last_gripper_data['gripper_position'])[-1]
        gripper_obs = {
            'robot0_gripper_width': np.repeat([[x]], self.gripper_obs_horizon, axis=0)
        }

        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                data={
                    'robot0_eef_pose': last_robot_data['ActualTCPPose'],
                    'robot0_joint_pos': last_robot_data['ActualQ'],
                    'robot0_joint_vel': last_robot_data['ActualQd'],
                },
                timestamps=last_robot_data['robot_timestamp']
            )
            self.obs_accumulator.put(
                data={
                    'robot0_gripper_width': np.array(last_gripper_data['gripper_position'])[..., None]
                },
                timestamps=last_gripper_data['gripper_timestamp']
            )

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data.update(gripper_obs)
        obs_data['timestamp'] = camera_obs_timestamps

        return obs_data
    
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray,
            compensate_latency=False):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        r_latency = self.robot_action_latency if compensate_latency else 0.0
        # g_latency = self.gripper_action_latency if compensate_latency else 0.0

        # Kx_trans = np.array([1000.0, 1000.0, 1000.0])
        # Kx_rot = np.array([30.0, 30.0, 30.0])

        # schedule waypoints
        for i in range(len(new_actions)):
            r_actions = new_actions[i,:6]
            # g_actions = new_actions[i, 9:]
            g_actions = new_actions[i, 6:]

            # Kx_trans = new_actions[i, 6:9]
            # Kx = np.concatenate([Kx_trans, Kx_rot])
            
            # Damping gains
            # Kxd = 2 * 0.707 * np.sqrt(Kx)

            # Update the impedance gains with Kx and Kxd
            # self.robot.set_impedance(Kx, Kxd)

            self.robot.schedule_waypoint(
                pose=r_actions,
                target_time=new_timestamps[i]-r_latency
            )
            self.gripper.schedule_waypoint(
                pos=g_actions)

        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
    
    def get_robot_state(self):
        return self.robot.get_state()

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        # episode_id = self.replay_buffer.n_episodes
        episode_id = self.episode_id_counter
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = 1
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
        
        # start recording on camera
        self.camera.restart_put(start_time=start_time)
        self.camera.start_recording(video_path=video_paths[0], start_time=start_time)

        # create accumulators
        self.obs_accumulator = ObsAccumulator()
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        print(f'Episode {episode_id} started!')
    
    def end_episode(self):
        "Stop recording"
        assert self.is_ready
        
        # stop video recorder
        self.camera.stop_recording()

        # TODO
        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            end_time = float('inf')
            for key, value in self.obs_accumulator.timestamps.items():
                end_time = min(end_time, value[-1])
            end_time = min(end_time, self.action_accumulator.timestamps[-1])

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            n_steps = 0
            if np.sum(self.action_accumulator.timestamps <= end_time) > 0:
                n_steps = np.nonzero(self.action_accumulator.timestamps <= end_time)[0][-1]+1

            if n_steps > 0:
                timestamps = action_timestamps[:n_steps]
                episode = {
                    'timestamp': timestamps,
                    'action': actions[:n_steps],
                }
                robot_pose_interpolator = PoseInterpolator(
                    t=np.array(self.obs_accumulator.timestamps['robot0_eef_pose']),
                    x=np.array(self.obs_accumulator.data['robot0_eef_pose'])
                )
                robot_pose = robot_pose_interpolator(timestamps)
                episode['robot0_eef_pos'] = robot_pose[:,:3]
                episode['robot0_eef_rot_axis_angle'] = robot_pose[:,3:]
                joint_pos_interpolator = get_interp1d(
                    np.array(self.obs_accumulator.timestamps['robot0_joint_pos']),
                    np.array(self.obs_accumulator.data['robot0_joint_pos'])
                )
                joint_vel_interpolator = get_interp1d(
                    np.array(self.obs_accumulator.timestamps['robot0_joint_vel']),
                    np.array(self.obs_accumulator.data['robot0_joint_vel'])
                )
                episode['robot0_joint_pos'] = joint_pos_interpolator(timestamps)
                episode['robot0_joint_vel'] = joint_vel_interpolator(timestamps)

                gripper_ts = np.array(self.obs_accumulator.timestamps['robot0_gripper_width'])
                gripper_data = np.array(self.obs_accumulator.data['robot0_gripper_width'])

                if len(gripper_data) == 0 or len(gripper_ts) == 0:
                    print("[Warning] No gripper data collected.")
                    episode['robot0_gripper_width'] = np.zeros((len(timestamps), 1))
                elif len(np.unique(gripper_ts)) < 2:
                    print("[Warning] Not enough unique gripper timestamps to interpolate. Repeating last value.")
                    episode['robot0_gripper_width'] = np.repeat(gripper_data[-1:], len(timestamps), axis=0)
                else:
                    gripper_interpolator = get_interp1d(t=gripper_ts, x=gripper_data)
                    episode['robot0_gripper_width'] = gripper_interpolator(timestamps)

                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')
            
            self.obs_accumulator = None
            self.action_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')
