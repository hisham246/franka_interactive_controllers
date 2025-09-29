"""
Run a policy on the real robot.
"""
import sys
import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager
from scipy.spatial.transform import Rotation as R
import av
import cv2
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
import json
import pandas as pd
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from policy_utils.replay_buffer import ReplayBuffer
from policy_utils.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from policy_utils.pytorch_util import dict_apply
from policy_utils.base_workspace import BaseWorkspace
from policy_utils.precise_sleep import precise_wait
from policy_utils.vic_umi_env import VicUmiEnv
from policy_utils.keystroke_counter import (
    KeystrokeCounter, KeyCode
)
from policy_utils.real_inference_util import (get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)

OmegaConf.register_new_resolver("eval", eval, replace=True)


# quat utilities
def _q_norm(q): return q / (np.linalg.norm(q) + 1e-12)
def _q_conj(q): return np.array([-q[0], -q[1], -q[2], q[3]])
def _q_mul(a,b):
    x1,y1,z1,w1 = a; x2,y2,z2,w2 = b
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])
def _slerp(q0,q1,t):
    q0=_q_norm(q0); q1=_q_norm(q1)
    if np.dot(q0,q1) < 0: q1 = -q1
    d = np.clip(np.dot(q0,q1), -1.0, 1.0)
    if d > 0.9995:
        out = _q_norm(q0 + t*(q1-q0))
    else:
        th = np.arccos(d)
        out = (np.sin((1-t)*th)*q0 + np.sin(t*th)*q1)/np.sin(th)
    return _q_norm(out)
def _geo_angle(q0,q1):
    d = np.clip(abs(np.dot(_q_norm(q0), _q_norm(q1))), -1.0, 1.0)
    return 2.0*np.arccos(d)  # [0, pi]

# geodesic mean around a reference quaternion
def _weighted_mean_quats_around(q_ref, quats, weights):
    # map each quat to tangent at q_ref, average, exp back
    vecs = []
    for q in quats:
        if np.dot(q, q_ref) < 0: q = -q
        q_err = _q_mul(_q_conj(q_ref), q)
        rotvec = R.from_quat(q_err).as_rotvec()
        vecs.append(rotvec)
    vbar = np.average(np.stack(vecs, 0), axis=0, weights=weights)
    q_delta = R.from_rotvec(vbar).as_quat()
    return _q_mul(q_ref, q_delta)

# SE(3) step limiter
def _limit_se3_step(p_prev, q_prev, p_cmd, q_cmd, v_max, w_max, dt):
    # translation
    dp = p_cmd - p_prev
    n = np.linalg.norm(dp)
    max_dp = v_max * dt
    if n > max_dp:
        dp *= (max_dp / (n + 1e-12))
    p_new = p_prev + dp
    # rotation
    ang = _geo_angle(q_prev, q_cmd)
    max_dang = w_max * dt
    if ang > max_dang + 1e-9:
        t = max_dang / ang
        q_new = _slerp(q_prev, q_cmd, t)
    else:
        # keep hemisphere continuity
        q_new = q_cmd if np.dot(q_prev, q_cmd) >= 0 else -q_cmd
    return p_new, _q_norm(q_new)

def _Rz(theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]], dtype=float)

# robot -> policy:  x_p = -y_r,  y_p = x_r,  z same
_M    = _Rz(np.deg2rad(-90.0))
# policy -> robot (inverse)
_Minv = _Rz(np.deg2rad(90.0))

def _map_pose_array(arr, M):
    """
    Apply 3x3 map M to position and axis-angle parts of pose arrays.
    Accepts shapes (...,6) or (...,7): [x,y,z, rx,ry,rz, (gripper?)]
    Returns a copy with same shape.
    """
    a = np.asarray(arr).copy()
    if a.ndim == 1:
        a = a[None, ...]
        squeeze = True
    else:
        squeeze = False
    a[..., 0:3] = a[..., 0:3] @ M
    a[..., 3:6] = a[..., 3:6] @ M
    if squeeze:
        a = a[0]
    return a

def main():
    output = '/home/hisham246/uwaterloo/reaching_ball_multimodal_3'
    gripper_ip = '129.97.71.27'
    gripper_port = 4242
    match_dataset = None
    match_camera = 0
    steps_per_inference = 8
    vis_camera_idx = 0
    max_duration = 120
    frequency = 10
    no_mirror = False
    sim_fov = None
    camera_intrinsics = None
    mirror_crop = False
    mirror_swap = False
            
    # Diffusion Transformer
    # ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/surface_wiping_transformer_position_control.ckpt'

    # Diffusion UNet
    ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/reaching_ball_multimodal.ckpt'
    # ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/diffusion_unet_pickplace_2.ckpt'
    # ckpt_path = '/home/hisham246/uwaterloo/cup_in_the_wild/cup_wild_vit_l_1img.ckpt'

    # Compliance policy unet
    # ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/diffusion_unet_compliance_trial_2.ckpt'

    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']

    # print("Config:", cfg)

    cfg._target_ = "diffusion_policy.train_diffusion_unet_image_workspace.TrainDiffusionUnetImageWorkspace"
    cfg.policy._target_ = "diffusion_policy.diffusion_unet_timm_policy.DiffusionUnetTimmPolicy"
    cfg.policy.obs_encoder._target_ = "policy_utils.timm_obs_encoder.TimmObsEncoder"
    cfg.ema._target_ = "policy_utils.ema_model.EMAModel"

    # cfg._target_ = "diffusion_policy.train_diffusion_transformer_timm_workspace.TrainDiffusionTransformerTimmWorkspace"
    # cfg.policy._target_ = "diffusion_policy.diffusion_transformer_timm_policy.DiffusionTransformerTimmPolicy"
    # cfg.policy.obs_encoder._target_ = "policy_utils.transformer_obs_encoder.TransformerObsEncoder"
    # cfg.ema._target_ = "policy_utils.ema_model.EMAModel"

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    # load fisheye converter
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print("steps_per_inference:", steps_per_inference)
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            VicUmiEnv(
                output_dir=output,
                # robot_interface=robot_interface,
                gripper_ip=gripper_ip,
                gripper_port=gripper_port, 
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=None,
                # init_joints=init_joints,
                # enable_multi_cam_vis=True,
                # latency
                camera_obs_latency=0.145,
                robot_obs_latency=0.0001,
                gripper_obs_latency=0.01,
                robot_action_latency=0.2,
                gripper_action_latency=0.1,
                # camera_obs_latency=0.0,
                # robot_obs_latency=0.0,
                # gripper_obs_latency=0.0,
                # robot_action_latency=0.0,
                # gripper_action_latency=0.0,
                # obs
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                mirror_crop=mirror_crop,
                mirror_swap=mirror_swap,
                # action
                max_pos_speed=2.5,
                max_rot_speed=1.5,
                shm_manager=shm_manager) as env:
            
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)

            # load match_dataset
            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
                match_video_dir = match_dir.joinpath('videos')
                for vid_dir in match_video_dir.glob("*/"):
                    episode_idx = int(vid_dir.stem)
                    match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                    if match_video_path.exists():
                        img = None
                        with av.open(str(match_video_path)) as container:
                            stream = container.streams.video[0]
                            for frame in container.decode(stream):
                                img = frame.to_ndarray(format='rgb24')
                                break
                        episode_first_frame_map[episode_idx] = img
            print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

            # creating model
            # have to be done after fork to prevent 
            # duplicating CUDA context with ffmpeg nvenc
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = 16 # DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr

            device = torch.device('cuda')
            policy.eval().to(device)

            obs = env.get_obs()

            # print("Observation dictionary", obs["camera0_rgb"])

            # obs_policy = dict(obs)
            # obs_policy['robot0_eef_pos'] = obs_policy['robot0_eef_pos'] @ _M
            # obs_policy['robot0_eef_rot_axis_angle'] = obs_policy['robot0_eef_rot_axis_angle'] @ _M
            # episode_start_pose = np.concatenate([
            #         obs_policy[f'robot0_eef_pos'],
            #         obs_policy[f'robot0_eef_rot_axis_angle']
            #     ], axis=-1)[-1]
            episode_start_pose = np.concatenate([
                    obs[f'robot0_eef_pos'],
                    obs[f'robot0_eef_rot_axis_angle']
                ], axis=-1)[-1]
            
            # SE(3) state and limits
            v_max = 1.75   # m/s
            w_max = 1.5    # rad/s

            p_last = obs['robot0_eef_pos'][-1].copy()
            q_last = R.from_rotvec(obs['robot0_eef_rot_axis_angle'][-1].copy()).as_quat()

            # print("start pose", episode_start_pose)
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    episode_start_pose=episode_start_pose)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                
                # print("Observation", obs_dict)

                result = policy.predict_action(obs_dict)
                action = result['action_pred'][0].detach().to('cpu').numpy()
                # assert action.shape[-1] == 16
                assert action.shape[-1] == 10
                # Action produced in policy frame, convert to robot frame for execution
                action = get_real_umi_action(action, obs, action_pose_repr)
                # action_robot  = _map_pose_array(action_policy, _Minv)  # map back
                # action = action_robot                
                # assert action.shape[-1] == 10
                assert action.shape[-1] == 7
                del result

            print("Waiting to get to the stop button...")
            time.sleep(3.0)  # wait to get to stop button for safety!
            print('Ready!')

            while True:                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    image_observations = []  # Collect camera images

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")

                    iter_idx = 0
                    # last_action_end_time = time.time()
                    action_log = []
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        obs = env.get_obs()
                        
                        for frame in obs['camera0_rgb']:
                            image_observations.append(frame)

                        # obs_policy = dict(obs)
                        # obs_policy['robot0_eef_pos'] = obs_policy['robot0_eef_pos'] @ _M
                        # print("Transformed observation to policy frame:", obs_policy['robot0_eef_pos'])

                        # obs_policy['robot0_eef_rot_axis_angle'] = obs_policy['robot0_eef_rot_axis_angle'] @ _M
                        # print("Camera:", obs['camera0_rgb'].shape)
                        episode_start_pose = np.concatenate([
                            obs[f'robot0_eef_pos'],
                            obs[f'robot0_eef_rot_axis_angle']
                        ], axis=-1)[-1]
                        obs_timestamps = obs['timestamp']
                        # print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep,
                                episode_start_pose=episode_start_pose)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

                            result = policy.predict_action(obs_dict)
                            raw_action = result['action_pred'][0].detach().cpu().numpy()

                            # policy to robot mapping for execution
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)
                            # action_robot  = _map_pose_array(action_policy, _Minv)  # map back
                            # action = action_robot
                            action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]
                            this_target_poses = action

                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        # print("Is new:", is_new)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        for a, t in zip(this_target_poses, action_timestamps):
                            a = a.tolist()
                            # print("Actions", a)
                            action_log.append({
                                'timestamp': t,
                                'ee_pos_0': a[0],
                                'ee_pos_1': a[1],
                                'ee_pos_2': a[2],
                                'ee_rot_0': a[3],
                                'ee_rot_1': a[4],
                                'ee_rot_2': a[5]
                            })

                        # action_exec_latency = 0.01
                        # curr_time = time.time()
                        # is_new = action_timestamps > (curr_time + action_exec_latency)

                        # if np.sum(is_new) == 0:
                        #     # exceeded time budget -> take last step only (align both frames!)
                        #     robot_subset  = action_robot[[-1]]
                        #     policy_subset = action_policy[[-1]]
                        #     next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                        #     action_timestamp = eval_t_start + (next_step_idx) * dt
                        #     action_timestamps = np.array([action_timestamp])
                        # else:
                        #     robot_subset  = action_robot[is_new]
                        #     policy_subset = action_policy[is_new]
                        #     action_timestamps = action_timestamps[is_new]

                        # this_target_poses = robot_subset

                        # # unified logging: robot + policy in one row
                        # for a_robot, a_policy, t in zip(robot_subset, policy_subset, action_timestamps):
                        #     action_log.append({
                        #         'timestamp': t,
                        #         # robot-frame pose (what you executed)
                        #         'robot_x':  a_robot[0], 'robot_y':  a_robot[1], 'robot_z':  a_robot[2],
                        #         'robot_rx': a_robot[3], 'robot_ry': a_robot[4], 'robot_rz': a_robot[5],
                        #         # policy-frame pose (what the policy predicted, before mapping)
                        #         'policy_x':  a_policy[0], 'policy_y':  a_policy[1], 'policy_z':  a_policy[2],
                        #         'policy_rx': a_policy[3], 'policy_ry': a_policy[4], 'policy_rz': a_policy[5]
                        #     })

                        # execute actions
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            compensate_latency=True
                        )
                        # print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # _ = cv2.pollKey()
                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                # Stop episode
                                # Hand control back to human
                                print('Stopped.')
                                stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            stop_episode = True
                        if stop_episode:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)

                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    if len(action_log) > 0:
                        df = pd.DataFrame(action_log)
                        time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
                        csv_path = os.path.join(output, f"policy_actions_{time_now}.csv")
                        df.to_csv(csv_path, index=False)
                        print(f"Saved actions to {csv_path}")

                    # Save camera observations
                    if len(image_observations) > 0:
                        images_array = np.stack(image_observations, axis=0)  # Shape: (N, 224, 224, 3)
                        time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
                        npy_path = os.path.join(output, f"camera_observations_{time_now}.npy")
                        np.save(npy_path, images_array)
                        print(f"Saved {images_array.shape[0]} camera frames to {npy_path}")
                        print(f"Final shape: {images_array.shape}")
                    # stop robot.
                    env.end_episode()
                    
if __name__ == '__main__':
    main()

                        # action_exec_latency = 0.01
                        # curr_time = time.time()
                        # is_new = action_timestamps > (curr_time + action_exec_latency)
                        # # print("Is new:", is_new)
                        # if np.sum(is_new) == 0:
                        #     # exceeded time budget, still do something
                        #     this_target_poses = this_target_poses[[-1]]
                        #     # schedule on next available step
                        #     next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                        #     action_timestamp = eval_t_start + (next_step_idx) * dt
                        #     print('Over budget', action_timestamp - curr_time)
                        #     action_timestamps = np.array([action_timestamp])
                        # else:
                        #     this_target_poses = this_target_poses[is_new]
                        #     action_timestamps = action_timestamps[is_new]

                        # # Convert action to position and quaternion format for step limiting
                        # limited_actions = []
                        # for i, target_pose in enumerate(this_target_poses):
                        #     # Extract position and rotation from action
                        #     p_cmd = target_pose[:3]  # position
                            
                        #     rotvec = target_pose[3:6]
                        #     q_cmd = R.from_rotvec(rotvec).as_quat()
                        #     # Assume quaternion format
                        #     q_cmd = target_pose[3:7]
                        
                        #     # Apply SE(3) step limiter
                        #     p_limited, q_limited = _limit_se3_step(
                        #         p_prev=p_last, 
                        #         q_prev=q_last, 
                        #         p_cmd=p_cmd, 
                        #         q_cmd=q_cmd, 
                        #         v_max=v_max, 
                        #         w_max=w_max, 
                        #         dt=dt
                        #     )
                            
                        #     # Convert back to action format
                        #     rotvec_limited = R.from_quat(q_limited).as_rotvec()
                        #     limited_action = np.concatenate([p_limited, rotvec_limited])
                            
                        #     # Add gripper action if present
                        #     if len(target_pose) > 6:
                        #         limited_action = np.concatenate([limited_action, target_pose[6:]])
                            
                        #     limited_actions.append(limited_action)
                            
                        #     # Update last pose for next iteration
                        #     p_last = p_limited.copy()
                        #     q_last = q_limited.copy()

                        # # Convert back to numpy array
                        # this_target_poses = np.array(limited_actions)

                            # # Build rate-limited poses in SE(3) (position + quaternion with slerp cap).
                            # # p_last/q_last should be kept outside the loop (you already init them once before the episode).
                            # this_target_poses = []
                            # # action_timestamps = (np.arange(min(steps_per_inference, len(action)), dtype=np.float64)) * dt + obs_timestamps[-1]

                            # for i in range(len(action)):
                            #     # desired command from model
                            #     p_cmd = action[i][:3]
                            #     q_cmd = R.from_rotvec(action[i][3:6]).as_quat()  # (x,y,z,w)

                            #     # rate-limit around last commanded pose (handles hemisphere continuity internally)
                            #     p_safe, q_safe = _limit_se3_step(p_last, q_last, p_cmd, q_cmd, v_max, w_max, dt)

                            #     # update "last" for next step
                            #     p_last, q_last = p_safe, q_safe

                            #     # compose executable action: rotvec back from the safe quaternion
                            #     a_exec = np.zeros_like(action[0])
                            #     a_exec[:3]  = p_safe
                            #     a_exec[3:6] = R.from_quat(q_safe).as_rotvec()
                            #     a_exec[6]   = action[i][6]   # keep per-step gripper intent

                            #     this_target_poses.append(a_exec)

                            # this_target_poses = np.stack(this_target_poses, axis=0)

                            # print('Inference latency:', time.time() - s)

                            # # Final execution
                            # if len(this_target_poses) > 0:
                            #     env.exec_actions(
                            #         actions=np.stack(this_target_poses),
                            #         timestamps=np.array(action_timestamps),
                            #         compensate_latency=True
                            #     )
                            #     print(f"Submitted {len(this_target_poses)} steps of actions.")
                            # else:
                            #     print("No valid actions to submit.")