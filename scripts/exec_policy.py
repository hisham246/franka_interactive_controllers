"""
Run a policy on the real robot.
"""
import rospy
import sys
import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import av
import cv2
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
import json

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

def main():
    output = '/home/hisham246/uwaterloo/test'
    gripper_ip = '129.97.71.27'
    gripper_port = 4242
    match_dataset = None
    match_camera = 0
    steps_per_inference = 1
    vis_camera_idx = 0
    max_duration = 120
    frequency = 10.0
    no_mirror = False
    sim_fov = None
    camera_intrinsics = None
    mirror_crop = False
    mirror_swap = False
    temporal_ensembling = True 
            
    # Diffusion Transformer
    # ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/diffusion_transformer_pickplace.ckpt'

    # Diffusion UNet
    # ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/diffusion_unet_pickplace_2.ckpt'

    # Compliance policy unet
    ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/diffusion_unet_compliance_trial_2.ckpt'

    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']

    cfg._target_ = "diffusion_policy.train_diffusion_unet_compliance_workspace.TrainDiffusionUnetComplianceWorkspace"
    cfg.policy._target_ = "diffusion_policy.diffusion_unet_timm_policy.DiffusionUnetTimmPolicy"
    cfg.policy.obs_encoder._target_ = "policy_utils.timm_obs_encoder.TimmObsEncoder"
    cfg.ema._target_ = "policy_utils.ema_model.EMAModel"

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
                # camera_obs_latency=0.145,
                # robot_obs_latency=0.0001,
                # gripper_obs_latency=0.01,
                # robot_action_latency=0.2,
                # gripper_action_latency=0.1,
                camera_obs_latency=0.0,
                robot_obs_latency=0.0,
                gripper_obs_latency=0.0,
                robot_action_latency=0.0,
                gripper_action_latency=0.0,
                # obs
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                mirror_crop=mirror_crop,
                mirror_swap=mirror_swap,
                # action
                max_pos_speed=1.5,
                max_rot_speed=2.0,
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
            # print("Observation", obs)           
            episode_start_pose = np.concatenate([
                    obs[f'robot0_eef_pos'],
                    obs[f'robot0_eef_rot_axis_angle']
                ], axis=-1)[-1]
            # print("start pose", episode_start_pose)
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    episode_start_pose=episode_start_pose)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action_pred'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 16
                # assert action.shape[-1] == 10
                action = get_real_umi_action(action, obs, action_pose_repr)
                assert action.shape[-1] == 10
                # assert action.shape[-1] == 7
                del result

            print("Waiting to get to the stop button...")
            time.sleep(3.0)  # wait to get to stop button for safety!
            print('Ready!')

            while  not rospy.is_shutdown():                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    if temporal_ensembling:
                        max_steps = int(max_duration * frequency) + steps_per_inference
                        temporal_action_buffer = [None] * max_steps

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
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)

                            # print("Actions", action)

                            # print('Inference latency:', time.time() - s)
                            if temporal_ensembling:
                                for i, a in enumerate(action):
                                    target_step = iter_idx + i
                                    if target_step < len(temporal_action_buffer):
                                        if temporal_action_buffer[target_step] is None:
                                            temporal_action_buffer[target_step] = []
                                        temporal_action_buffer[target_step].append(a)
                            for a, t in zip(action, obs_timestamps[-1] + dt + np.arange(len(action)) * dt):
                                a = a.tolist()
                                action_log.append({'timestamp': t, 
                                                    'ee_pos_0': a[0],
                                                    'ee_pos_1': a[1],
                                                    'ee_pos_2': a[2],
                                                    'ee_rot_0': a[3],
                                                    'ee_rot_1': a[4],
                                                    'ee_rot_2': a[5]})

                        action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]
                        
                        if temporal_ensembling:
                            ensembled_actions = []
                            valid_timestamps = []
                            for i in range(len(action)):
                                target_step = iter_idx + i
                                if target_step >= len(temporal_action_buffer):
                                    continue
                                cached = temporal_action_buffer[target_step]
                                if cached is None or len(cached) == 0:
                                    continue
                                k = 0.01
                                n = len(cached)
                                weights = np.exp(-k * np.arange(n))
                                weights = weights / weights.sum()
                                ensembled_action = np.average(np.stack(cached), axis=0, weights=weights)
                                ensembled_actions.append(ensembled_action)
                                valid_timestamps.append(action_timestamps[i])

                            this_target_poses = ensembled_actions
                            action_timestamps = valid_timestamps
                        else:
                            this_target_poses = action

                        # Final execution
                        if len(this_target_poses) > 0:
                            env.exec_actions(
                                actions=np.stack(this_target_poses),
                                timestamps=np.array(action_timestamps),
                                compensate_latency=True
                            )
                            # print(f"Submitted {len(this_target_poses)} steps of actions.")
                        else:
                            print("No valid actions to submit.")


                        # # visualize
                        # episode_id = env.replay_buffer.n_episodes
                        # if mirror_crop:
                        #     vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
                        #     crop_img = obs['camera0_rgb_mirror_crop'][-1]
                        #     vis_img = np.concatenate([vis_img, crop_img], axis=1)
                        # else:
                        #     vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
                        # text = 'Episode: {}, Time: {:.1f}'.format(
                        #     episode_id, time.monotonic() - t_start
                        # )
                        # cv2.putText(
                        #     vis_img,
                        #     text,
                        #     (10,20),
                        #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        #     fontScale=0.5,
                        #     thickness=1,
                        #     color=(255,255,255)
                        # )
                        # cv2.imshow('default', vis_img[...,::-1])

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
                    # stop robot.
                    env.end_episode()
                    # if len(action_log) > 0:
                    #     df = pd.DataFrame(action_log)
                    #     # csv_path = os.path.join(output, f"policy_actions_episode_{episode_id}.csv")
                    #     df.to_csv(csv_path, index=False)
                    #     print(f"Saved actions to {csv_path}")
                    
                print("Stopped.")

# %%
if __name__ == '__main__':
    main()