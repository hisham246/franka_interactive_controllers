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
from collections import deque
import cvxpy as cp

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

# LiPo
class ActionLiPo:
    def __init__(self, solver="CLARABEL", 
                 chunk_size=100, 
                 blending_horizon=10, 
                 action_dim=7, 
                 len_time_delay=0,
                 dt=0.0333,
                 epsilon_blending=0.02,
                 epsilon_path=0.003):
        """
        ActionLiPo (Action Lightweight Post-Optimizer) for action optimization.      
        Parameters:
        - solver: The solver to use for the optimization problem.
        - chunk_size: The size of the action chunk to optimize.
        - blending_horizon: The number of actions to blend with past actions.
        - action_dim: The dimension of the action space.
        - len_time_delay: The length of the time delay for the optimization.
        - dt: Time step for the optimization.
        - epsilon_blending: Epsilon value for blending actions.
        - epsilon_path: Epsilon value for path actions.
        """

        self.solver = solver
        self.N = chunk_size
        self.B = blending_horizon
        self.D = action_dim
        self.TD = len_time_delay

        self.dt = dt
        self.epsilon_blending = epsilon_blending
        self.epsilon_path = epsilon_path
        
        JM = 3  # margin for jerk calculation
        self.JM = JM
        self.epsilon = cp.Variable((self.N+JM, self.D)) # previous + 3 to consider previous vel/acc/jrk
        self.ref = cp.Parameter((self.N+JM, self.D),value=np.zeros((self.N+JM, self.D))) # previous + 3
        
        D_j = np.zeros((self.N+JM, self.N+JM))
        for i in range(self.N - 2):
            D_j[i, i]     = -1
            D_j[i, i+1]   = 3
            D_j[i, i+2]   = -3
            D_j[i, i+3]   = 1
        D_j = D_j / self.dt**3

        q_total = self.epsilon + self.ref  # (N, D)
        cost = cp.sum([cp.sum_squares(D_j @ q_total[:, d]) for d in range(self.D)])

        constraints = []

        constraints += [self.epsilon[self.B+JM:] <= self.epsilon_path]
        constraints += [self.epsilon[self.B+JM:] >= - self.epsilon_path]
        constraints += [self.epsilon[JM+1+self.TD:self.B+JM] <= self.epsilon_blending]
        constraints += [self.epsilon[JM+1+self.TD:self.B+JM] >= - self.epsilon_blending]
        constraints += [self.epsilon[0:JM+1+self.TD] == 0.0]

        np.set_printoptions(precision=3, suppress=True, linewidth=100)

        self.p = cp.Problem(cp.Minimize(cost), constraints)

        # Initialize the problem & warm up
        self.p.solve(warm_start=True, verbose=False, solver=self.solver, time_limit=0.05)
        
        self.log = []

    def solve(self, actions: np.ndarray, past_actions: np.ndarray, len_past_actions: int):
        """
        Solve the optimization problem with the given actions and past actions.
        Parameters:
        - actions: The current actions to optimize.
        - past_actions: The past actions to blend with.
        - len_past_actions: The number of past actions to consider for blending.
        Returns:
        - solved: The optimized actions after solving the problem.
        - ref: The reference actions used in the optimization.
        """

        blend_len = len_past_actions
        JM = self.JM
        self.ref.value[JM:] = actions.copy()
        
        if blend_len > 0:
            # update last actions
            self.ref.value[:JM+self.TD] = past_actions[-blend_len-JM:-blend_len + self.TD].copy()
            ratio_space = np.linspace(0, 1, blend_len-self.TD) # (B,1)    
            self.ref.value[JM+self.TD:blend_len+JM] = ratio_space[:, None] * actions[self.TD:blend_len] + (1 - ratio_space[:, None]) * past_actions[-blend_len+self.TD:]
        else: # blend_len == 0
            # update last actions
            self.ref.value[:JM] = actions[0]
            
        t0 = time.time()
        try:
            self.p.solve(warm_start=True, verbose=False, solver=self.solver, time_limit=0.05)
        except Exception as e:
            return None, e
        t1 = time.time()

        solved_time = t1 - t0
        self.solved = self.epsilon.value.copy() + self.ref.value.copy()

        self.log.append({
            "time": solved_time,
            "epsilon": self.epsilon.value.copy(),
            "ref": self.ref.value.copy(),
            "solved": self.solved.copy()
        })

        return self.solved[JM:].copy(), self.ref.value[JM:].copy()

    def get_log(self):
        return self.log

    def reset_log(self):
        self.log = []

    def print_solved_times(self):
        if self.log:
            avg_time = np.mean([entry["time"] for entry in self.log])
            std_time = np.std([entry["time"] for entry in self.log])
            num_logs = len(self.log)
            print(f"Number of logs: {num_logs}")
            print(f"Average solved time: {avg_time:.4f} seconds, Std: {std_time:.4f} seconds")
        else:
            print("No logs available.")

def split_pose_gripper(actions7):
    # actions7: [..., 7] = [x,y,z,rx,ry,rz, g]
    return actions7[:, :6].copy(), actions7[:, 6:7].copy()

def merge_pose_gripper(pose6, grip1):
    return np.concatenate([pose6, grip1], axis=-1)

def make_timestamps(start_from, n, dt):
    return (np.arange(n, dtype=np.float64) * dt) + start_from

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

# ---- geodesic mean around a reference quaternion ----
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

# ---- SE(3) step limiter ----
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


def main():
    output = '/home/hisham246/uwaterloo/pickplace_test_unet'
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
    # ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/surface_wiping_unet_position_control.ckpt'
    ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/diffusion_unet_pickplace_2.ckpt'


    # Compliance policy unet
    # ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/diffusion_unet_compliance_trial_2.ckpt'

    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']

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

            # LiPo knobs
            H  = steps_per_inference         # horizon length predicted each call
            B  = 3                           # overlap (in steps) to blend/optimize (try 3–4 @10 Hz)
            EPS_B = 0.02                     # blend-zone bound (rad or m), tune
            EPS_P = 0.003                    # path bound (tighter), tune
            DT    = 1.0 / frequency

            # LiPo instance (optimize 6D pose; gripper passes through)
            lipo = ActionLiPo(
                solver="CLARABEL",
                chunk_size=H,
                blending_horizon=B,
                action_dim=6,
                len_time_delay=0,            # will be set per-solve with measured TD
                dt=DT,
                epsilon_blending=EPS_B,
                epsilon_path=EPS_P
            )

            # Rolling plan of future actions (pose+time) that we’ve already optimized
            planned_actions   = deque()   # each item: (7D action, timestamp)
            last_optimized6   = None      # numpy [N,6] of last smoothed chunk (for LiPo "past")
            last_opt_tail6    = None      # tail we keep for blending
            last_tail_len     = 0         # how many past steps we offer for blending
            JM = lipo.JM                  # LiPo’s jerk-margin (3)

            obs = env.get_obs()
            # print("Observation", obs)           
            episode_start_pose = np.concatenate([
                    obs[f'robot0_eef_pos'],
                    obs[f'robot0_eef_rot_axis_angle']
                ], axis=-1)[-1]
            
            # ---- SE(3) state & limits ----
            v_max = 0.75   # m/s (tune)
            w_max = 0.9    # rad/s (tune)

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
                result = policy.predict_action(obs_dict)
                action = result['action_pred'][0].detach().to('cpu').numpy()
                # assert action.shape[-1] == 16
                assert action.shape[-1] == 10
                action = get_real_umi_action(action, obs, action_pose_repr)
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
                        effective_stride = H - B
                        t_cycle_end = t_start + (iter_idx + effective_stride) * dt # get obs
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
                            action7 = get_real_umi_action(raw_action, obs, action_pose_repr)  # (H,7)
                            del result

                            # Timestamps for this raw chunk
                            action_timestamps = make_timestamps(obs_timestamps[-1], len(action7), DT)

                            # Measure effective delay (inference + exec) -> steps
                            policy_time = time.time() - s
                            action_exec_latency = 0.01
                            effective_delay_sec = policy_time + action_exec_latency
                            TD = int(np.clip(np.round(effective_delay_sec / DT), 0, H-1))

                            # Prepare LiPo inputs (pose only)
                            pose6, grip1 = split_pose_gripper(action7)

                            # Decide how much past to blend with (tail from last optimized chunk)
                            # Keep at most B steps; also ensure we provide the JM+TD context rows
                            if last_opt_tail6 is None:
                                blend_len = 0
                                past_for_lipo = np.zeros((JM + TD, 6), dtype=np.float64)  # dummy; will be overwritten inside solver
                            else:
                                # we want [JM + blend_len] rows available for LiPo:
                                tail = last_opt_tail6
                                blend_len = min(B, tail.shape[0])
                                past_for_lipo = np.vstack([
                                    # give LiPo its JM context (last 3 samples preceding blending window)
                                    tail[max(0, tail.shape[0] - (blend_len + JM)) : max(0, tail.shape[0] - blend_len), :],
                                    # then the blend window that overlaps with the new chunk
                                    tail[-blend_len:, :]
                                ])
                                # If tail was shorter than JM+blend_len, LiPo still has internal zeros; it’s ok over short horizons.

                            # Configure LiPo’s per-call delay in steps
                            lipo.TD = TD

                            # Solve LiPo (returns (N,6) smoothed and reference used)
                            smoothed6, _ref6 = lipo.solve(actions=pose6, past_actions=past_for_lipo, len_past_actions=blend_len)
                            if smoothed6 is None:
                                # Fallback on raw in case the solver timed out
                                smoothed6 = pose6.copy()

                            # Re-attach gripper (not optimized)
                            smoothed7 = merge_pose_gripper(smoothed6, grip1)

                            # Keep a new tail for the next overlap: last B samples of the smoothed chunk
                            last_opt_tail6 = smoothed6[-B:].copy()
                            last_tail_len  = last_opt_tail6.shape[0]

                            action_exec_latency = 0.01
                            curr_time = time.time()
                            send_t0 = curr_time + action_exec_latency
                            is_new = action_timestamps > send_t0                            

                            print("Is new:", is_new)
                            if np.sum(is_new) == 0:
                                # budget overrun: send 1 step in the nearest slot
                                this_target_poses = smoothed7[[-1]]
                                next_step_idx = int(np.ceil((curr_time - eval_t_start) / DT))
                                action_timestamps = np.array([eval_t_start + next_step_idx * DT], dtype=np.float64)
                            else:
                                this_target_poses = smoothed7[is_new]
                                action_timestamps = action_timestamps[is_new]

                            for a, t in zip(this_target_poses, action_timestamps):
                                a = a.tolist()
                                action_log.append({
                                    'timestamp': t,
                                    'ee_pos_0': a[0],
                                    'ee_pos_1': a[1],
                                    'ee_pos_2': a[2],
                                    'ee_rot_0': a[3],
                                    'ee_rot_1': a[4],
                                    'ee_rot_2': a[5]
                                })

                            # execute actions
                            env.exec_actions(
                                actions=this_target_poses,
                                timestamps=action_timestamps,
                                compensate_latency=True
                            )
                            print(f"Submitted {len(this_target_poses)} steps of actions.")

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
                        iter_idx += effective_stride

                except KeyboardInterrupt:
                    print("Interrupted!")
                    if len(action_log) > 0:
                        df = pd.DataFrame(action_log)
                        time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
                        csv_path = os.path.join(output, f"policy_actions_{time_now}.csv")
                        df.to_csv(csv_path, index=False)
                        print(f"Saved actions to {csv_path}")
                    # stop robot.
                    env.end_episode()
                    
if __name__ == '__main__':
    main()