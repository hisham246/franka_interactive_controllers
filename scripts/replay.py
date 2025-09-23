
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation as R
import os
import glob
from policy_utils.pose_repr_util import convert_pose_mat_rep
from policy_utils.pose_util import (
    pose_to_mat, mat_to_pose10d, mat_to_pose, pose10d_to_mat)

from policy_utils.real_inference_util import get_real_umi_action
from policy_utils.pytorch_util import dict_apply
import torch
from matplotlib.animation import FuncAnimation

# Transformation: policy frame → robot frame
POLICY_TO_ROBOT_ROT = np.array([
    [0,  1,  0],
    [-1, 0,  0],
    [0,  0,  1]
], dtype=float)

ROBOT_TO_POLICY_ROT = POLICY_TO_ROBOT_ROT.T  # Inverse is transpose

def save_episode_predictions_full_horizon(results, out_csv_path, save_raw_policy=False):
    """
    Save ALL predicted actions (entire horizon) for each timestep to CSV.
    
    CSV columns: time, horizon_step, pos_x, pos_y, pos_z, rot6d_0, rot6d_1, rot6d_2, rot6d_3, rot6d_4, rot6d_5, gripper
    """
    T = len(results['predicted_actions'])
    rows = []
    
    for t in range(T):
        if save_raw_policy and 'raw_policy_actions' in results:
            # Use raw 10D policy output
            actions = results['raw_policy_actions'][t]
        else:
            # Use converted 7D actions (convert to similar format)
            actions = results['predicted_actions'][t]
        
        if len(actions) == 0:
            # no prediction — add one NaN row
            if save_raw_policy:
                rows.append([results['timestamps'][t], 0] + [float('nan')]*10)
            else:
                rows.append([results['timestamps'][t], 0] + [float('nan')]*7)
            continue
        
        # Save all horizon steps
        for h, action in enumerate(actions):
            if save_raw_policy:
                # Raw 10D format
                row = [results['timestamps'][t], h] + action.tolist()
            else:
                # 7D format: pad with NaN for missing dimensions to match 10D structure
                x, y, z = action[0], action[1], action[2]
                rx, ry, rz = action[3], action[4], action[5]
                gripper = action[6] if len(action) > 6 else 0.0
                # Convert rotation vector to 6D representation (placeholder - you'd need proper conversion)
                row = [results['timestamps'][t], h, x, y, z, rx, ry, rz, 0.0, 0.0, gripper]
            
            rows.append(row)
    
    # Build DataFrame and save
    if save_raw_policy:
        cols = ['time','horizon_step','pos_x','pos_y','pos_z','rot6d_0','rot6d_1','rot6d_2','rot6d_3','rot6d_4','rot6d_5','gripper']
    else:
        cols = ['time','horizon_step','x','y','z','rx','ry','rz','pad1','pad2','gripper']
    
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_csv_path, index=False)
    print(f"Saved full horizon predictions to: {out_csv_path}")

def rotvec_to_quat(rv: np.ndarray) -> np.ndarray:
    """[rx,ry,rz] (axis-angle, rad) -> quaternion [x,y,z,w]."""
    return R.from_rotvec(rv).as_quat()

def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Quaternion [x,y,z,w] -> 3x3 rotation matrix."""
    return R.from_quat(quat).as_matrix()

def create_coordinate_frame_from_quat(position, quat, scale=0.03):
    """Return axis vectors (in world coords) for a frame oriented by 'quat'."""
    rot_matrix = quat_to_matrix(quat)        # 3x3
    x_axis = np.array([scale, 0, 0])
    y_axis = np.array([0, scale, 0])
    z_axis = np.array([0, 0, scale])
    x_w = rot_matrix @ x_axis
    y_w = rot_matrix @ y_axis
    z_w = rot_matrix @ z_axis
    return x_w, y_w, z_w

def quat_angle_error(q1, q2):
    # both [x,y,z,w], unit quats. angle = 2*arccos(|dot|) to handle double cover
    d = abs(np.dot(q1, q2))
    d = np.clip(d, -1.0, 1.0)
    return 2.0 * np.arccos(d)

class PolicyActionSimulator:
    """Simulate policy action prediction from observations."""
    
    def __init__(self, obs_pose_repr='relative', action_pose_repr='relative', obs_horizon=2, action_horizon=8, 
                 policy=None, device=None):
        self.obs_pose_repr = obs_pose_repr
        self.action_pose_repr = action_pose_repr
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.policy = policy
        self.device = device if device is not None else torch.device('cpu')
    
    def load_episode_data(self, episode_dir):
        """Load episode data from numpy files."""
        positions = np.load(os.path.join(episode_dir, "robot0_eef_pos.npy"))
        rotations = np.load(os.path.join(episode_dir, "robot0_eef_rot_axis_angle.npy"))
        timestamps = np.load(os.path.join(episode_dir, "timestamp.npy"))
        gripper_width = np.load(os.path.join(episode_dir, "robot0_gripper_width.npy"))
        demo_start_pose = np.load(os.path.join(episode_dir, "robot0_demo_start_pose.npy"))
        demo_end_pose = np.load(os.path.join(episode_dir, "robot0_demo_end_pose.npy"))
        camera_rgb = np.load(os.path.join(episode_dir, "camera0_rgb.npy"))
        
        return {
            'positions': positions,
            'rotations': rotations, 
            'timestamps': timestamps,
            'gripper_width': gripper_width,
            'demo_start_pose': demo_start_pose,
            'demo_end_pose': demo_end_pose,
            'camera_rgb': camera_rgb
        }
    
    def simulate_obs_processing(self, episode_data, timestep):
        """Simulate how observations get processed for the policy."""
        positions = episode_data['positions']
        rotations = episode_data['rotations']
        camera_rgb = episode_data['camera_rgb']
        gripper_width = episode_data['gripper_width']
        demo_start_pose = episode_data['demo_start_pose']
        
        # Get observation window
        start_idx = max(0, timestep - self.obs_horizon + 1)
        end_idx = timestep + 1
        
        obs_positions = positions[start_idx:end_idx]
        obs_rotations = rotations[start_idx:end_idx]
        obs_images = camera_rgb[start_idx:end_idx]
        obs_gripper = gripper_width[start_idx:end_idx]
        
        # Pad if necessary (repeat first observation)
        if len(obs_positions) < self.obs_horizon:
            padding_needed = self.obs_horizon - len(obs_positions)
            obs_positions = np.concatenate([
                np.tile(obs_positions[0:1], (padding_needed, 1)),
                obs_positions
            ])
            obs_rotations = np.concatenate([
                np.tile(obs_rotations[0:1], (padding_needed, 1)),
                obs_rotations
            ])
            obs_images = np.concatenate([
                np.tile(obs_images[0:1], (padding_needed, 1, 1, 1)),
                obs_images
            ])
            obs_gripper = np.concatenate([
                np.tile(obs_gripper[0:1], (padding_needed, 1)),
                obs_gripper
            ])
        
        # Process images exactly like inference: uint8 -> float32 and THWC -> TCHW
        if obs_images.dtype == np.uint8:
            obs_images = obs_images.astype(np.float32) / 255.0
        # Convert THWC to TCHW
        obs_images = np.moveaxis(obs_images, -1, 1)
        
        # Convert poses to matrices and apply representation conversion
        obs_poses = np.concatenate([obs_positions, obs_rotations], axis=-1)
        pose_matrices = pose_to_mat(obs_poses)
        
        # Apply observation pose representation (relative to most recent pose)
        if self.obs_pose_repr != 'abs':
            current_pose_mat = pose_matrices[-1]  # Most recent pose
            processed_pose_mat = convert_pose_mat_rep(
                pose_matrices,
                base_pose_mat=current_pose_mat,
                pose_rep=self.obs_pose_repr,
                backward=False
            )
        else:
            processed_pose_mat = pose_matrices
        
        # Convert to 10D representation for robot poses
        obs_10d = mat_to_pose10d(processed_pose_mat)
        
        # Compute relative pose with respect to episode start (like in UmiDataset)
        # Your real robot starting pose (column-major → row-major 4x4 matrix)
        robot_start_pose_colmaj = np.array([
            0.6054297437135656, -0.0005734414600870319, 0.7958864973046205, 0.0,
            5.205885202447247e-05, -0.9999901197196974, -0.0007601005593292354, 0.0,
            0.7958943348494556, 0.0005016300454869032, -0.605435344295194, 0.0,
            0.312387626442811, 0.3905454094029647, 0.4898630997026142, 1.0
        ], dtype=float).reshape(4,4, order='F')  # Column-major to row-major

        # Convert robot starting pose to policy frame
        robot_R = robot_start_pose_colmaj[:3,:3]
        robot_t = robot_start_pose_colmaj[:3,3]

        policy_R = ROBOT_TO_POLICY_ROT @ robot_R
        policy_t = ROBOT_TO_POLICY_ROT @ robot_t

        start_pose_mat = np.eye(4)
        start_pose_mat[:3,:3] = policy_R
        start_pose_mat[:3,3] = policy_t        
        
        rel_obs_pose_mat = convert_pose_mat_rep(
            pose_matrices,
            base_pose_mat=start_pose_mat,
            pose_rep='relative',
            backward=False)
        
        rel_obs_pose_10d = mat_to_pose10d(rel_obs_pose_mat)
        
        # Prepare observation dict exactly like inference
        obs_dict = {
            'robot0_eef_pos': obs_10d[..., :3],  # 3D positions
            'robot0_eef_rot_axis_angle': obs_10d[..., 3:9],  # 6D rotations
            'robot0_eef_rot_axis_angle_wrt_start': rel_obs_pose_10d[..., 3:9],  # 6D rotations relative to start
            'camera0_rgb': obs_images,  # Processed images TCHW format
            'robot0_gripper_width': obs_gripper  # Gripper width
        }
        
        return obs_dict, obs_poses[-1]  # Return processed obs and current absolute pose
    
    def simulate_policy_prediction(self, obs_dict):
        """Run actual policy prediction."""
        
        if self.policy is None:
            raise ValueError("Policy must be provided. Set policy_path in the main function.")
        
        # Run actual policy inference
        with torch.no_grad():
            # Convert to torch tensors and move to device
            obs_dict_torch = dict_apply(obs_dict, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
            
            # Run policy prediction
            result = self.policy.predict_action(obs_dict_torch)
            action_pred = result['action_pred'][0].detach().to('cpu').numpy()

            print(f"Predicted action: {action_pred}")
            
            return action_pred
    
    def process_episode(self, csv_file):
        """Process an entire episode and return action predictions."""
        positions, rotations, timestamps = self.load_episode_data(csv_file)
        
        predicted_actions = []
        processed_observations = []
        current_poses = []
        
        # Simulate policy execution at each timestep
        for t in range(len(positions)):
            # Process observations
            obs_10d, current_pose = self.simulate_obs_processing(positions, rotations, t)
            
            # Simulate policy prediction
            predicted_action_10d = self.simulate_policy_prediction(obs_10d)
            
            # Convert policy output back to robot actions
            # Create mock environment observations
            env_obs = {
                'robot0_eef_pos': positions[max(0, t-1):t+1],  # Recent positions
                'robot0_eef_rot_axis_angle': rotations[max(0, t-1):t+1]  # Recent rotations
            }
            
            # Convert to executable actions
            executable_actions = []
            for action_step in predicted_action_10d:
                real_action = get_real_umi_action(
                    action_step, env_obs, self.action_pose_repr
                )
                executable_actions.append(real_action)
            
            predicted_actions.append(executable_actions)
            processed_observations.append(obs_10d)
            current_poses.append(current_pose)
        
        return {
            'predicted_actions': predicted_actions,
            'processed_observations': processed_observations,
            'current_poses': current_poses,
            'original_positions': positions,
            'original_rotations': rotations,
            'timestamps': timestamps
        }

def visualize_action_predictions(results, episode_name):
    """Visualize the action predictions and transformations."""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    timestamps = results['timestamps']
    original_positions = results['original_positions']
    original_rotations = results['original_rotations']
    predicted_actions = results['predicted_actions']
    
    # Extract first action step from each prediction for visualization
    predicted_positions = []
    predicted_rotations = []
    
    for actions in predicted_actions:
        if len(actions) > 0:
            first_action = actions[0]  # First action step
            predicted_positions.append(first_action[:3])
            predicted_rotations.append(first_action[3:6])
        else:
            predicted_positions.append([0, 0, 0])
            predicted_rotations.append([0, 0, 0])
    
    predicted_positions = np.array(predicted_positions)
    predicted_rotations = np.array(predicted_rotations)
    
    # Plot 1: Original vs Predicted Positions
    axes[0, 0].plot(timestamps, original_positions[:, 0], 'b-', label='Original X', linewidth=2)
    axes[0, 0].plot(timestamps, predicted_positions[:, 0], 'r--', label='Predicted X', alpha=0.7)
    axes[0, 0].set_ylabel('X Position (m)')
    axes[0, 0].set_title('X Position: Original vs Predicted')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].plot(timestamps, original_positions[:, 1], 'b-', label='Original Y', linewidth=2)
    axes[1, 0].plot(timestamps, predicted_positions[:, 1], 'r--', label='Predicted Y', alpha=0.7)
    axes[1, 0].set_ylabel('Y Position (m)')
    axes[1, 0].set_title('Y Position: Original vs Predicted')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[2, 0].plot(timestamps, original_positions[:, 2], 'b-', label='Original Z', linewidth=2)
    axes[2, 0].plot(timestamps, predicted_positions[:, 2], 'r--', label='Predicted Z', alpha=0.7)
    axes[2, 0].set_ylabel('Z Position (m)')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_title('Z Position: Original vs Predicted')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 2: Original vs Predicted Rotations
    axes[0, 1].plot(timestamps, original_rotations[:, 0], 'b-', label='Original Rx', linewidth=2)
    axes[0, 1].plot(timestamps, predicted_rotations[:, 0], 'r--', label='Predicted Rx', alpha=0.7)
    axes[0, 1].set_ylabel('Rx Rotation (rad)')
    axes[0, 1].set_title('Rx Rotation: Original vs Predicted')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 1].plot(timestamps, original_rotations[:, 1], 'b-', label='Original Ry', linewidth=2)
    axes[1, 1].plot(timestamps, predicted_rotations[:, 1], 'r--', label='Predicted Ry', alpha=0.7)
    axes[1, 1].set_ylabel('Ry Rotation (rad)')
    axes[1, 1].set_title('Ry Rotation: Original vs Predicted')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[2, 1].plot(timestamps, original_rotations[:, 2], 'b-', label='Original Rz', linewidth=2)
    axes[2, 1].plot(timestamps, predicted_rotations[:, 2], 'r--', label='Predicted Rz', alpha=0.7)
    axes[2, 1].set_ylabel('Rz Rotation (rad)')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_title('Rz Rotation: Original vs Predicted')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Action Prediction Simulation - {episode_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def create_coordinate_frame(position, rotation_vec, scale=0.05):
    """Create coordinate frame vectors for visualization."""
    # Convert axis-angle to rotation matrix
    rot = st.Rotation.from_rotvec(rotation_vec)
    rot_matrix = rot.as_matrix()
    
    # Define unit vectors for x, y, z axes
    x_axis = np.array([scale, 0, 0])
    y_axis = np.array([0, scale, 0])
    z_axis = np.array([0, 0, scale])
    
    # Transform axes by rotation
    x_transformed = rot_matrix @ x_axis
    y_transformed = rot_matrix @ y_axis
    z_transformed = rot_matrix @ z_axis
    
    return x_transformed, y_transformed, z_transformed

def animate_predicted_vs_ground_truth(results, episode_name):
    """Create 3D animation comparing predicted vs ground truth trajectories."""
    
    timestamps = results['timestamps']
    original_positions = results['original_positions']
    original_rotations = results['original_rotations']
    predicted_actions = results['predicted_actions']
    
    # Extract first action step from each prediction for animation
    predicted_positions = []
    predicted_rotations = []
    
    for actions in predicted_actions:
        if len(actions) > 0:
            first_action = actions[0]
            predicted_positions.append(first_action[:3])
            predicted_rotations.append(first_action[3:6])
        else:
            predicted_positions.append([0, 0, 0])
            predicted_rotations.append([0, 0, 0])
    
    predicted_positions = np.array(predicted_positions)
    predicted_rotations = np.array(predicted_rotations)

    gt_quats   = np.array([rotvec_to_quat(rv) for rv in original_rotations])
    pred_quats = np.array([rotvec_to_quat(rv) for rv in predicted_rotations])
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate bounds for better visualization
    all_positions = np.concatenate([original_positions, predicted_positions])
    pos_min = all_positions.min(axis=0) - 0.05
    pos_max = all_positions.max(axis=0) + 0.05
    
    # Set equal aspect ratio
    max_range = np.array([pos_max[0]-pos_min[0], 
                         pos_max[1]-pos_min[1], 
                         pos_max[2]-pos_min[2]]).max() / 2.0
    mid_x = (pos_max[0] + pos_min[0]) * 0.5
    mid_y = (pos_max[1] + pos_min[1]) * 0.5
    mid_z = (pos_max[2] + pos_min[2]) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Initialize plot elements
    gt_trajectory, = ax.plot([], [], [], 'b-', alpha=0.8, linewidth=3, label='Ground Truth')
    pred_trajectory, = ax.plot([], [], [], 'r--', alpha=0.8, linewidth=3, label='Predicted')
    gt_point, = ax.plot([], [], [], 'bo', markersize=10, label='GT Current')
    pred_point, = ax.plot([], [], [], 'ro', markersize=10, label='Pred Current')
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title(f'Policy vs Ground Truth - {episode_name}')
    ax.legend()
    
    # Store reference to coordinate frame arrows
    coordinate_arrows = []
    
    # Animation function
    def animate(frame):
        # Clear previous coordinate frame arrows
        for arrow in coordinate_arrows:
            arrow.remove()
        coordinate_arrows.clear()
        
        # Show trajectories up to current frame
        if frame > 0:
            gt_trajectory.set_data_3d(original_positions[:frame+1, 0], 
                                    original_positions[:frame+1, 1], 
                                    original_positions[:frame+1, 2])
            pred_trajectory.set_data_3d(predicted_positions[:frame+1, 0], 
                                       predicted_positions[:frame+1, 1], 
                                       predicted_positions[:frame+1, 2])
        
        # Current positions
        gt_pos = original_positions[frame]
        pred_pos = predicted_positions[frame]
        
        gt_point.set_data_3d([gt_pos[0]], [gt_pos[1]], [gt_pos[2]])
        pred_point.set_data_3d([pred_pos[0]], [pred_pos[1]], [pred_pos[2]])
        
        # # Create coordinate frames
        # gt_x, gt_y, gt_z = create_coordinate_frame(gt_pos, original_rotations[frame], scale=0.03)
        # pred_x, pred_y, pred_z = create_coordinate_frame(pred_pos, predicted_rotations[frame], scale=0.03)
        
        # # Draw ground truth coordinate frame (solid colors)
        # gt_x_arrow = ax.quiver(gt_pos[0], gt_pos[1], gt_pos[2],
        #                       gt_x[0], gt_x[1], gt_x[2], 
        #                       color='darkred', alpha=0.8, linewidth=2)
        # gt_y_arrow = ax.quiver(gt_pos[0], gt_pos[1], gt_pos[2],
        #                       gt_y[0], gt_y[1], gt_y[2], 
        #                       color='darkgreen', alpha=0.8, linewidth=2)
        # gt_z_arrow = ax.quiver(gt_pos[0], gt_pos[1], gt_pos[2],
        #                       gt_z[0], gt_z[1], gt_z[2], 
        #                       color='darkblue', alpha=0.8, linewidth=2)
        
        # # Draw predicted coordinate frame (lighter colors)
        # pred_x_arrow = ax.quiver(pred_pos[0], pred_pos[1], pred_pos[2],
        #                         pred_x[0], pred_x[1], pred_x[2], 
        #                         color='darkred', alpha=0.6, linewidth=2)
        # pred_y_arrow = ax.quiver(pred_pos[0], pred_pos[1], pred_pos[2],
        #                         pred_y[0], pred_y[1], pred_y[2], 
        #                         color='darkgreen', alpha=0.6, linewidth=2)
        # pred_z_arrow = ax.quiver(pred_pos[0], pred_pos[1], pred_pos[2],
        #                         pred_z[0], pred_z[1], pred_z[2], 
        #                         color='darkblue', alpha=0.6, linewidth=2)
        
        # coordinate_arrows.extend([gt_x_arrow, gt_y_arrow, gt_z_arrow,
        #                          pred_x_arrow, pred_y_arrow, pred_z_arrow])

        # Create coordinate frames from QUATs now
        gt_x, gt_y, gt_z = create_coordinate_frame_from_quat(gt_pos,   gt_quats[frame],   scale=0.03)
        pr_x, pr_y, pr_z = create_coordinate_frame_from_quat(pred_pos, pred_quats[frame], scale=0.03)

        # Draw ground-truth frame
        gt_x_arrow = ax.quiver(gt_pos[0], gt_pos[1], gt_pos[2], gt_x[0], gt_x[1], gt_x[2],
                            color='darkred', alpha=0.8, linewidth=2)
        gt_y_arrow = ax.quiver(gt_pos[0], gt_pos[1], gt_pos[2], gt_y[0], gt_y[1], gt_y[2],
                            color='darkgreen', alpha=0.8, linewidth=2)
        gt_z_arrow = ax.quiver(gt_pos[0], gt_pos[1], gt_pos[2], gt_z[0], gt_z[1], gt_z[2],
                            color='darkblue', alpha=0.8, linewidth=2)

        # Draw predicted frame (lighter)
        pr_x_arrow = ax.quiver(pred_pos[0], pred_pos[1], pred_pos[2], pr_x[0], pr_x[1], pr_x[2],
                            color='darkred', alpha=0.6, linewidth=2)
        pr_y_arrow = ax.quiver(pred_pos[0], pred_pos[1], pred_pos[2], pr_y[0], pr_y[1], pr_y[2],
                            color='darkgreen', alpha=0.6, linewidth=2)
        pr_z_arrow = ax.quiver(pred_pos[0], pred_pos[1], pred_pos[2], pr_z[0], pr_z[1], pr_z[2],
                            color='darkblue', alpha=0.6, linewidth=2)

        coordinate_arrows.extend([gt_x_arrow, gt_y_arrow, gt_z_arrow,
                                pr_x_arrow, pr_y_arrow, pr_z_arrow])
                
        # Calculate error metrics
        pos_error = np.linalg.norm(gt_pos - pred_pos)
        # rot_error = np.linalg.norm(original_rotations[frame] - predicted_rotations[frame])
        rot_error = quat_angle_error(gt_quats[frame], pred_quats[frame])  # radians
        
        # Update title with current info and errors
        ax.set_title(f'Policy vs Ground Truth - {episode_name}\n'
                    f'Frame {frame}/{len(original_positions)-1}, '
                    f'Time: {timestamps[frame]:.3f}s\n'
                    f'Pos Error: {pos_error:.4f}m, Rot Error: {rot_error:.4f}rad')
        
        return gt_trajectory, pred_trajectory, gt_point, pred_point
    
    # Create animation
    frames = len(original_positions)
    interval = max(50, int(1000 * (timestamps[-1] - timestamps[0]) / frames))
    
    anim = FuncAnimation(fig, animate, frames=frames, interval=interval, 
                        blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim

def visualize_representation_conversion(results, episode_name):
    """Visualize the representation conversion process."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    timestamps = results['timestamps']
    original_positions = results['original_positions']
    processed_obs = results['processed_observations']
    
    # Extract processed observation positions and rotations from dict format
    if len(processed_obs) > 0:
        # processed_obs now contains dictionaries, not numpy arrays
        processed_positions = np.array([obs['robot0_eef_pos'][-1] for obs in processed_obs])  # Last timestep
        processed_rot6d = np.array([obs['robot0_eef_rot_axis_angle'][-1] for obs in processed_obs])  # 6D rotation
    else:
        processed_positions = np.zeros_like(original_positions)
        processed_rot6d = np.zeros((len(timestamps), 6))
    
    # Plot 1: Original vs Processed Positions
    axes[0, 0].plot(timestamps, original_positions, linewidth=2, label=['Orig X', 'Orig Y', 'Orig Z'])
    axes[0, 0].plot(timestamps, processed_positions, '--', alpha=0.7, label=['Proc X', 'Proc Y', 'Proc Z'])
    axes[0, 0].set_title('Position: Original vs Processed')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: 6D Rotation Components
    for i in range(6):
        axes[0, 1].plot(timestamps, processed_rot6d[:, i], label=f'R6D_{i}', alpha=0.8)
    axes[0, 1].set_title('6D Rotation Representation')
    axes[0, 1].set_ylabel('6D Rotation Values')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Action Magnitude Over Time
    action_magnitudes = []
    for actions in results['predicted_actions']:
        if len(actions) > 0:
            first_action = actions[0]
            pos_mag = np.linalg.norm(first_action[:3])
            rot_mag = np.linalg.norm(first_action[3:6])
            action_magnitudes.append([pos_mag, rot_mag])
        else:
            action_magnitudes.append([0, 0])
    
    action_magnitudes = np.array(action_magnitudes)
    
    axes[1, 0].plot(timestamps, action_magnitudes[:, 0], 'g-', linewidth=2, label='Position Magnitude')
    axes[1, 0].plot(timestamps, action_magnitudes[:, 1], 'm-', linewidth=2, label='Rotation Magnitude')
    axes[1, 0].set_title('Predicted Action Magnitudes')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Sample Camera Image
    if len(processed_obs) > 10:
        sample_obs = processed_obs[100]
        sample_image = sample_obs['camera0_rgb'][-1]  # Most recent image, shape (C, H, W)
        
        # Convert from TCHW back to HWC for display
        if sample_image.ndim == 3:
            display_image = np.moveaxis(sample_image, 0, -1)  # CHW -> HWC
        else:
            display_image = sample_image
            
        axes[1, 1].imshow(display_image)
        axes[1, 1].set_title(f'Sample Camera Image (Timestep 10)')
        axes[1, 1].axis('off')
    
    plt.suptitle(f'Representation Conversion Analysis - {episode_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def run_policy_simulation(dataset_directory, obs_pose_repr='relative', action_pose_repr='relative', 
                         policy_path=None, obs_horizon=2, action_horizon=8):
    """Run policy simulation on all episodes in directory."""
    
    # Find all episode directories
    episode_dirs = glob.glob(os.path.join(dataset_directory, "episode_*"))
    episode_dirs.sort()
    
    if not episode_dirs:
        print(f"No episode_* directories found in {dataset_directory}")
        return
    
    print(f"Found {len(episode_dirs)} episodes to simulate")
    print(f"Using obs_pose_repr='{obs_pose_repr}', action_pose_repr='{action_pose_repr}'")
    
    # Policy is required
    if policy_path is None:
        raise ValueError("policy_path must be provided. Set the path to your trained policy checkpoint.")
    
    print(f"Loading policy from: {policy_path}")
    try:
        import dill
        import hydra
        from policy_utils.base_workspace import BaseWorkspace
        
        # Load policy checkpoint
        payload = torch.load(open(policy_path, 'rb'), map_location='cpu', pickle_module=dill)
        cfg = payload['cfg']
        
        # Apply the same config overrides as your inference script
        # cfg._target_ = "diffusion_policy.train_diffusion_transformer_timm_workspace.TrainDiffusionTransformerTimmWorkspace"
        # cfg.policy._target_ = "diffusion_policy.diffusion_transformer_timm_policy.DiffusionTransformerTimmPolicy"
        # cfg.policy.obs_encoder._target_ = "policy_utils.transformer_obs_encoder.TransformerObsEncoder"
        # cfg.ema._target_ = "policy_utils.ema_model.EMAModel"

        cfg._target_ = "diffusion_policy.train_diffusion_unet_image_workspace.TrainDiffusionUnetImageWorkspace"
        cfg.policy._target_ = "diffusion_policy.diffusion_unet_timm_policy.DiffusionUnetTimmPolicy"
        cfg.policy.obs_encoder._target_ = "policy_utils.timm_obs_encoder.TimmObsEncoder"
        cfg.ema._target_ = "policy_utils.ema_model.EMAModel"
        
        # Create workspace and load model
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        policy.eval().to(device)
        
        print(f"Policy loaded successfully on device: {device}")
        
    except Exception as e:
        print(f"Error loading policy: {str(e)}")
        raise
    
    # Create simulator
    simulator = PolicyActionSimulator(
        obs_pose_repr=obs_pose_repr,
        action_pose_repr=action_pose_repr,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        policy=policy,
        device=device
    )
    
    # Process only the first episode for testing
    episode_dir = episode_dirs[0]  # Just take the first episode
    episode_name = os.path.basename(episode_dir)
    print(f"\nProcessing single episode: {episode_name}")
    
    try:
        # Run simulation
        results = simulator.process_episode(episode_dir)

        # Save predictions to CSV (rotvec format)
        out_dir = os.path.join(dataset_directory, "predictions")
        os.makedirs(out_dir, exist_ok=True)
        
        # # Save converted 7D actions (original behavior)
        # out_csv = os.path.join(out_dir, f"{episode_name}_pred_pose.csv")
        # save_episode_predictions(results, out_csv_path=out_csv, include_quat=False)

        # # Save quaternions
        # out_csv_q = os.path.join(out_dir, f"{episode_name}_pred_pose_quat.csv")
        # save_episode_predictions(results, out_csv_path=out_csv_q, include_quat=True)
        
        # # Save raw 10D policy output
        # out_csv_raw = os.path.join(out_dir, f"{episode_name}_raw_policy_10d.csv")
        # save_episode_predictions(results, out_csv_path=out_csv_raw, save_raw_policy=True)

        out_csv_raw_full = os.path.join(out_dir, f"{episode_name}_raw_policy_10d_full_horizon.csv")
        save_episode_predictions_full_horizon(results, out_csv_path=out_csv_raw_full, save_raw_policy=True)
        
        print(f"  - Processed {len(results['predicted_actions'])} timesteps")
        print(f"  - Action horizon: {simulator.action_horizon}")
        print(f"  - Observation horizon: {simulator.obs_horizon}")
        print(f"  - Camera images shape: {results['episode_data']['camera_rgb'].shape}")
        print(f"  - Using REAL policy predictions")
        
        # Visualize results
        visualize_action_predictions(results, episode_name)
        visualize_representation_conversion(results, episode_name)
        
        # 3D Animation
        print(f"  - Creating 3D animation...")
        animate_predicted_vs_ground_truth(results, episode_name)
        
    except Exception as e:
        print(f"  - Error processing {episode_name}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nSimulation complete!")

def process_episode_method(self, episode_dir):
    """Process episode method for the simulator."""
    episode_data = self.load_episode_data(episode_dir)

    positions  = episode_data['positions']
    rotations  = episode_data['rotations']
    timestamps = episode_data['timestamps']

    predicted_actions      = []  # absolute 7D per t (first H steps)
    raw_policy_actions     = []  # raw 10D policy output per t (first H steps)
    processed_observations = []
    current_poses          = []

    for t in range(len(positions)):
        # Build obs for this timestep
        obs_dict, current_pose = self.simulate_obs_processing(episode_data, t)

        # Model prediction: sequence of 10D relative poses (policy frame)
        predicted_action_10d = self.simulate_policy_prediction(obs_dict)
        
        # Store raw policy output
        raw_policy_actions.append(predicted_action_10d.copy())

        # Convert to executable absolute 7D using the recent env obs
        env_obs = {
            'robot0_eef_pos': positions[max(0, t-1):t+1],
            'robot0_eef_rot_axis_angle': rotations[max(0, t-1):t+1]
        }
        executable_actions = []
        for a10 in predicted_action_10d:
            # First, convert to 4x4 pose matrix in policy frame
            pose_mat_policy = pose10d_to_mat(a10)

            # Transform rotation and translation into robot frame
            R_policy = pose_mat_policy[:3, :3]
            t_policy = pose_mat_policy[:3, 3]

            R_robot = POLICY_TO_ROBOT_ROT @ R_policy
            t_robot = POLICY_TO_ROBOT_ROT @ t_policy

            pose_mat_robot = np.eye(4)
            pose_mat_robot[:3, :3] = R_robot
            pose_mat_robot[:3, 3] = t_robot

            # Convert back to 7D pose for execution or animation
            real_action = mat_to_pose(pose_mat_robot)  # returns [x,y,z, rx,ry,rz]
            executable_actions.append(real_action)


        predicted_actions.append(executable_actions)
        processed_observations.append(obs_dict)
        current_poses.append(current_pose)

    return {
        'predicted_actions': predicted_actions,             # absolute 7D
        'raw_policy_actions': raw_policy_actions,           # raw 10D policy output
        'processed_observations': processed_observations,
        'current_poses': current_poses,
        'original_positions': positions,
        'original_rotations': rotations,
        'timestamps': timestamps,
        'episode_data': episode_data                        
    }

# Monkey patch the method
PolicyActionSimulator.process_episode = process_episode_method

if __name__ == "__main__":
    # Configuration - updated for numpy array format
    dataset_directory = "/home/hisham246/uwaterloo/umi/reaching_ball_multimodal/dataset/"
    
    # Path to your trained policy checkpoint (REQUIRED)
    policy_path = "/home/hisham246/uwaterloo/diffusion_policy_models/reaching_ball_multimodal.ckpt"
    
    # Policy parameters (should match your trained model)
    obs_horizon = 2  # Number of observation timesteps
    action_horizon = 8  # Number of action prediction timesteps
    
    print("="*60)
    print("POLICY ACTION PREDICTION SIMULATION")
    print("="*60)
    print("This script simulates your diffusion policy pipeline:")
    print("1. Loads observations from numpy arrays")
    print("2. Processes camera images and robot poses") 
    print("3. Applies observation horizon and proper transformations")
    print("4. Runs REAL policy prediction")
    print("5. Converts back to robot actions (7D)")
    print("6. Visualizes the entire pipeline")
    
    print(f"\nUsing policy: {policy_path}")
    
    # Run simulation
    run_policy_simulation(
        dataset_directory=dataset_directory,
        obs_pose_repr='relative',      # Same as your dataset
        action_pose_repr='relative',   # Same as your execution script
        policy_path=policy_path,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )