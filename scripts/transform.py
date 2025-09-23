import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


csv_path = "/home/hisham246/uwaterloo/umi/reaching_ball_multimodal/csv/episode_1.csv"
df = pd.read_csv(csv_path)

# Define your mapping matrices
def _Rz(theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)

_M    = _Rz(np.deg2rad(-90.0))  # robot → policy
_Minv = _Rz(np.deg2rad(90.0))   # policy → robot

def _map_pose_array(arr, M):
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

# Extract position and axis-angle rotation from CSV
positions = df[['robot0_eef_pos_0', 'robot0_eef_pos_1', 'robot0_eef_pos_2']].values
rotations = df[['robot0_eef_rot_axis_angle_0', 'robot0_eef_rot_axis_angle_1', 'robot0_eef_rot_axis_angle_2']].values
poses_policy = np.concatenate([positions, rotations], axis=1)

# Convert all poses to robot frame
poses_robot = _map_pose_array(poses_policy, _Minv)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# --- Policy Frame ---
axs[0].plot(poses_policy[:, 0], poses_policy[:, 1], 'o-b')
axs[0].quiver(
    poses_policy[:-1, 0], poses_policy[:-1, 1],
    poses_policy[1:, 0] - poses_policy[:-1, 0],
    poses_policy[1:, 1] - poses_policy[:-1, 1],
    scale_units='xy', angles='xy', scale=1, width=0.004, color='blue'
)
axs[0].set_title("Policy Frame")
axs[0].set_xlabel("x (right)")
axs[0].set_ylabel("y (forward)")
axs[0].axis('equal')
axs[0].grid(True)

# --- Robot Frame ---
axs[1].plot(poses_robot[:, 0], poses_robot[:, 1], 'o-r')
axs[1].quiver(
    poses_robot[:-1, 0], poses_robot[:-1, 1],
    poses_robot[1:, 0] - poses_robot[:-1, 0],
    poses_robot[1:, 1] - poses_robot[:-1, 1],
    scale_units='xy', angles='xy', scale=1, width=0.004, color='red'
)
axs[1].set_title("Robot Frame")
axs[1].set_xlabel("x (forward)")
axs[1].set_ylabel("y (left)")
axs[1].axis('equal')
axs[1].grid(True)

plt.suptitle("Absolute EE Trajectory: Policy Frame vs Robot Frame")
plt.tight_layout()
plt.show()
