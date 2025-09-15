import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial.transform as st

def load_robot_data(csv_file):
    """Load actual EE trajectory data from CSV file."""
    df = pd.read_csv(csv_file)

    # positions = df[['filt_ee_x', 'filt_ee_y', 'filt_ee_z']].values
    # rotations = df[['filt_ee_rx', 'filt_ee_ry', 'filt_ee_rz']].values
    positions = df[['ee_pos_0', 'ee_pos_1', 'ee_pos_2']].values
    rotations = df[['ee_rot_0', 'ee_rot_1', 'ee_rot_2']].values
    timestamps = df['timestamp'].values

    # normalize timestamps to start from 0
    timestamps = timestamps - timestamps[0]
    
    return positions, rotations, timestamps

def create_coordinate_frame(position, rotation_vec, scale=0.05):
    """Create coordinate frame vectors for visualization."""
    rot = st.Rotation.from_rotvec(rotation_vec)
    R = rot.as_matrix()
    
    # Unit vectors scaled
    axes = np.eye(3) * scale
    transformed = R @ axes
    
    return transformed[:,0], transformed[:,1], transformed[:,2]


def plot_positions_and_rotations(csv_file, skip=1, start=0):
    """Plot actual EE positions (x,y,z) and rotations (rx,ry,rz) vs time."""
    positions, rotations, timestamps = load_robot_data(csv_file)

    # positions, rotations, timestamps = positions[10000:], rotations[10000:], timestamps[10000:]


    # cut start if needed
    positions, rotations, timestamps = positions[start:], rotations[start:], timestamps[start:]

    # stride if needed
    positions, rotations, timestamps = positions[::skip], rotations[::skip], timestamps[::skip]
    timestamps = timestamps - timestamps[0]
    print("Total time: ", timestamps[-1] - timestamps[0], "s")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # --- positions ---
    labels_pos = ['X', 'Y', 'Z']
    for i in range(3):
        ax = axes[0, i]
        ax.plot(timestamps, positions[:, i], lw=2)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{labels_pos[i]} [m]")
        ax.set_title(f"EE Position {labels_pos[i]}")
        ax.grid(True)

    # --- rotations ---
    labels_rot = ['Rx', 'Ry', 'Rz']
    for i in range(3):
        ax = axes[1, i]
        ax.plot(timestamps, rotations[:, i], lw=2)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{labels_rot[i]} [rad]")
        ax.set_title(f"EE Rotation {labels_rot[i]}")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def animate_robot_frame(csv_file, save_animation=False, filename='ee_animation.gif',
                        skip=1, interval=1):
    """
    skip = stride in frames (e.g. 10 = show every 10th frame, ~10x faster)
    interval = ms between frames (1 = as fast as matplotlib can go)
    """
    positions, rotations, timestamps = load_robot_data(csv_file)

    # Cut beginning if needed
    # positions, rotations, timestamps = positions[15000:], rotations[15000:], timestamps[15000:]

    # Skip frames for speed
    positions, rotations, timestamps = positions[::skip], rotations[::skip], timestamps[::skip]
    timestamps = timestamps - timestamps[0]  # re-normalize

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # bounds
    pos_min, pos_max = positions.min(0)-0.05, positions.max(0)+0.05
    max_range = (pos_max - pos_min).max()/2.0
    mid = (pos_max + pos_min)/2.0
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    traj_line, = ax.plot([], [], [], 'b-', lw=2, label='Trajectory')
    current_point, = ax.plot([], [], [], 'ro', label='EE Position')
    coordinate_arrows = []

    def animate(i):
        for art in coordinate_arrows:
            art.remove()
        coordinate_arrows.clear()

        traj_line.set_data(positions[:i+1,0], positions[:i+1,1])
        traj_line.set_3d_properties(positions[:i+1,2])

        pos = positions[i]
        current_point.set_data([pos[0]], [pos[1]])
        current_point.set_3d_properties([pos[2]])

        xvec,yvec,zvec = create_coordinate_frame(pos, rotations[i])
        ax.quiver(pos[0], pos[1], pos[2], *xvec, color='r', linewidth=2)
        ax.quiver(pos[0], pos[1], pos[2], *yvec, color='g', linewidth=2)
        ax.quiver(pos[0], pos[1], pos[2], *zvec, color='b', linewidth=2)

        coordinate_arrows.extend(ax.collections[-3:])
        ax.set_title(f"Frame {i}/{len(positions)-1}, Time={timestamps[i]:.2f}s")
        return traj_line, current_point

    anim = FuncAnimation(fig, animate, frames=len(positions),
                         interval=interval, blit=False, repeat=True)

    if save_animation:
        anim.save(filename, writer='pillow', fps=500)
        print(f"Saved animation as {filename}")

    plt.legend()
    plt.show()

# Example usage:
# csv_file = "/home/hisham246/uwaterloo/surface_wiping_test_3/policy_actions_20250911_131839.csv"
csv_file = "/home/hisham246/uwaterloo/surface_wiping_test_2/policy_actions_20250909_204904.csv"
plot_positions_and_rotations(csv_file, skip=1, start=0)
animate_robot_frame(csv_file, save_animation=False)