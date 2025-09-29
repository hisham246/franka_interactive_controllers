import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial.transform as st
from mpl_toolkits.mplot3d import Axes3D

def load_robot_data(csv_file):
    """Load robot trajectory data from CSV file."""
    df = pd.read_csv(csv_file)
    
    # Extract position and rotation data
    positions = df[['robot0_eef_pos_0', 'robot0_eef_pos_1', 'robot0_eef_pos_2']].values
    rotations = df[['robot0_eef_rot_axis_angle_0', 'robot0_eef_rot_axis_angle_1', 'robot0_eef_rot_axis_angle_2']].values
    timestamps = df['timestamp'].values
    
    return positions, rotations, timestamps

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

def animate_robot_frame(csv_file, save_animation=False, filename='robot_animation.gif'):
    """Create 3D animation of robot end-effector movement."""
    
    # Load data
    positions, rotations, timestamps = load_robot_data(csv_file)
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate bounds for better visualization
    pos_min = positions.min(axis=0) - 0.1
    pos_max = positions.max(axis=0) + 0.1
    
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
    trajectory_line, = ax.plot([], [], [], 'b-', alpha=0.6, linewidth=2, label='Trajectory')
    current_point, = ax.plot([], [], [], 'ro', markersize=8, label='Current Position')
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('Robot End-Effector 3D Movement Animation')
    
    # Store reference to coordinate frame arrows for updating
    coordinate_arrows = []
    
    # Animation function
    def animate(frame):
        # Clear previous coordinate frame arrows
        for arrow in coordinate_arrows:
            arrow.remove()
        coordinate_arrows.clear()
        
        # Show trajectory up to current frame
        if frame > 0:
            trajectory_line.set_data_3d(positions[:frame+1, 0], 
                                      positions[:frame+1, 1], 
                                      positions[:frame+1, 2])
        
        # Current position
        current_pos = positions[frame]
        current_point.set_data_3d([current_pos[0]], [current_pos[1]], [current_pos[2]])
        
        # Create coordinate frame at current position
        x_vec, y_vec, z_vec = create_coordinate_frame(current_pos, rotations[frame])
        
        # Draw new coordinate frame and store references
        x_arrow = ax.quiver(current_pos[0], current_pos[1], current_pos[2],
                           x_vec[0], x_vec[1], x_vec[2], 
                           color='red', arrow_length_ratio=0.2, linewidth=3)
        y_arrow = ax.quiver(current_pos[0], current_pos[1], current_pos[2],
                           y_vec[0], y_vec[1], y_vec[2], 
                           color='green', arrow_length_ratio=0.2, linewidth=3)
        z_arrow = ax.quiver(current_pos[0], current_pos[1], current_pos[2],
                           z_vec[0], z_vec[1], z_vec[2], 
                           color='blue', arrow_length_ratio=0.2, linewidth=3)
        
        coordinate_arrows.extend([x_arrow, y_arrow, z_arrow])
        
        # Update title with current info
        ax.set_title(f'Robot End-Effector Movement (Frame {frame}/{len(positions)-1})\n'
                    f'Time: {timestamps[frame]:.3f}s')
        
        return trajectory_line, current_point
    
    # Create animation
    frames = len(positions)
    interval = max(50, int(1000 * (timestamps[-1] - timestamps[0]) / frames))  # Adaptive timing
    
    anim = FuncAnimation(fig, animate, frames=frames, interval=interval, 
                        blit=False, repeat=True)
    
    if save_animation:
        print(f"Saving animation to {filename}...")
        anim.save(filename, writer='pillow', fps=20)
        print("Animation saved!")
    
    plt.tight_layout()
    plt.show()
    
    return anim

def plot_trajectory_summary(csv_file):
    """Create static summary plots of the robot trajectory."""
    positions, rotations, timestamps = load_robot_data(csv_file)
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               c='green', s=100, label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               c='red', s=100, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # Position vs time
    ax2 = fig.add_subplot(222)
    ax2.plot(timestamps, positions[:, 0], label='X', linewidth=2)
    ax2.plot(timestamps, positions[:, 1], label='Y', linewidth=2)
    ax2.plot(timestamps, positions[:, 2], label='Z', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs Time')
    ax2.legend()
    ax2.grid(True)
    
    # Rotation magnitude vs time
    ax3 = fig.add_subplot(223)
    rotation_magnitudes = np.linalg.norm(rotations, axis=1)
    ax3.plot(timestamps, rotation_magnitudes, 'purple', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Rotation Magnitude (rad)')
    ax3.set_title('Rotation Magnitude vs Time')
    ax3.grid(True)
    
    # Individual rotation components
    ax4 = fig.add_subplot(224)
    ax4.plot(timestamps, rotations[:, 0], label='Rx', linewidth=2)
    ax4.plot(timestamps, rotations[:, 1], label='Ry', linewidth=2)
    ax4.plot(timestamps, rotations[:, 2], label='Rz', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Rotation (rad)')
    ax4.set_title('Rotation Components vs Time')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    csv_file = "/home/hisham246/uwaterloo/umi/reaching_ball_multimodal/csv/episode_1.csv"
    
    print("Creating trajectory summary plots...")
    plot_trajectory_summary(csv_file)
    
    print("\nStarting 3D animation...")
    print("Close the plot window to continue...")
    
    # Create animation (set save_animation=True to save as GIF)
    anim = animate_robot_frame(csv_file, save_animation=False)
    
    # Keep the animation running
    plt.show()
    
    print("Animation complete!")

# Additional utility function for analysis
def analyze_motion_characteristics(csv_file):
    """Analyze motion characteristics of the robot trajectory."""
    positions, rotations, timestamps = load_robot_data(csv_file)
    
    # Calculate velocities
    dt = np.diff(timestamps)
    position_deltas = np.diff(positions, axis=0)
    velocities = position_deltas / dt[:, np.newaxis]
    
    # Calculate angular velocities
    rotation_deltas = np.diff(rotations, axis=0)
    angular_velocities = rotation_deltas / dt[:, np.newaxis]
    
    print("=== Motion Analysis ===")
    print(f"Total trajectory time: {timestamps[-1] - timestamps[0]:.3f} seconds")
    print(f"Total distance traveled: {np.sum(np.linalg.norm(position_deltas, axis=1)):.4f} meters")
    print(f"Average speed: {np.mean(np.linalg.norm(velocities, axis=1)):.4f} m/s")
    print(f"Max speed: {np.max(np.linalg.norm(velocities, axis=1)):.4f} m/s")
    print(f"Average angular velocity: {np.mean(np.linalg.norm(angular_velocities, axis=1)):.4f} rad/s")
    print(f"Max angular velocity: {np.max(np.linalg.norm(angular_velocities, axis=1)):.4f} rad/s")
    
    # Position bounds
    print(f"\nPosition bounds:")
    print(f"X: [{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}] meters")
    print(f"Y: [{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}] meters") 
    print(f"Z: [{positions[:, 2].min():.4f}, {positions[:, 2].max():.4f}] meters")



