import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def load_robot_data(csv_file):
    """Load robot trajectory data from CSV file."""
    df = pd.read_csv(csv_file)
    
    # Extract position and rotation data
    positions = df[['robot0_eef_pos_0', 'robot0_eef_pos_1', 'robot0_eef_pos_2']].values
    rotations = df[['robot0_eef_rot_axis_angle_0', 'robot0_eef_rot_axis_angle_1', 'robot0_eef_rot_axis_angle_2']].values
    timestamps = df['timestamp'].values
    
    return positions, rotations, timestamps

def plot_all_orientations(csv_directory):
    """Plot all 3 orientation components from all demos in 3 separate subplots."""
    
    # Find all episode CSV files
    csv_files = glob.glob(os.path.join(csv_directory, "episode_*.csv"))
    csv_files.sort()  # Sort to process in order
    
    if not csv_files:
        print(f"No episode_*.csv files found in {csv_directory}")
        return
        
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Color palette for different episodes
    colors = plt.cm.tab10(np.linspace(0, 1, len(csv_files)))
    
    # Labels for the rotation axes
    rotation_labels = ['Rx (Roll)', 'Ry (Pitch)', 'Rz (Yaw)']
    
    episode_names = []
    
    # Process each episode
    for i, csv_file in enumerate(csv_files):
        episode_name = os.path.basename(csv_file).replace('.csv', '')
        episode_names.append(episode_name)
        
        try:
            # Load data
            positions, rotations, timestamps = load_robot_data(csv_file)
            
            # Normalize timestamps to start from 0 for better comparison
            timestamps_normalized = timestamps - timestamps[0]
            
            # Plot each rotation component in its respective subplot
            for axis_idx in range(3):
                axes[axis_idx].plot(timestamps_normalized, rotations[:, axis_idx], 
                                  color=colors[i], linewidth=2, alpha=0.8,
                                  label=episode_name)
                
                axes[axis_idx].set_ylabel(f'{rotation_labels[axis_idx]} (rad)')
                axes[axis_idx].grid(True, alpha=0.3)
                axes[axis_idx].set_title(f'{rotation_labels[axis_idx]} - All Episodes')
                
        except Exception as e:
            print(f"Error processing {episode_name}: {str(e)}")
            continue
    
    # Set xlabel only for bottom subplot
    axes[2].set_xlabel('Time (s)')
    
    plt.suptitle(f'Robot End-Effector Orientation Comparison\n{len(csv_files)} Episodes', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    
    for i, csv_file in enumerate(csv_files):
        episode_name = os.path.basename(csv_file).replace('.csv', '')
        try:
            positions, rotations, timestamps = load_robot_data(csv_file)
            
            print(f"\n{episode_name}:")
            print(f"  Duration: {timestamps[-1] - timestamps[0]:.2f}s")
            print(f"  Rx range: [{rotations[:, 0].min():.3f}, {rotations[:, 0].max():.3f}] rad")
            print(f"  Ry range: [{rotations[:, 1].min():.3f}, {rotations[:, 1].max():.3f}] rad") 
            print(f"  Rz range: [{rotations[:, 2].min():.3f}, {rotations[:, 2].max():.3f}] rad")
            print(f"  Total rotation magnitude: {np.linalg.norm(rotations, axis=1).max():.3f} rad")
            
        except Exception as e:
            print(f"  Error: {str(e)}")


# Main execution
if __name__ == "__main__":
    # Directory containing your CSV files
    csv_directory = "/home/hisham246/uwaterloo/umi/reaching_ball_multimodal/csv/"
    
    # Plot all orientations in 3 subplots
    print("\n1. Creating orientation comparison plots...")
    plot_all_orientations(csv_directory)