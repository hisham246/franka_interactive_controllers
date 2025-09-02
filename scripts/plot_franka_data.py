import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def load_franka_data(csv_path):
    """Load Franka robot data from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_ee_poses(df, save_path=None, show_plot=True):
    """Plot EE poses in 6 subplots (3x2): X,Y,Z positions and Rx,Ry,Rz rotations"""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('End-Effector Poses: Desired vs Actual', fontsize=16, fontweight='bold')
    
    time = df['timestamp'] - df['timestamp'].iloc[0]
    
    # Position plots
    positions = ['x', 'y', 'z']
    for i, pos in enumerate(positions):
        ax = axes[i, 0]
        ax.plot(time, df[f'filt_ee_{pos}'], 'b-', label=f'Desired {pos.upper()}', linewidth=2)
        ax.plot(time, df[f'act_ee_{pos}'], 'r--', label=f'Actual {pos.upper()}', linewidth=2)
        ax.set_title(f'EE Position {pos.upper()}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Rotation plots
    rotations = ['rx', 'ry', 'rz']
    for i, rot in enumerate(rotations):
        ax = axes[i, 1]
        ax.plot(time, df[f'filt_ee_{rot}'], 'b-', label=f'Desired {rot.upper()}', linewidth=2)
        ax.plot(time, df[f'act_ee_{rot}'], 'r--', label=f'Actual {rot.upper()}', linewidth=2)
        ax.set_title(f'EE Rotation {rot.upper()}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Rotation (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        if save_path.endswith('.png'):
            pose_path = save_path.replace('.png', '_ee_poses.png')
        else:
            pose_path = f"{save_path}_ee_poses.png"
        plt.savefig(pose_path, dpi=300, bbox_inches='tight')
        print(f"EE poses plot saved to: {pose_path}")
    
    if show_plot:
        plt.show()
    
    return fig

def plot_ee_velocities(df, save_path=None, show_plot=True):
    """Plot EE velocities in 6 subplots (3x2): Vx,Vy,Vz linear and Wx,Wy,Wz angular"""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('End-Effector Velocities: Desired vs Actual', fontsize=16, fontweight='bold')
    
    time = df['timestamp'] - df['timestamp'].iloc[0]
    
    # Linear velocity plots
    linear_vels = ['vx', 'vy', 'vz']
    for i, vel in enumerate(linear_vels):
        ax = axes[i, 0]
        ax.plot(time, df[f'des_ee_{vel}'], 'b-', label=f'Desired {vel.upper()}', linewidth=2)
        ax.plot(time, df[f'act_ee_{vel}'], 'r--', label=f'Actual {vel.upper()}', linewidth=2)
        ax.set_title(f'EE Linear Velocity {vel.upper()}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Linear Velocity (m/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Angular velocity plots
    angular_vels = ['wx', 'wy', 'wz']
    for i, vel in enumerate(angular_vels):
        ax = axes[i, 1]
        ax.plot(time, df[f'des_ee_{vel}'], 'b-', label=f'Desired {vel.upper()}', linewidth=2)
        ax.plot(time, df[f'act_ee_{vel}'], 'r--', label=f'Actual {vel.upper()}', linewidth=2)
        ax.set_title(f'EE Angular Velocity {vel.upper()}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        if save_path.endswith('.png'):
            vel_path = save_path.replace('.png', '_ee_velocities.png')
        else:
            vel_path = f"{save_path}_ee_velocities.png"
        plt.savefig(vel_path, dpi=300, bbox_inches='tight')
        print(f"EE velocities plot saved to: {vel_path}")
    
    if show_plot:
        plt.show()
    
    return fig

def plot_joint_positions(df, save_path=None, show_plot=True):
    """Plot joint positions in 7 subplots arranged in 3x3 grid (last 2 empty)"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle('Joint Positions: Desired vs Actual', fontsize=16, fontweight='bold')
    
    time = df['timestamp'] - df['timestamp'].iloc[0]
    
    for i in range(7):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        ax.plot(time, df[f'des_q{i+1}'], 'b-', label=f'Desired J{i+1}', linewidth=2)
        ax.plot(time, df[f'act_q{i+1}'], 'r--', label=f'Actual J{i+1}', linewidth=2)
        ax.set_title(f'Joint {i+1} Position')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    axes[2, 1].set_visible(False)
    axes[2, 2].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        if save_path.endswith('.png'):
            pos_path = save_path.replace('.png', '_joint_positions.png')
        else:
            pos_path = f"{save_path}_joint_positions.png"
        plt.savefig(pos_path, dpi=300, bbox_inches='tight')
        print(f"Joint positions plot saved to: {pos_path}")
    
    if show_plot:
        plt.show()
    
    return fig

def plot_joint_velocities(df, save_path=None, show_plot=True):
    """Plot joint velocities in 7 subplots arranged in 3x3 grid (last 2 empty)"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle('Joint Velocities: Desired vs Actual', fontsize=16, fontweight='bold')
    
    time = df['timestamp'] - df['timestamp'].iloc[0]
    
    for i in range(7):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        ax.plot(time, df[f'des_dq{i+1}'], 'b-', label=f'Desired J{i+1}', linewidth=2)
        ax.plot(time, df[f'act_dq{i+1}'], 'r--', label=f'Actual J{i+1}', linewidth=2)
        ax.set_title(f'Joint {i+1} Velocity')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (rad/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    axes[2, 1].set_visible(False)
    axes[2, 2].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        if save_path.endswith('.png'):
            vel_path = save_path.replace('.png', '_joint_velocities.png')
        else:
            vel_path = f"{save_path}_joint_velocities.png"
        plt.savefig(vel_path, dpi=300, bbox_inches='tight')
        print(f"Joint velocities plot saved to: {vel_path}")
    
    if show_plot:
        plt.show()
    
    return fig

def plot_joint_torques(df, save_path=None, show_plot=True):
    """Plot joint torques in 7 subplots arranged in 3x3 grid (last 2 empty)"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle('Joint Torques: Desired vs Actual', fontsize=16, fontweight='bold')
    
    time = df['timestamp'] - df['timestamp'].iloc[0]
    
    for i in range(7):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        ax.plot(time, df[f'des_tau{i+1}'], 'b-', label=f'Desired J{i+1}', linewidth=2)
        ax.plot(time, df[f'act_tau{i+1}'], 'r--', label=f'Actual J{i+1}', linewidth=2)
        ax.set_title(f'Joint {i+1} Torque')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Torque (Nm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    axes[2, 1].set_visible(False)
    axes[2, 2].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        if save_path.endswith('.png'):
            torque_path = save_path.replace('.png', '_joint_torques.png')
        else:
            torque_path = f"{save_path}_joint_torques.png"
        plt.savefig(torque_path, dpi=300, bbox_inches='tight')
        print(f"Joint torques plot saved to: {torque_path}")
    
    if show_plot:
        plt.show()
    
    return fig

def plot_all_data(df, save_path=None, show_plot=True):
    """Plot all data in 5 separate figures"""
    figures = []
    
    print("Plotting EE poses...")
    fig1 = plot_ee_poses(df, save_path, show_plot=False)  # Don't show individual plots
    figures.append(fig1)
    
    print("Plotting EE velocities...")
    fig2 = plot_ee_velocities(df, save_path, show_plot=False)
    figures.append(fig2)
    
    print("Plotting joint positions...")
    fig3 = plot_joint_positions(df, save_path, show_plot=False)
    figures.append(fig3)
    
    print("Plotting joint velocities...")
    fig4 = plot_joint_velocities(df, save_path, show_plot=False)
    figures.append(fig4)
    
    print("Plotting joint torques...")
    fig5 = plot_joint_torques(df, save_path, show_plot=False)
    figures.append(fig5)
    
    # Show all figures at once if requested
    if show_plot:
        plt.show()
    
    return figures

def print_data_summary(df):
    """Print summary statistics of the data"""
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    
    time = df['timestamp'] - df['timestamp'].iloc[0]
    print(f"Duration: {time.iloc[-1]:.2f} seconds")
    print(f"Data points: {len(df)}")
    print(f"Sampling rate: {len(df)/time.iloc[-1]:.1f} Hz")
    
    print("\nEE Position Range:")
    for axis in ['x', 'y', 'z']:
        des_range = df[f'des_ee_{axis}'].max() - df[f'des_ee_{axis}'].min()
        act_range = df[f'act_ee_{axis}'].max() - df[f'act_ee_{axis}'].min()
        print(f"  {axis.upper()}: Desired {des_range:.4f}m, Actual {act_range:.4f}m")
    
    print("\nEE Position RMS Error:")
    for axis in ['x', 'y', 'z']:
        error = df[f'des_ee_{axis}'] - df[f'act_ee_{axis}']
        rms_error = np.sqrt(np.mean(error**2))
        print(f"  {axis.upper()}: {rms_error:.6f}m")
    
    print("\nJoint Position RMS Error:")
    for i in range(1, 8):
        error = df[f'des_q{i}'] - df[f'act_q{i}']
        rms_error = np.sqrt(np.mean(error**2))
        print(f"  Joint {i}: {rms_error:.6f} rad")

def main():
    parser = argparse.ArgumentParser(description='Plot Franka robot data in 5 separate figures')
    parser.add_argument('csv_file', help='Path to CSV data file')
    parser.add_argument('--save', help='Base name for saved plots (will add suffixes)')
    parser.add_argument('--no-show', action='store_true', help='Don\'t display plots')
    parser.add_argument('--summary', action='store_true', help='Print data summary')
    
    args = parser.parse_args()
    
    # Load data
    df = load_franka_data(args.csv_file)
    if df is None:
        return
    
    print(f"Loaded {len(df)} data points from {args.csv_file}")
    
    # Print summary if requested
    if args.summary:
        print_data_summary(df)
    
    # Plot all data
    figures = plot_all_data(df, save_path=args.save, show_plot=not args.no_show)
    
    print(f"\nGenerated {len(figures)} figures")
    if args.save:
        print(f"All plots saved with base name: {args.save}")

if __name__ == "__main__":
    main()