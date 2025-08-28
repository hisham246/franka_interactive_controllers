#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
import argparse

def plot_positions(csv_path):
    df = pd.read_csv(csv_path)

    if not all(col in df.columns for col in ["x","y","z"]):
        raise ValueError("CSV must contain x,y,z columns")

    # 3D trajectory plot
    fig = plt.figure(figsize=(12, 5))

    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax3d.plot(df["x"], df["y"], df["z"], label="Trajectory", linewidth=2)
    ax3d.scatter(df["x"].iloc[0], df["y"].iloc[0], df["z"].iloc[0], c='g', marker='o', s=60, label="Start")
    ax3d.scatter(df["x"].iloc[-1], df["y"].iloc[-1], df["z"].iloc[-1], c='r', marker='x', s=80, label="End")
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title("3D End-Effector Trajectory")
    ax3d.legend()

    # Time series plot
    ax2d = fig.add_subplot(1, 2, 2)
    if "time" in df.columns:
        t = df["time"] - df["time"].iloc[0]
        ax2d.plot(t, df["x"], label="x", linewidth=2)
        ax2d.plot(t, df["y"], label="y", linewidth=2)
        ax2d.plot(t, df["z"], label="z", linewidth=2)
        ax2d.set_xlabel("Time [s]")
    else:
        ax2d.plot(df.index, df["x"], label="x", linewidth=2)
        ax2d.plot(df.index, df["y"], label="y", linewidth=2)
        ax2d.plot(df.index, df["z"], label="z", linewidth=2)
        ax2d.set_xlabel("Timestep")

    ax2d.set_ylabel("Position [m]")
    ax2d.set_title("End-Effector Positions vs Time")
    ax2d.grid(True, alpha=0.3)
    ax2d.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_positions("/home/hisham246/uwaterloo/umi/surface_wiping_tp/dataset/predictions/episode_117_pred_pose.csv")
