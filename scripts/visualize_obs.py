import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import os
import glob


def load_camera_observations(npy_path):
    """Load camera observations from .npy file."""
    images = np.load(npy_path)
    print(f"Loaded camera observations with shape: {images.shape}")
    print(f"Data type: {images.dtype}")
    print(f"Value range: [{images.min()}, {images.max()}]")
    return images


def display_static_grid(images, num_images=12, save_path=None):
    """Display a static grid of sample frames."""
    n_images = min(num_images, len(images))
    
    # Calculate grid dimensions
    cols = 4
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    axes = axes.flatten() if n_images > 1 else [axes]
    
    # Sample evenly spaced frames
    indices = np.linspace(0, len(images)-1, n_images, dtype=int)
    
    for idx, (i, ax) in enumerate(zip(indices, axes)):
        img = images[i]
        
        # Handle different data types
        if img.dtype == np.float32 or img.dtype == np.float64:
            # Assume normalized [0, 1]
            display_img = np.clip(img, 0, 1)
        elif img.dtype == np.uint8:
            display_img = img / 255.0
        else:
            display_img = img
            
        ax.imshow(display_img)
        ax.set_title(f'Frame {i}/{len(images)-1}')
        ax.axis('off')
    
    # Hide unused subplots
    for ax in axes[n_images:]:
        ax.axis('off')
    
    plt.suptitle(f'Camera Observations - Sample Frames', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved static grid to: {save_path}")
    
    plt.show()


def animate_camera_observations(images, fps=10, save_path=None):
    """Create an animated visualization of camera observations."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Initialize with first frame
    img = images[0]
    if img.dtype == np.float32 or img.dtype == np.float64:
        display_img = np.clip(img, 0, 1)
    elif img.dtype == np.uint8:
        display_img = img / 255.0
    else:
        display_img = img
    
    im = ax.imshow(display_img)
    ax.axis('off')
    title = ax.set_title(f'Frame 0/{len(images)-1}', fontsize=14, fontweight='bold')
    
    def animate(frame):
        """Update function for animation."""
        img = images[frame]
        
        # Handle different data types
        if img.dtype == np.float32 or img.dtype == np.float64:
            display_img = np.clip(img, 0, 1)
        elif img.dtype == np.uint8:
            display_img = img / 255.0
        else:
            display_img = img
        
        im.set_data(display_img)
        title.set_text(f'Frame {frame}/{len(images)-1}')
        return [im, title]
    
    # Create animation
    interval = 1000 / fps  # milliseconds per frame
    anim = FuncAnimation(fig, animate, frames=len(images), 
                        interval=interval, blit=True, repeat=True)
    
    # Save if requested
    if save_path:
        print(f"Saving animation to: {save_path}")
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Animation saved!")
    
    plt.tight_layout()
    plt.show()
    
    return anim


def create_side_by_side_animation(images1, images2, labels=None, fps=10, save_path=None):
    """
    Create side-by-side animation of two image sequences.
    Useful for comparing camera views or predicted vs actual observations.
    """
    if labels is None:
        labels = ['Sequence 1', 'Sequence 2']
    
    # Match lengths
    min_len = min(len(images1), len(images2))
    images1 = images1[:min_len]
    images2 = images2[:min_len]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Initialize with first frames
    def prepare_img(img):
        if img.dtype == np.float32 or img.dtype == np.float64:
            return np.clip(img, 0, 1)
        elif img.dtype == np.uint8:
            return img / 255.0
        return img
    
    im1 = ax1.imshow(prepare_img(images1[0]))
    im2 = ax2.imshow(prepare_img(images2[0]))
    
    ax1.axis('off')
    ax2.axis('off')
    ax1.set_title(labels[0], fontsize=14, fontweight='bold')
    ax2.set_title(labels[1], fontsize=14, fontweight='bold')
    
    title = fig.suptitle(f'Frame 0/{min_len-1}', fontsize=16, fontweight='bold')
    
    def animate(frame):
        im1.set_data(prepare_img(images1[frame]))
        im2.set_data(prepare_img(images2[frame]))
        title.set_text(f'Frame {frame}/{min_len-1}')
        return [im1, im2, title]
    
    interval = 1000 / fps
    anim = FuncAnimation(fig, animate, frames=min_len, 
                        interval=interval, blit=True, repeat=True)
    
    if save_path:
        print(f"Saving side-by-side animation to: {save_path}")
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Animation saved!")
    
    plt.tight_layout()
    plt.show()
    
    return anim


def export_to_video(images, output_path, fps=10):
    """Export image sequence to MP4 video using OpenCV."""
    if len(images) == 0:
        print("No images to export!")
        return
    
    # Get dimensions
    height, width = images[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for img in images:
        # Handle different data types
        if img.dtype == np.float32 or img.dtype == np.float64:
            frame = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        elif img.dtype == np.uint8:
            frame = img
        else:
            frame = img.astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Video saved to: {output_path}")


def analyze_observations(images):
    """Print statistical analysis of the observations."""
    print("\n" + "="*60)
    print("OBSERVATION ANALYSIS")
    print("="*60)
    print(f"Total frames: {len(images)}")
    print(f"Image shape: {images.shape[1:]}")
    print(f"Data type: {images.dtype}")
    print(f"Value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Mean value: {images.mean():.3f}")
    print(f"Std deviation: {images.std():.3f}")
    
    # Check for motion (frame differences)
    if len(images) > 1:
        diffs = np.abs(np.diff(images.astype(float), axis=0))
        avg_motion = diffs.mean()
        max_motion = diffs.max()
        print(f"\nMotion Analysis:")
        print(f"Average frame difference: {avg_motion:.3f}")
        print(f"Maximum frame difference: {max_motion:.3f}")
    
    print("="*60 + "\n")


def main():
    """Main function to visualize saved camera observations."""
    
    # Path to your saved .npy file
    npy_path = "/home/hisham246/uwaterloo/reaching_ball_multimodal_3/camera_observations_20250929_175538.npy"
    
    # Load images
    images = load_camera_observations(npy_path)
    
    # Analyze the observations
    analyze_observations(images)
    
    # Create output directory for visualizations
    output_dir = os.path.dirname(npy_path)
    base_name = os.path.splitext(os.path.basename(npy_path))[0]
    
    # 1. Display static grid of sample frames
    print("Creating static grid visualization...")
    grid_path = os.path.join(output_dir, f"{base_name}_grid.png")
    display_static_grid(images, num_images=12, save_path=grid_path)
    
    # 2. Create animated visualization
    print("\nCreating animation...")
    gif_path = os.path.join(output_dir, f"{base_name}_animation.gif")
    anim = animate_camera_observations(images, fps=10, save_path=gif_path)
    
    # 3. Export to video (optional, requires opencv)
    print("\nExporting to video...")
    video_path = os.path.join(output_dir, f"{base_name}.mp4")
    try:
        export_to_video(images, video_path, fps=10)
    except Exception as e:
        print(f"Could not export video: {e}")
    
    print("\nVisualization complete!")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
