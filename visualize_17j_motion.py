import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys

# --- IMPORTS ---
try:
    from utils.data_utils import parse_trc_robust
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from utils.data_utils import parse_trc_robust

# --- SKELETON DEFINITION (17 Joints - Human3.6M format) ---
# MotionBERT outputs exactly 17 joints in this order.
SKELETON_EDGES = [
    [0, 1], [1, 2], [2, 3],       # Right Leg
    [0, 4], [4, 5], [5, 6],       # Left Leg
    [0, 7], [7, 8], [8, 9], [9, 10], # Spine & Head
    [8, 11], [11, 12], [12, 13],  # Left Arm
    [8, 14], [14, 15], [15, 16]   # Right Arm
]

def update_plot(frame, data, lines, title_text):
    """Update function for the animation."""
    idx = frame % data.shape[0]
    current_pose = data[idx] 
    
    # Check if the skeleton is completely blank (zeroed out by our strict detector)
    if np.sum(np.abs(current_pose)) < 1e-4:
        for i in range(len(SKELETON_EDGES)):
            lines[i].set_data([np.nan, np.nan], [np.nan, np.nan])
            lines[i].set_3d_properties([np.nan, np.nan])
        title_text.set_text(f"MotionBERT 17-Joint Skeleton | Frame {idx} (NO DETECTION)")
        return lines + [title_text]
        
    for i, (start_idx, end_idx) in enumerate(SKELETON_EDGES):
        if start_idx < len(current_pose) and end_idx < len(current_pose):
            p1 = current_pose[start_idx]
            p2 = current_pose[end_idx]
            
            # TRC Mapping: X->0, Z->1 (Up/Height), Y->2 (Depth) 
            x = [p1[0], p2[0]]
            y = [p1[2], p2[2]] 
            z = [p1[1], p2[1]] 
            
            lines[i].set_data(x, y)
            lines[i].set_3d_properties(z)

    title_text.set_text(f"MotionBERT 17-Joint Skeleton | Frame {idx}")
    return lines + [title_text]

def main():
    print("="*50)
    print("  MotionBERT 17-Joint 3D Visualizer")
    print("="*50)

    trc_file = ""
    if len(sys.argv) >= 2:
        trc_file = sys.argv[1]
    else:
        print("No file provided. Opening file dialog...")
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw() 
            
            selected_file = filedialog.askopenfilename(
                title="Select a 17-joint .trc Motion File",
                filetypes=[("TRC files", "*.trc"), ("All files", "*.*")],
                initialdir=os.getcwd()
            )
            root.destroy()
            
            if selected_file:
                trc_file = selected_file
        except ImportError:
            print("Error: 'tkinter' not found. Please provide a file via command line.")
            return

    if not trc_file or not os.path.exists(trc_file):
        print("❌ Error: Valid file not selected. Exiting.")
        return

    print(f"Loading motion: {trc_file}")
    try:
        # Load the raw 17-joint data
        data = parse_trc_robust(trc_file)
        
        # TRC data is usually saved in millimeters. If values are large, convert to meters
        if np.max(np.abs(data)) > 10.0:
            data = data / 1000.0
            
        print(f"Data shape: {data.shape} -> ({data.shape[0]} frames, {data.shape[1]} joints, 3 coords)")
        
        if data.shape[1] > 20:
            print("⚠️ WARNING: This file has more than 17 joints.")
            print("This visualizer is specifically designed for the raw 17-joint MotionBERT skeleton.")
            print("Please use visualization_motion.py for 64-marker files.")
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return

    print("Starting Animation...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize lines
    lines = [ax.plot([], [], [], 'b-', linewidth=2.5, alpha=0.8)[0] for _ in SKELETON_EDGES]
    
    # Optional: Add joints as scatter points
    # scatter = ax.scatter([], [], [], c='red', s=20)
    
    # Dynamically scale the plot to fit the skeleton perfectly
    max_range = np.max([
        np.max(data[:, :, 0]) - np.min(data[:, :, 0]),
        np.max(data[:, :, 2]) - np.min(data[:, :, 2]),
        np.max(data[:, :, 1]) - np.min(data[:, :, 1])
    ]) / 2.0

    mid_x = (np.max(data[:, :, 0]) + np.min(data[:, :, 0])) * 0.5
    mid_y = (np.max(data[:, :, 2]) + np.min(data[:, :, 2])) * 0.5
    mid_z = (np.max(data[:, :, 1]) + np.min(data[:, :, 1])) * 0.5
    
    # Add a 10% padding so it doesn't touch the edges
    max_range *= 1.1

    ax.set_xlim([mid_x - max_range, mid_x + max_range])
    ax.set_ylim([mid_y - max_range, mid_y + max_range])
    ax.set_zlim([mid_z - max_range, mid_z + max_range])
    
    ax.set_xlabel('X (Side-to-Side)')
    ax.set_ylabel('Depth (Front-to-Back)')
    ax.set_zlabel('Height')
    
    title_text = ax.set_title("Initializing...", loc='left', fontsize=12, fontfamily='monospace')
    
    # Interactive view settings
    ax.view_init(elev=20., azim=45)

    ani = animation.FuncAnimation(
        fig, 
        update_plot, 
        frames=data.shape[0], 
        fargs=(data, lines, title_text), 
        interval=1000/30, # 30 FPS playback
        blit=False
    )
    
    plt.show()

if __name__ == "__main__":
    main()
