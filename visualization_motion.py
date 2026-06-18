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

# --- SKELETON DEFINITION (64 Markers) ---
SKELETON_EDGES = [
    [38, 7],  [42, 7],       # Hips
    [7, 8], [8, 9], [9, 10], [10, 13], [13, 14], [14, 15], # Spine
    [18, 16], [17, 16], [15, 17], [15, 18],   # Head
    [15, 20], [15, 21],                       # Shoulders
    [20, 22], [22, 28], [28, 26], [26, 32],   # Right Arm
    [21, 24], [24, 31], [31, 29], [29, 35],   # Left Arm
    [38, 39], [39, 46], [46, 56],             # Right Leg
    [42, 43], [43, 49], [49, 62]              # Left Leg
]

def update_plot(frame, data, lines, title_text):
    """Update function for the animation."""
    idx = frame % data.shape[0]
    current_pose = data[idx] 
    
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

    title_text.set_text(f"Frame {idx}")
    return lines + [title_text]

def main():
    trc_file = "/home/dl-box/Documents/MotionToMotion/MotionToMotion/cam-001_predicted.trc"

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
                title="Select a .trc Motion File",
                filetypes=[("TRC files", "*.trc"), ("All files", "*.*")],
                initialdir=os.path.join(os.getcwd(), "results")
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
        # [FIX] Load the raw, real-world data directly! No normalization!
        data = parse_trc_robust(trc_file)
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return
    
    # Setup Figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize skeleton lines
    lines = []
    for _ in SKELETON_EDGES:
        line, = ax.plot([], [], [], 'b-', linewidth=2.0, alpha=0.8)
        lines.append(line)
    
    ax.set_xlabel('X (Side - mm)')
    ax.set_ylabel('Y (Depth - mm)')
    ax.set_zlabel('Z (Height - mm)')
    
    # [FIX] Set view limits to real human dimensions (millimeters)
    # 1000mm = 1 meter. So the floor is 2x2 meters, height is 2 meters.
    ax.set_xlim([-1000, 1000])
    ax.set_ylim([-1000, 1000])
    ax.set_zlim([0, 2000])
    
    ax.set_box_aspect([1, 1, 1])
    
    title_text = ax.set_title("Initializing...", loc='left', fontsize=12)
    print("Starting Animation...")
    
    fps = 30
    ani = animation.FuncAnimation(
        fig, 
        update_plot, 
        frames=data.shape[0], 
        fargs=(data, lines, title_text),
        interval=1000/fps, 
        blit=False
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()