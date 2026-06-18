import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys

# --- IMPORTS ---
try:
    from utils.data_utils import parse_trc_robust, normalize_and_center_data
except ImportError:
    # Handle cases where the script is run from different locations
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from utils.data_utils import parse_trc_robust, normalize_and_center_data

# --- SKELETON DEFINITION ---
# Matches the user's latest update in visualization_text_and_motion.py
SKELETON_EDGES = [
    [38, 7],  [42, 7],       # Hips
    [7, 8], [8, 9], [9, 10], [10, 13], [13, 14], [14, 15], # Spine
    [18, 16], [17, 16], [15, 17], [15, 18],   # Head
    [15, 20], [15, 21],                       # Shoulders
    [20, 22], [22, 28], [28, 26], [26, 32],   # Right Arm
    [21, 24], [24, 31], [31, 29], [29, 35],   # Left Arm
     [38, 39], [39, 46], [46, 56], # Right Leg
     [42, 43], [43, 49], [49, 62]  # Left Leg
]

def load_captions(caption_path):
    """Parses the CSV format: Frame,Rank,Probability,Sentence"""
    captions = {}
    if not os.path.exists(caption_path):
        return captions
    raw_entries = {}
    with open(caption_path, 'r') as f:
        try:
            header = next(f) # Skip header
        except StopIteration:
            return captions
            
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                try:
                    frame = int(parts[0])
                    rank = parts[1]
                    prob = float(parts[2])
                    text = ",".join(parts[3:])
                    if frame not in raw_entries: raw_entries[frame] = []
                    raw_entries[frame].append((rank, prob, text))
                except ValueError: continue
                
    for frame, entries in raw_entries.items():
        entries.sort(key=lambda x: int(x[0]))
        formatted_lines = [f"{rank}. [{prob:.1%}] {text}" for rank, prob, text in entries]
        captions[frame] = "\n".join(formatted_lines)
    return captions

def update_plot(frame, data, lines, title_text, captions, state):
    idx = frame % data.shape[0]
    current_pose = data[idx]
    for i, (start_idx, end_idx) in enumerate(SKELETON_EDGES):
        if start_idx < len(current_pose) and end_idx < len(current_pose):
            p1, p2 = current_pose[start_idx], current_pose[end_idx]
            # TRC Mapping: X->0, Z->1 (Up), Y->2 (Depth) -> Plot(x, z, y)
            x, y, z = [p1[0], p2[0]], [p1[2], p2[2]], [p1[1], p2[1]]
            lines[i].set_data(x, y)
            lines[i].set_3d_properties(z)
            
    if idx in captions: 
        state[0] = captions[idx]
        
    title_text.set_text(f"Frame {idx}\n{state[0]}")
    return lines + [title_text]

def main():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("Error: 'tkinter' not found. Please ensure it is installed.")
        return

    # 1. Interactive Selection
    root = tk.Tk()
    root.withdraw()
    
    print("Selecting TRC file...")
    trc_path = filedialog.askopenfilename(
        title="1. Select .trc Motion File", 
        filetypes=[("TRC files", "*.trc"), ("All files", "*.*")]
    )
    if not trc_path: 
        print("No TRC file selected. Exiting.")
        return
    
    print("Selecting Caption file (Optional)...")
    caption_path = filedialog.askopenfilename(
        title="2. Select .txt Caption File (Optional - Cancel to skip)", 
        filetypes=[("TXT files", "*.txt"), ("All files", "*.*")]
    )
    
    print("Selecting Output location...")
    out_path = filedialog.asksaveasfilename(
        title="3. Save as MP4", 
        defaultextension=".mp4", 
        filetypes=[("MP4 files", "*.mp4")]
    )
    if not out_path: 
        print("No output location selected. Exiting.")
        return

    root.destroy()

    # 2. Load Data
    print(f"Loading motion: {trc_path}")
    raw_data = parse_trc_robust(trc_path)
    data, _ = normalize_and_center_data(raw_data, root_idx=0)
    
    captions = {}
    if caption_path:
        print(f"Loading captions: {caption_path}")
        captions = load_captions(caption_path)
    
    # 3. Setup Figure (Exact same style as visualization_text_and_motion.py)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    lines = [ax.plot([], [], [], 'b-', linewidth=1.5, alpha=0.7)[0] for _ in SKELETON_EDGES]
    
    ax.set_xlabel('X')
    ax.set_ylabel('Depth')
    ax.set_zlabel('Height')
    
    radius = 1.5
    ax.set_xlim([-radius, radius])
    ax.set_ylim([-radius, radius])
    ax.set_zlim([0, radius*2])
    
    title_text = ax.set_title("Initializing...", loc='left', fontsize=10, fontfamily='monospace')
    state = ["(Waiting for analysis...)"] if captions else [""]

    fps = 30
    ani = animation.FuncAnimation(
        fig, 
        update_plot, 
        frames=data.shape[0], 
        fargs=(data, lines, title_text, captions, state), 
        blit=False
    )
    
    # 4. Save to MP4
    print(f"Exporting to: {out_path}")
    print("Please wait, this may take a minute depending on the sequence length...")
    
    try:
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='MotionToMotion'), bitrate=2000)
        ani.save(out_path, writer=writer)
        print(f"✅ Success! Video saved to: {out_path}")
    except Exception as e:
        print(f"❌ Error during export: {e}")
        print("Make sure 'ffmpeg' is installed on your system.")

if __name__ == "__main__":
    main()
