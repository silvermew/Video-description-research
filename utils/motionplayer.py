import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import filedialog
import os

# --- 1. ROBUST PARSER ---
def parse_trc_for_animation(file_path):
    print(f"Loading {os.path.basename(file_path)}...")
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find Header Line
    header_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("Frame#"):
            header_idx = i
            break

    # Get Marker Names
    header_parts = [p.strip() for p in lines[header_idx].split('\t') if p.strip()]
    marker_names = header_parts[2:]
    
    # Parse Data
    data = []
    # Skip Header + Subheader (X/Y/Z)
    for i in range(header_idx + 2, len(lines)):
        line = lines[i].strip()
        if not line: continue
        try:
            parts = [p for p in line.split('\t') if p.strip()]
            coords = [float(x) for x in parts[2:]] # Skip Frame/Time
            if len(coords) == len(marker_names) * 3:
                data.append(coords)
        except ValueError: continue
            
    # Shape: (Frames, Markers, 3)
    np_data = np.array(data).reshape(len(data), len(marker_names), 3)
    return np_data, marker_names

# --- 2. DEFINE SKELETON CONNECTIONS ---
def get_connections(marker_names):
    # Dynamic mapper to find indices by name
    def find(name_part):
        for idx, m_name in enumerate(marker_names):
            if name_part.lower() in m_name.lower(): return idx
        return -1

    # Standard Skeleton Map
    links = [
        # Trunk
        ('pHipOrigin', 'pSacrum'), ('pSacrum', 'pC7SpinalProcess'), 
        ('pC7SpinalProcess', 'pTopOfHead'),
        # Right Leg
        ('pHipOrigin', 'pRightGreaterTrochanter'), ('pRightGreaterTrochanter', 'pRightKneeLatEpicondyle'),
        ('pRightKneeLatEpicondyle', 'pRightLatMalleolus'), ('pRightLatMalleolus', 'pRightToe'),
        # Left Leg
        ('pHipOrigin', 'pLeftGreaterTrochanter'), ('pLeftGreaterTrochanter', 'pLeftKneeLatEpicondyle'),
        ('pLeftKneeLatEpicondyle', 'pLeftLatMalleolus'), ('pLeftLatMalleolus', 'pLeftToe'),
        # Arms
        ('pC7SpinalProcess', 'pRightAcromion'), ('pRightAcromion', 'pRightOlecranon'), 
        ('pRightOlecranon', 'pRightUlnarStyloid'),
        ('pC7SpinalProcess', 'pLeftAcromion'), ('pLeftAcromion', 'pLeftOlecranon'), 
        ('pLeftOlecranon', 'pLeftUlnarStyloid')
    ]

    connections = []
    for s_name, e_name in links:
        s, e = find(s_name), find(e_name)
        if s != -1 and e != -1: connections.append((s, e))
            
    return connections

# --- 3. ANIMATION PLAYER ---
def play_animation():
    # File Selector
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Generated TRC File", 
        filetypes=[("TRC", "*.trc")]
    )
    if not file_path: return

    # Load Data
    data, names = parse_trc_for_animation(file_path)
    connections = get_connections(names)
    num_frames = data.shape[0]
    
    print(f"Loaded {num_frames} frames.")
    print("Preparing Animation... (Close the window to exit)")

    # Setup Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initial Plot Objects
    scatter = ax.scatter([], [], [], c='blue', s=10)
    lines = [ax.plot([], [], [], c='red', linewidth=2)[0] for _ in connections]
    
    # Set Axes Limits (Fixed to encompass typical human size in mm)
    # Since it's centered at 0, +/- 1500mm is usually safe
    limit = 1200
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z (Depth/Up)')
    ax.set_zlabel('Y (Up/Depth)')
    ax.set_title(f"Generated Motion: {os.path.basename(file_path)}")

    # Update Function for Animation
    def update(frame_idx):
        # Loop playback
        idx = frame_idx % num_frames
        current_pose = data[idx]
        
        # Update Joints (Scatter)
        # Note: Depending on software, Z is often Up. 
        # Here we map TRC(X, Z, Y) -> Plot(X, Y, Z) to orient it upright usually
        # If the person is lying down, try swapping these indices below.
        xs = current_pose[:, 0]
        ys = current_pose[:, 2] # TRC Z mapped to Plot Y
        zs = current_pose[:, 1] # TRC Y mapped to Plot Z
        
        scatter._offsets3d = (xs, ys, zs)
        
        # Update Bones (Lines)
        for line, (start, end) in zip(lines, connections):
            line.set_data([xs[start], xs[end]], [ys[start], ys[end]])
            line.set_3d_properties([zs[start], zs[end]])
            
        return lines + [scatter]

    # Create Animation
    # Interval = 1000ms / 30fps = 33ms per frame
    ani = FuncAnimation(fig, update, frames=range(0, num_frames, 2), interval=33, blit=False)
    
    plt.show()

if __name__ == "__main__":
    play_animation()