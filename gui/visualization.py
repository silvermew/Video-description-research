
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class SkeletonVisualizer:
    def __init__(self, master, width=5, height=5, title="Motion Visualization"):
        self.master = master
        self.fig = plt.Figure(figsize=(width, height), dpi=100, facecolor="#2b2b2b")
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor("#2b2b2b")
        self.ax.tick_params(colors='white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.title = title
        self.lines = []
        self.scatters = []
        self.text_artist = None
        
        # Default Limits
        self.x_limits = (-800, 800)
        self.y_limits = (-800, 800)
        self.z_limits = (0, 1800)

    def setup_plot(self):
        self.ax.clear()
        self.ax.set_facecolor("#2b2b2b")
        self.ax.set_title(self.title, color='white')
        self.ax.set_xlim(*self.x_limits)
        self.ax.set_ylim(*self.y_limits)
        self.ax.set_zlim(*self.z_limits)
        self.ax.set_xlabel('X', color='white')
        self.ax.set_ylabel('Y', color='white')
        self.ax.set_zlabel('Z', color='white')
        self.lines = []
        self.scatters = []
        self.text_artist = None

    def plot_skeleton(self, frame_data, edges, color='#4ED2A6', label=None, clear=True, auto_scale=False):
        """
        frame_data: (Markers, 3)
        edges: List of [start_idx, end_idx]
        """
        if clear:
             self.setup_plot()
        
        # If not clearing, we might want to just update data for lines/scatters if they exist?
        # Typically for animation we clear or update. 
        # To reduce flicker, we could update line data, but structure changes (number of lines) might happen.
        # Simple flicker reduction: Don't clear if we are just overplotting frame-by-frame in animation loop efficiently?
        # Actually, matplotlib verify 'clear' usage. 
        # Let's keep 'clear=True' as default for single frame. Animation loop handles its own clearing or uses clear=False for overlay.
        
        # Draw connections
        for start, end in edges:
            if start < len(frame_data) and end < len(frame_data):
                xs = [frame_data[start, 0], frame_data[end, 0]]
                ys = [frame_data[start, 1], frame_data[end, 1]]
                zs = [frame_data[start, 2], frame_data[end, 2]]
                line, = self.ax.plot(xs, ys, zs, color=color, linewidth=2)
                self.lines.append(line)

        # Draw joints
        scatter = self.ax.scatter(frame_data[:, 0], frame_data[:, 1], frame_data[:, 2], c=color, s=20, label=label)
        self.scatters.append(scatter)
        
        if label:
            self.ax.legend()
            
        if auto_scale:
             # Very simple auto-scale based on this frame
             self.ax.set_xlim(frame_data[:, 0].min(), frame_data[:, 0].max())
             self.ax.set_ylim(frame_data[:, 1].min(), frame_data[:, 1].max())
             self.ax.set_zlim(frame_data[:, 2].min(), frame_data[:, 2].max())

        self.canvas.draw()

    def set_plot_limits(self, x_range, y_range, z_range):
        self.x_limits = x_range
        self.y_limits = y_range
        self.z_limits = z_range
        self.ax.set_xlim(x_range)
        self.ax.set_ylim(y_range)
        self.ax.set_zlim(z_range)

    def update_text(self, text, position=(0, 0, 1800)):
        if self.text_artist:
            self.text_artist.remove()
        self.text_artist = self.ax.text(position[0], position[1], position[2], text, color='yellow', fontsize=12, ha='center')
        self.canvas.draw()
