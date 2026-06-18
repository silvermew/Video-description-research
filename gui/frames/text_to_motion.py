
import customtkinter as ctk
from gui.fonts import Fonts

from gui.fonts import Fonts
from gui.visualization import SkeletonVisualizer

class TextToMotionFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        
        self.lbl_title = ctk.CTkLabel(self, text="Text-to-Motion Generator", font=Fonts.font_title)
        self.lbl_title.grid(row=0, column=0, pady=20)
        
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.grid(row=1, column=0, pady=10)
        
        self.input = ctk.CTkEntry(self.input_frame, width=400, placeholder_text="Enter prompt e.g. 'A person walking forward'", font=Fonts.font_body)
        self.input.pack(side="left", padx=10)
        
        self.btn_gen = ctk.CTkButton(self.input_frame, text="Generate Motion", command=self.generate, font=Fonts.font_button)
        self.btn_gen.pack(side="left", padx=10)
        
        self.vis_panel = ctk.CTkFrame(self)
        self.vis_panel.grid(row=2, column=0, sticky="nsew", padx=20, pady=20)
        
        self.visualizer = SkeletonVisualizer(self.vis_panel, title="Generated Motion")
        self.visualizer.setup_plot()
        
        Fonts.register_observer(self.refresh_ui)

    def refresh_ui(self):
        self.lbl_title.configure(font=Fonts.font_title)
        self.input.configure(font=Fonts.font_body)
        self.btn_gen.configure(font=Fonts.font_button)
        
    def generate(self):
        print("[INFO] Text-to-Motion logic coming soon...")
