
import customtkinter as ctk
import sys
import os

# Local Imports
from gui.frames.motion_to_text import MotionToTextFrame
from gui.frames.motion_to_motion import MotionToMotionFrame
from gui.frames.text_to_motion import TextToMotionFrame
from gui.frames.settings import SettingsFrame
from gui.utils import PrintLogger
from gui.fonts import Fonts

class MotionStudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Initialize Fonts after root window is created
        Fonts.initialize()
        Fonts.set_scale(2.0)

        self.title("Motion Studio - Deep Learning Hub")
        self.geometry("1400x900")

        # Config
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Layout: 2 Columns (Sidebar, Main)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Motion Studio", font=Fonts.font_title)
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.btn_m2t = ctk.CTkButton(self.sidebar_frame, text="Motion to Text", command=self.show_m2t, font=Fonts.font_button)
        self.btn_m2t.grid(row=1, column=0, padx=20, pady=10)

        self.btn_m2m = ctk.CTkButton(self.sidebar_frame, text="Motion to Motion", command=self.show_m2m, font=Fonts.font_button)
        self.btn_m2m.grid(row=2, column=0, padx=20, pady=10)

        self.btn_t2m = ctk.CTkButton(self.sidebar_frame, text="Text to Motion", command=self.show_t2m, font=Fonts.font_button)
        self.btn_t2m.grid(row=3, column=0, padx=20, pady=10)

        self.btn_settings = ctk.CTkButton(self.sidebar_frame, text="Settings", command=self.show_settings, font=Fonts.font_button, fg_color="#555555")
        self.btn_settings.grid(row=5, column=0, padx=20, pady=20, sticky="s")

        # 2. Main Content Area
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        # 3. Console Output (Bottom)
        self.console_frame = ctk.CTkFrame(self, height=150, corner_radius=0)
        self.console_frame.grid(row=1, column=1, sticky="ew", padx=20, pady=(0, 20))
        
        self.console_label = ctk.CTkLabel(self.console_frame, text="Console Output", anchor="w", font=Fonts.font_header)
        self.console_label.pack(fill="x", padx=5, pady=2)
        
        self.console_textbox = ctk.CTkTextbox(self.console_frame, height=120, font=Fonts.font_small)
        self.console_textbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Redirect stdout
        sys.stdout = PrintLogger(self.console_textbox)
        print("[INFO] Application Initialized.")
        
        # --- Initialize Modules ---
        self.m2t_frame = MotionToTextFrame(self.main_frame)
        self.m2m_frame = MotionToMotionFrame(self.main_frame)
        self.t2m_frame = TextToMotionFrame(self.main_frame)
        self.settings_frame = SettingsFrame(self.main_frame)

        # Register observer
        Fonts.register_observer(self.refresh_ui)

        # Default View
        self.show_m2t()

    def show_m2t(self):
        self.clear_frame()
        self.m2t_frame.pack(fill="both", expand=True)

    def show_m2m(self):
        self.clear_frame()
        self.m2m_frame.pack(fill="both", expand=True)

    def show_t2m(self):
        self.clear_frame()
        self.t2m_frame.pack(fill="both", expand=True)

    def show_settings(self):
        self.clear_frame()
        self.settings_frame.pack(fill="both", expand=True)

    def refresh_ui(self):
        # Update fonts on sidebar and console
        self.logo_label.configure(font=Fonts.font_title)
        self.btn_m2t.configure(font=Fonts.font_button)
        self.btn_m2m.configure(font=Fonts.font_button)
        self.btn_t2m.configure(font=Fonts.font_button)
        self.btn_settings.configure(font=Fonts.font_button)
        self.console_label.configure(font=Fonts.font_header)
        self.console_textbox.configure(font=Fonts.font_small)

    def clear_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.pack_forget()
