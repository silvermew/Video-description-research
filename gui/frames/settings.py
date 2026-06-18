
import customtkinter as ctk
from gui.fonts import Fonts

class SettingsFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.lbl_title = ctk.CTkLabel(self, text="Application Settings", font=Fonts.font_title)
        self.lbl_title.pack(pady=20)
        
        self.frame_font = ctk.CTkFrame(self)
        self.frame_font.pack(pady=20, padx=20, fill="x")
        
        self.lbl_font_size = ctk.CTkLabel(self.frame_font, text="Text Size Scaling", font=Fonts.font_header)
        self.lbl_font_size.pack(pady=10)
        
        # Font Scale Slider
        self.slider = ctk.CTkSlider(self.frame_font, from_=0.8, to=2.0, number_of_steps=12, command=self.update_scale)
        self.slider.set(1.0) # Default
        self.slider.pack(pady=10, fill="x", padx=50)
        
        self.lbl_value = ctk.CTkLabel(self.frame_font, text="1.0x", font=Fonts.font_body)
        self.lbl_value.pack(pady=5)
        
        # Register for updates to refresh its own labels
        Fonts.register_observer(self.refresh_ui)

    def update_scale(self, value):
        scale = round(value, 1)
        self.lbl_value.configure(text=f"{scale}x")
        Fonts.set_scale(scale)

    def refresh_ui(self):
        # Update own fonts
        self.lbl_title.configure(font=Fonts.font_title)
        self.lbl_font_size.configure(font=Fonts.font_header)
        self.lbl_value.configure(font=Fonts.font_body)
