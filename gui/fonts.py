
import customtkinter as ctk

class FontManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FontManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        if self.initialized: return

        # Default Sizes
        self.size_title = 26
        self.size_header = 18
        self.size_body = 16
        self.size_small = 14
        
        # Font Objects
        self.font_title = ctk.CTkFont(size=self.size_title, weight="bold")
        self.font_header = ctk.CTkFont(size=self.size_header, weight="bold")
        self.font_body = ctk.CTkFont(size=self.size_body)
        self.font_button = ctk.CTkFont(size=self.size_body)
        self.font_small = ctk.CTkFont(size=self.size_small)
        
        # Observers for update
        self.observers = []
        self.initialized = True

    def set_scale(self, scale_factor):
        """
        updates font sizes by a factor (e.g. 1.0, 1.2, 1.5)
        Base sizes: Title=26, Header=18, Body=16, Small=14
        """
        self.size_title = int(26 * scale_factor)
        self.size_header = int(18 * scale_factor)
        self.size_body = int(16 * scale_factor)
        self.size_small = int(14 * scale_factor)
        
        self.update_fonts()

    def update_fonts(self):
        # Recreate font objects or configure them? 
        # CTkFont objects might need to be recreated or updated in place depending on library version.
        # Safest is to recreate and notify widgets to re-configure.
        self.font_title.configure(size=self.size_title)
        self.font_header.configure(size=self.size_header)
        self.font_body.configure(size=self.size_body)
        self.font_button.configure(size=self.size_body)
        self.font_small.configure(size=self.size_small)
        
        self.notify_observers()

    def register_observer(self, callback):
        self.observers.append(callback)

    def notify_observers(self):
        for callback in self.observers:
            callback()

# Global Instance
Fonts = FontManager()
