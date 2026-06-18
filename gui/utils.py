
import sys

class PrintLogger:
    """Redirects sys.stdout to a Tkinter Text widget."""
    def __init__(self, textbox):
        self.textbox = textbox

    def write(self, text):
        self.textbox.insert("end", text)
        self.textbox.see("end")  # Auto-scroll

    def flush(self):
        pass
