
import sys
import os

# Add root to sys.path to ensure modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui.app import MotionStudioApp

if __name__ == "__main__":
    app = MotionStudioApp()
    app.mainloop()
