import cv2
import sys
import os
from ultralytics import YOLO

def visualize_yolo_pose(video_path):
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    # Initialize YOLO Pose (automatically downloads yolov8n-pose.pt if not found)
    print("Loading YOLOv8 Nano Pose model...")
    model = YOLO("yolov8n-pose.pt")

    print("Playing video with YOLO Pose. Press 'q' to quit.")
    
    # Optional: Read FPS to try to play at original speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 60
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame with YOLO Pose
        # conf=0.5 ensures it only draws when it's at least 50% sure it's a human
        results = model(frame, conf=0.5, verbose=False)
        
        # YOLO has a built-in plot function that perfectly draws the skeleton
        annotated_frame = results[0].plot()

        # Show the frame
        cv2.imshow("YOLO Pose Visualization", annotated_frame)
        
        # Play at roughly the original FPS, and listen for 'q' to quit
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLO Pose Visualizer")
    parser.add_argument("input_video", type=str, nargs='?', help="Path to input video")
    args = parser.parse_args()
    
    video_input = args.input_video
    
    # If no video is provided via command line, open a file dialog
    if not video_input:
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            print("Please select a video file...")
            video_input = filedialog.askopenfilename(
                title="Select a Video File",
                filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
            )
            root.destroy()
        except ImportError:
            print("Error: 'tkinter' not found. Please provide a file via command line.")
            sys.exit(1)
            
        if not video_input:
            print("No file selected. Exiting.")
            sys.exit(0)
            
    if os.path.exists(video_input):
        visualize_yolo_pose(video_input)
    else:
        print(f"Error: The file '{video_input}' does not exist.")
