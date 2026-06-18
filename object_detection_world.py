from ultralytics import YOLO
import cv2
import os
import sys

def detect_with_yolo_world(video_path, custom_classes=None, sample_rate=15):
    """
    Runs YOLO-World open-vocabulary object detection on a video.
    """
    print("Loading YOLOv8s-Worldv2 (Open-Vocabulary Object Detector)...")
    model = YOLO("yolov8s-world.pt")

    if custom_classes:
        print(f"Setting custom search classes: {custom_classes}")
        model.set_classes(custom_classes)
    else:
        print("Using default extensive YOLO-World vocabulary.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_txt_path = f"{base_name}_yoloworld_detections.txt"

    print(f"Processing video. Sampling log every {sample_rate} frames...")
    print("Press 'q' on the video window to quit early.")

    with open(output_txt_path, 'w') as f:
        f.write(f"YOLO-World Object Detection Log for: {video_path}\n")
        if custom_classes:
            f.write(f"Custom Target Classes: {', '.join(custom_classes)}\n")
        f.write("="*50 + "\n")

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO-World inference on every frame for smooth visualization
            results = model(frame, verbose=False)[0]
            
            # Draw bounding boxes and labels directly on the frame
            annotated_frame = results.plot()
            
            # Display the video
            cv2.imshow("YOLO-World Object Detection", annotated_frame)
            
            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Only log to text file every Nth frame to prevent massive log files
            if frame_idx % sample_rate == 0:
                detected_objects = []
                
                # Extract class names from the results
                for box in results.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    conf = float(box.conf[0])
                    
                    # Only keep confident detections
                    if conf > 0.1: # YOLO-World confidence distribution differs slightly
                        detected_objects.append(class_name)

                # Remove duplicates for a cleaner log
                unique_objects = list(set(detected_objects))
                
                # Format the log line with a timestamp
                timestamp = frame_idx / fps
                log_line = f"Frame {frame_idx:04d} ({timestamp:.2f}s): {', '.join(unique_objects) if unique_objects else 'None'}"
                
                print(log_line)
                f.write(log_line + "\n")

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ YOLO-World detection log saved to {output_txt_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLO-World Object Detection")
    parser.add_argument("input_video", type=str, nargs='?', help="Path to the input video.")
    
    # You can pass custom classes from the terminal!
    # Example: python3 object_detection_world.py --classes dog cat "coffee mug" laptop
    parser.add_argument("--classes", type=str, nargs='+', help="Custom objects to search for (Open Vocabulary).", default=None)
    
    args = parser.parse_args()
    
    video_input = args.input_video
    
    if not video_input:
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            print("Please select a video file in the dialog window...")
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
        detect_with_yolo_world(video_input, custom_classes=args.classes, sample_rate=15)
    else:
        print(f"Error: The video '{video_input}' does not exist.")
