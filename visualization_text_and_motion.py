import os
import cv2
import argparse
import numpy as np

def load_captions(caption_path):
    """
    Parses the CSV format: Frame,Rank,Probability,Sentence
    Or plain text if it's the narrative transcript.
    """
    captions = {}
    if not caption_path or not os.path.exists(caption_path):
        print(f"Warning: Caption file not found at {caption_path}")
        return captions

    with open(caption_path, 'r') as f:
        # Check if it's the new Narrative Transcript format
        first_line = f.readline()
        f.seek(0)
        
        if "Narrative Transcript" in first_line:
            # Parse the Narrative Transcript
            lines = f.readlines()
            current_frame = 0
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith("[") and "s -" in line:
                    # Parse time: [0.0s - 2.0s]
                    try:
                        start_s = float(line.split("s -")[0].strip().strip("["))
                        current_frame = int(start_s * 30.0) # Assuming 30fps
                    except:
                        pass
                elif line.startswith("Narrative:"):
                    captions[current_frame] = line
        else:
            # Parse the old CSV format
            raw_entries = {}
            header = next(f) # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    try:
                        frame = int(parts[0])
                        rank = parts[1]
                        prob = float(parts[2])
                        text = ",".join(parts[3:])
                        
                        if frame not in raw_entries:
                            raw_entries[frame] = []
                        raw_entries[frame].append((rank, prob, text))
                    except ValueError:
                        continue
            
            for frame, entries in raw_entries.items():
                entries.sort(key=lambda x: int(x[0]))
                formatted_lines = []
                for rank, prob, text in entries:
                    formatted_lines.append(f"{rank}. [{prob:.1%}] {text}")
                captions[frame] = "\n".join(formatted_lines)
                
    return captions

def wrap_text(text, font, font_scale, thickness, max_width):
    """Wraps text to fit within a given width."""
    words = text.split()
    lines = []
    current_line = words[0] if words else ""
    
    for word in words[1:]:
        test_line = current_line + " " + word
        size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
        if size[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def main():
    parser = argparse.ArgumentParser(description="Fast OpenCV Video + Text Visualizer")
    parser.add_argument("--video", type=str, default="cam-001.mp4", help="Path to MP4")
    parser.add_argument("--text", type=str, default="cam-001_narrative_transcript.txt", help="Path to predicted text/csv")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return

    print(f"Loading captions: {args.text}")
    captions = load_captions(args.text)
    
    print(f"Opening video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
    delay = int(1000 / fps)

    window_name = "Video + Text Visualization (Press 'Q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 700)

    frame_idx = 0
    current_text = "(Waiting for analysis...)"
    
    # UI Constants
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    bg_color = (0, 0, 0)
    text_color = (255, 255, 255)

    print("Playing video... Press 'Q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            continue

        # Check for caption update
        valid_frames = [f for f in captions.keys() if f <= frame_idx]
        if valid_frames:
            closest_frame = max(valid_frames)
            current_text = captions[closest_frame]

        # Draw a black rectangle at the top to hold the text
        frame_h, frame_w = frame.shape[:2]
        
        # Format text lines (handle multiple ranks or wrapped narrative)
        raw_lines = current_text.split('\n')
        wrapped_lines = []
        for rl in raw_lines:
            wrapped_lines.extend(wrap_text(rl, font, font_scale, thickness, frame_w - 40))
            
        header_height = max(80, len(wrapped_lines) * 25 + 30)
        
        # Add black banner at the top
        canvas = np.zeros((frame_h + header_height, frame_w, 3), dtype=np.uint8)
        canvas[header_height:, :] = frame
        
        # Draw frame number
        cv2.putText(canvas, f"Frame: {frame_idx}", (20, 25), font, 0.5, (150, 150, 150), 1)
        
        # Draw text lines
        y_offset = 55
        for line in wrapped_lines:
            cv2.putText(canvas, line, (20, y_offset), font, font_scale, text_color, thickness, cv2.LINE_AA)
            y_offset += 25

        cv2.imshow(window_name, canvas)
        
        # Wait and handle keyboard input
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()