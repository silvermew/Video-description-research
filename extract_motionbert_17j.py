"""
MotionBERT 17-Joint 3D Extractor
================================
Extracts the raw 17-joint 3D skeleton directly from the frozen MotionBERT backbone.
Outputs a 17-marker TRC file for visualization and downstream adapter training.

Usage:
    python3 extract_motionbert_17j.py cam-001.mp4
"""

import sys
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse
from functools import partial
from scipy.signal import savgol_filter
from ultralytics import YOLO

# Import MotionBERT DSTformer
MOTIONBERT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MotionBERT")
sys.path.insert(0, MOTIONBERT_ROOT)
from lib.model.DSTformer import DSTformer

# ==========================================
# EXTRACTOR PIPELINE
# ==========================================
class MotionBERT17jExtractor:
    def __init__(self, backbone_checkpoint):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing MotionBERT 17-joint extractor on: {self.device}")
        
        # --- A. Load YOLO Pose ---
        print("Loading YOLOv8 Nano Pose model...")
        self.pose_model = YOLO("yolov8n-pose.pt")
        
        # --- B. Load DSTformer Backbone ---
        print("Loading DSTformer backbone...")
        self.backbone = DSTformer(
            dim_in=3, dim_out=3,
            dim_feat=512, dim_rep=512,
            depth=5, num_heads=8, mlp_ratio=2,
            num_joints=17, maxlen=243,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            att_fuse=True
        )
        
        if not os.path.exists(backbone_checkpoint):
            print(f"❌ ERROR: Backbone checkpoint not found at {backbone_checkpoint}")
            sys.exit(1)
        
        ckpt = torch.load(backbone_checkpoint, map_location="cpu")
        state_dict = ckpt.get('model_pos', ckpt.get('state_dict', ckpt))
        clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.backbone.load_state_dict(clean_state, strict=False)
        
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"   ✅ DSTformer backbone loaded.")
        
    def extract_video(self, video_path, output_trc_path, max_frames=2000):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps != fps:
            fps = 30.0 
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is not None:
            frame_count = min(frame_count, max_frames)
        print(f"\nProcessing '{video_path}' | FPS: {fps:.2f} | Total Frames to Process: {frame_count}")

        # --- PHASE 1: Extract all 2D keypoints from video ---
        print("Phase 1: Extracting 2D poses with YOLO...")
        all_keypoints = []  # Will store (17, 3) per frame: [x, y, conf]
        valid_frames = []   # Tracks if YOLO detected a human (True) or not (False)
        
        frame_idx = 0
        while cap.isOpened():
            if max_frames is not None and frame_idx >= max_frames:
                break
                
            ret, frame = cap.read()
            if not ret:
                break

            # STRICT YOLO DETECTION
            results = self.pose_model(frame, conf=0.70, verbose=False)
            
            is_valid = False
            
            if len(results[0].boxes) > 0 and results[0].keypoints is not None:
                kpts_xyn = results[0].keypoints.xyn[0]   # (17, 2) normalized
                kpts_conf = results[0].keypoints.conf[0]  # (17,)
                
                # Core visibility gate (shoulders + hips)
                core_vis = (kpts_conf[5] + kpts_conf[6] + kpts_conf[11] + kpts_conf[12]) / 4.0
                
                if core_vis >= 0.70:
                    is_valid = True
                    
            if not is_valid:
                # If invalid, copy the previous pose to feed MotionBERT so the Transformer 
                # doesn't glitch from discontinuous gaps, but we will zero the TRC output later!
                if len(all_keypoints) > 0:
                    all_keypoints.append(all_keypoints[-1].copy())
                else:
                    all_keypoints.append(np.zeros((17, 3), dtype=np.float32))
                valid_frames.append(False)
            else:
                coco_kpts = np.zeros((17, 3), dtype=np.float32)
                for j in range(17):
                    coco_kpts[j, 0] = kpts_xyn[j, 0].item()
                    coco_kpts[j, 1] = -kpts_xyn[j, 1].item()  # Y-flip for TRC convention
                    coco_kpts[j, 2] = kpts_conf[j].item()      # Confidence
                    
                # Map COCO (YOLO) to H36M (MotionBERT)
                frame_kpts = np.zeros((17, 3), dtype=np.float32)
                frame_kpts[0] = (coco_kpts[11] + coco_kpts[12]) / 2.0  # Pelvis
                frame_kpts[1] = coco_kpts[12]  # R Hip
                frame_kpts[2] = coco_kpts[14]  # R Knee
                frame_kpts[3] = coco_kpts[16]  # R Ankle
                frame_kpts[4] = coco_kpts[11]  # L Hip
                frame_kpts[5] = coco_kpts[13]  # L Knee
                frame_kpts[6] = coco_kpts[15]  # L Ankle
                frame_kpts[8] = (coco_kpts[5] + coco_kpts[6]) / 2.0  # Neck
                frame_kpts[7] = (frame_kpts[0] + frame_kpts[8]) / 2.0  # Spine
                frame_kpts[9] = coco_kpts[0]   # Nose
                frame_kpts[10] = (coco_kpts[3] + coco_kpts[4]) / 2.0 # Head
                frame_kpts[11] = coco_kpts[5]  # L Shoulder
                frame_kpts[12] = coco_kpts[7]  # L Elbow
                frame_kpts[13] = coco_kpts[9]  # L Wrist
                frame_kpts[14] = coco_kpts[6]  # R Shoulder
                frame_kpts[15] = coco_kpts[8]  # R Elbow
                frame_kpts[16] = coco_kpts[10] # R Wrist
                    
                # Hip centering (Pelvis is now at index 0)
                hip_x = frame_kpts[0, 0]
                hip_y = frame_kpts[0, 1]
                frame_kpts[:, 0] -= hip_x
                frame_kpts[:, 1] -= hip_y
                
                all_keypoints.append(frame_kpts)
                valid_frames.append(True)

            frame_idx += 1
            if frame_idx % 200 == 0:
                print(f"   Extracted {frame_idx}/{frame_count} frames...")

        cap.release()
        
        if not all_keypoints:
            print("No frames extracted.")
            return
        
        total_frames = len(all_keypoints)
        print(f"   ✅ Extracted {total_frames} frames of 2D poses.")
        
        # --- PHASE 2: Process through MotionBERT in 243-frame chunks ---
        print("Phase 2: Lifting to 3D with MotionBERT (17-Joint output)...")
        
        CHUNK_SIZE = 243
        all_keypoints_np = np.array(all_keypoints, dtype=np.float32)  # (N, 17, 3)
        
        # Output buffer (17 joints * 3 coords = 51)
        trc_output = np.zeros((total_frames, 51), dtype=np.float32)
        count_buffer = np.zeros(total_frames, dtype=np.float32)  # For averaging overlapping chunks
        
        # Process overlapping chunks for smooth transitions
        stride = CHUNK_SIZE // 2  # 50% overlap
        
        with torch.no_grad():
            for start in range(0, total_frames, stride):
                end = min(start + CHUNK_SIZE, total_frames)
                chunk = all_keypoints_np[start:end]  # (chunk_len, 17, 3)
                chunk_len = chunk.shape[0]
                
                # Pad to CHUNK_SIZE if the last chunk is shorter
                if chunk_len < CHUNK_SIZE:
                    padding = np.tile(chunk[-1:], (CHUNK_SIZE - chunk_len, 1, 1))
                    chunk_padded = np.concatenate([chunk, padding], axis=0)
                else:
                    chunk_padded = chunk
                
                # Convert to tensor: (1, 243, 17, 3)
                input_tensor = torch.tensor(chunk_padded, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # DSTformer backbone → (1, 243, 17, 3)
                lifted_3d = self.backbone(input_tensor)
                
                # Flatten joints: (1, 243, 51)
                B, F, J, C = lifted_3d.shape
                lifted_flat = lifted_3d.reshape(B, F, J * C).cpu().squeeze(0).numpy() # (243, 51)
                
                # Only take the valid frames (not the padding)
                valid_pred = lifted_flat[:chunk_len]
                
                # Accumulate with overlap averaging
                trc_output[start:start+chunk_len] += valid_pred
                count_buffer[start:start+chunk_len] += 1.0
                
                if (start + CHUNK_SIZE) % 1000 < stride:
                    print(f"   Processed up to frame {min(start+CHUNK_SIZE, total_frames)}/{total_frames}...")
        
        # Average overlapping regions
        count_buffer = np.maximum(count_buffer, 1.0)
        trc_output = trc_output / count_buffer[:, np.newaxis]
        
        # --- ZERO OUT INVALID FRAMES ---
        print("Masking out frames with no human detection...")
        valid_mask = np.array(valid_frames, dtype=bool)
        trc_output[~valid_mask] = 0.0
        
        # --- PHASE 3: Temporal smoothing ---
        print("Phase 3: Applying temporal smoothing...")
        window_length = min(11, total_frames)
        if window_length % 2 == 0:
            window_length -= 1
        
        if window_length >= 3:
            trc_output = savgol_filter(trc_output, window_length=window_length, polyorder=3, axis=0)
        
        # --- PHASE 4: Write TRC ---
        self._write_trc(output_trc_path, trc_output, fps)

    def _write_trc(self, output_path, sequence_51, fps):
        print(f"Writing to {output_path}...")
        num_frames = len(sequence_51)
        num_markers = 17 # MotionBERT native 17 joints
        
        with open(output_path, 'w', newline='') as f:
            f.write("PathFileType\t4\t(X/Y/Z)\tpredicted_motion\n")
            f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
            f.write(f"{fps:.2f}\t{fps:.2f}\t{num_frames}\t{num_markers}\tmm\t{fps:.2f}\t1\t{num_frames}\n")
            
            # Use Human3.6M standard joint names for clarity if desired, but Marker_X is fine
            marker_names = [f"MB_Joint_{i}" for i in range(1, num_markers + 1)]
            f.write("Frame#\tTime\t" + "\t\t\t".join(marker_names) + "\t\t\n")
            
            xyz_headers = []
            for i in range(1, num_markers + 1):
                xyz_headers.extend([f"X{i}", f"Y{i}", f"Z{i}"])
            f.write("\t\t" + "\t".join(xyz_headers) + "\n\n")

            for frame_idx, frame_data in enumerate(sequence_51):
                time_sec = frame_idx / fps
                row = [frame_idx + 1, f"{time_sec:.5f}"]
                frame_data_mm = frame_data * 1000.0
                row.extend([f"{val:.5f}" for val in frame_data_mm])
                f.write("\t".join(map(str, row)) + "\n")
                
        print(f"✅ Success! 17-Joint TRC file saved to {output_path}")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 17-joint 3D skeleton from MP4 using MotionBERT.")
    parser.add_argument("input_video", type=str, nargs='?', help="Path to the input MP4 video.")
    parser.add_argument("--output", type=str, default=None, help="Path to the output 17-joint TRC file.")
    parser.add_argument("--backbone", type=str, 
                        default=os.path.join("MotionBERT", "checkpoint", "pretrain", "MB_pretrain", "best_epoch.bin"),
                        help="Path to the MotionBERT pretrained backbone checkpoint.")
    parser.add_argument("--max_frames", type=int, default=2000, 
                        help="Maximum number of frames to process (default: 2000). Set to -1 for full video.")
    
    args = parser.parse_args()
    video_path = args.input_video
    
    if not video_path:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        print("Please select an MP4 file in the dialog window...")
        video_path = filedialog.askopenfilename(
            title="Select an MP4 video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if not video_path:
            print("No file selected. Exiting.")
            sys.exit()
    
    if not os.path.exists(video_path):
        print(f"Error: The input video '{video_path}' does not exist.")
        sys.exit(1)
    
    out_path = args.output
    if out_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_path = f"{base_name}_17j.trc"
    
    max_frames_val = None if args.max_frames == -1 else args.max_frames
    
    try:
        extractor = MotionBERT17jExtractor(backbone_checkpoint=args.backbone)
        extractor.extract_video(video_path, out_path, max_frames=max_frames_val)
    except Exception as e:
        print(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()
