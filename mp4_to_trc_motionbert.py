"""
MP4 to TRC via MotionBERT
==========================
Uses YOLO Pose (2D extraction) → DSTformer backbone (3D lifting) → Trained TRC Head (64-marker projection)
to convert any MP4 video into a dense 64-marker TRC motion file.

Usage:
    python3 mp4_to_trc_motionbert.py
    python3 mp4_to_trc_motionbert.py cam-001.mp4
    python3 mp4_to_trc_motionbert.py cam-001.mp4 --output my_output.trc
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
# 1. TRC HEAD (must match training architecture)
# ==========================================
class TRCHead(nn.Module):
    def __init__(self):
        super(TRCHead, self).__init__()
        self.trc_head = nn.Sequential(
            nn.Linear(51, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 192)
        )
    
    def forward(self, x):
        return self.trc_head(x)

# ==========================================
# 2. FULL PIPELINE
# ==========================================
class MP4toTRCMotionBERT:
    def __init__(self, backbone_checkpoint, head_checkpoint):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing MotionBERT pipeline on: {self.device}")
        
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
        
        # --- C. Load Trained TRC Head ---
        print("Loading trained TRC head...")
        self.head = TRCHead().to(self.device)
        
        if not os.path.exists(head_checkpoint):
            print(f"⚠️ Warning: TRC head weights not found at {head_checkpoint}.")
            print("Running in untrained mode.")
        else:
            head_ckpt = torch.load(head_checkpoint, map_location=self.device)
            self.head.trc_head.load_state_dict(head_ckpt['trc_head_state_dict'])
            epoch = head_ckpt.get('epoch', '?')
            loss = head_ckpt.get('loss', '?')
            print(f"   ✅ TRC head loaded (trained epoch {epoch}, loss {loss})")
        
        self.head.eval()
        
    def convert_video(self, video_path, output_trc_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps != fps:
            fps = 30.0 
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\nProcessing '{video_path}' | FPS: {fps:.2f} | Total Frames: {frame_count}")

        # --- PHASE 1: Extract all 2D keypoints from video ---
        print("Phase 1: Extracting 2D poses with YOLO...")
        all_keypoints = []  # Will store (17, 3) per frame: [x, y, conf]
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.pose_model(frame, conf=0.5, verbose=False)
            
            if len(results[0].boxes) > 0 and results[0].keypoints is not None:
                kpts_xyn = results[0].keypoints.xyn[0]   # (17, 2) normalized
                kpts_conf = results[0].keypoints.conf[0]  # (17,)
                
                # Core visibility gate
                core_vis = (kpts_conf[5] + kpts_conf[6] + kpts_conf[11] + kpts_conf[12]) / 4.0
                
                if core_vis < 0.60:
                    if len(all_keypoints) > 0:
                        all_keypoints.append(all_keypoints[-1].copy())
                    else:
                        all_keypoints.append(np.zeros((17, 3), dtype=np.float32))
                else:
                    frame_kpts = np.zeros((17, 3), dtype=np.float32)
                    for j in range(17):
                        frame_kpts[j, 0] = kpts_xyn[j, 0].item()
                        frame_kpts[j, 1] = -kpts_xyn[j, 1].item()  # Y-flip for TRC convention
                        frame_kpts[j, 2] = kpts_conf[j].item()      # Confidence as 3rd channel
                    
                    # Hip centering (using midpoint of hips: joints 11 and 12)
                    hip_x = (frame_kpts[11, 0] + frame_kpts[12, 0]) / 2.0
                    hip_y = (frame_kpts[11, 1] + frame_kpts[12, 1]) / 2.0
                    frame_kpts[:, 0] -= hip_x
                    frame_kpts[:, 1] -= hip_y
                    
                    all_keypoints.append(frame_kpts)
            else:
                if len(all_keypoints) > 0:
                    all_keypoints.append(all_keypoints[-1].copy())
                else:
                    all_keypoints.append(np.zeros((17, 3), dtype=np.float32))

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
        print("Phase 2: Lifting to 3D with MotionBERT + TRC Head...")
        
        CHUNK_SIZE = 243
        all_keypoints_np = np.array(all_keypoints, dtype=np.float32)  # (N, 17, 3)
        
        # Output buffer
        trc_output = np.zeros((total_frames, 192), dtype=np.float32)
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
                lifted_flat = lifted_3d.reshape(B, F, J * C)
                
                # TRC Head → (1, 243, 192)
                trc_pred = self.head(lifted_flat)
                trc_pred_np = trc_pred.cpu().squeeze(0).numpy()  # (243, 192)
                
                # Only take the valid frames (not the padding)
                valid_pred = trc_pred_np[:chunk_len]
                
                # Accumulate with overlap averaging
                trc_output[start:start+chunk_len] += valid_pred
                count_buffer[start:start+chunk_len] += 1.0
                
                if (start + CHUNK_SIZE) % 1000 < stride:
                    print(f"   Processed up to frame {min(start+CHUNK_SIZE, total_frames)}/{total_frames}...")
        
        # Average overlapping regions
        count_buffer = np.maximum(count_buffer, 1.0)
        trc_output = trc_output / count_buffer[:, np.newaxis]
        
        # --- PHASE 3: Temporal smoothing ---
        print("Phase 3: Applying temporal smoothing...")
        window_length = min(11, total_frames)
        if window_length % 2 == 0:
            window_length -= 1
        
        if window_length >= 3:
            trc_output = savgol_filter(trc_output, window_length=window_length, polyorder=3, axis=0)
        
        # --- PHASE 4: Write TRC ---
        self._write_trc(output_trc_path, trc_output, fps)

    def _write_trc(self, output_path, sequence_192, fps):
        print(f"Writing to {output_path}...")
        num_frames = len(sequence_192)
        num_markers = 64
        
        with open(output_path, 'w', newline='') as f:
            f.write("PathFileType\t4\t(X/Y/Z)\tpredicted_motion\n")
            f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
            f.write(f"{fps:.2f}\t{fps:.2f}\t{num_frames}\t{num_markers}\tmm\t{fps:.2f}\t1\t{num_frames}\n")
            
            marker_names = [f"Marker_{i}" for i in range(1, num_markers + 1)]
            f.write("Frame#\tTime\t" + "\t\t\t".join(marker_names) + "\t\t\n")
            
            xyz_headers = []
            for i in range(1, num_markers + 1):
                xyz_headers.extend([f"X{i}", f"Y{i}", f"Z{i}"])
            f.write("\t\t" + "\t".join(xyz_headers) + "\n\n")

            for frame_idx, frame_data in enumerate(sequence_192):
                time_sec = frame_idx / fps
                row = [frame_idx + 1, f"{time_sec:.5f}"]
                frame_data_mm = frame_data * 1000.0
                row.extend([f"{val:.5f}" for val in frame_data_mm])
                f.write("\t".join(map(str, row)) + "\n")
                
        print(f"✅ Success! TRC file saved to {output_path}")

# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MP4 to 64-marker TRC using MotionBERT.")
    parser.add_argument("input_video", type=str, nargs='?', help="Path to the input MP4 video.")
    parser.add_argument("--output", type=str, default=None, help="Path to the output TRC file.")
    parser.add_argument("--backbone", type=str, 
                        default=os.path.join("MotionBERT", "checkpoint", "pretrain", "MB_pretrain", "best_epoch.bin"),
                        help="Path to the MotionBERT pretrained backbone checkpoint.")
    parser.add_argument("--head", type=str, default="motionbert_trc_head_best.pth",
                        help="Path to the trained TRC head weights.")
    
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
        out_path = f"{base_name}_motionbert_3d.trc"
    
    try:
        converter = MP4toTRCMotionBERT(
            backbone_checkpoint=args.backbone,
            head_checkpoint=args.head
        )
        converter.convert_video(video_path, out_path)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
