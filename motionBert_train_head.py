"""
MotionBERT TRC Head Training Script
====================================
Freezes the pretrained DSTformer backbone and trains ONLY a lightweight
linear projection head to map MotionBERT's 17-joint 3D output (51 features)
to our dense 64-marker TRC format (192 features).

Usage:
    1. Download MotionBERT pretrained checkpoint from:
       https://1drv.ms/f/s!AvAdh0LSjEOlgS425shtVi9e5reN?e=6UeBa2
       
    2. Place the checkpoint file (e.g., 'best_epoch.bin') inside:
       MotionBERT/checkpoint/pretrain/MB_pretrain/

    3. Run:
       python3 motionBert_train_head.py
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from functools import partial
import pandas as pd
import numpy as np
import glob

# ==========================================
# 1. IMPORT MOTIONBERT BACKBONE
# ==========================================
# Add the MotionBERT repo to sys.path so we can import DSTformer
MOTIONBERT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MotionBERT")
sys.path.insert(0, MOTIONBERT_ROOT)

from lib.model.DSTformer import DSTformer
from lib.utils.learning import load_pretrained_weights

# ==========================================
# 2. THE WRAPPER MODEL (Frozen Backbone + Trainable Head)
# ==========================================
class MotionBERT_TRC_Lifter(nn.Module):
    def __init__(self, backbone):
        super(MotionBERT_TRC_Lifter, self).__init__()
        
        self.backbone = backbone
        
        # Freeze every parameter in the backbone
        print("Freezing DSTformer backbone weights...")
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        # MotionBERT outputs 17 joints × 3 coords = 51 features per frame
        # Our TRC format needs 64 markers × 3 coords = 192 features per frame
        self.trc_head = nn.Sequential(
            nn.Linear(51, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 192)
        )
        
        # Initialize the head
        for m in self.trc_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x shape: (Batch, Frames, 17, 3)
        with torch.no_grad():
            # DSTformer output: (Batch, Frames, 17, 3) — 3D joint positions
            out_3d = self.backbone(x)
            
        # Flatten last two dims: (Batch, Frames, 51)
        B, F, J, C = out_3d.shape
        out_flat = out_3d.reshape(B, F, J * C)
        
        # Project to TRC space: (Batch, Frames, 192)
        trc_out = self.trc_head(out_flat)
        
        return trc_out

# ==========================================
# 3. THE DATASET (Synthesizes MotionBERT-format input from TRC)
# ==========================================
class MotionBERTTRCDataset(Dataset):
    """
    Loads real 3D TRC data and synthesizes what MotionBERT's backbone
    would receive as input: (Frames, 17, 3) where the 3 channels are
    [x_normalized, y_normalized, confidence].
    
    The target is the full 192-dimensional TRC vector (64 markers × 3).
    """
    def __init__(self, trc_folder, seq_len=243, noise_std=0.005):
        self.seq_len = seq_len
        self.noise_std = noise_std
        self.sequences_input = []  # Will be (seq_len, 17, 3)
        self.sequences_target = [] # Will be (seq_len, 192)
        
        # YOLO COCO 17-keypoint to TRC 64-marker index mapping
        # These are the TRC marker indices (0-63) that correspond to each YOLO joint
        YOLO_TO_TRC_INDICES = [16, 17, 18, 17, 18, 21, 20, 24, 22, 29, 26, 2, 1, 43, 39, 49, 46]
        
        trc_files = glob.glob(os.path.join(trc_folder, "**", "*.trc"), recursive=True)
        print(f"Synthesizing MotionBERT training data from {len(trc_files)} TRC files...")
        
        for file in trc_files:
            try:
                df = pd.read_csv(file, sep='\t', skiprows=6, header=None)
                coords = df.iloc[:, 2:194].interpolate(axis=0).fillna(0.0).values.astype(np.float32)
                coords = coords / 1000.0  # mm to meters
                
                num_frames = coords.shape[0]
                if num_frames < seq_len:
                    continue
                
                stride = seq_len // 2
                for start in range(0, num_frames - seq_len, stride):
                    chunk_3d = coords[start:start+seq_len].copy()
                    
                    # Hip centering (per frame)
                    for f_idx in range(seq_len):
                        hip_x, hip_y, hip_z = chunk_3d[f_idx, 0:3]
                        for i in range(0, 192, 3):
                            chunk_3d[f_idx, i]   -= hip_x
                            chunk_3d[f_idx, i+1] -= hip_y
                            chunk_3d[f_idx, i+2] -= hip_z
                    
                    # Extract synthetic MotionBERT input: (seq_len, 17, 3)
                    # Channel format: [x, y, confidence_score]
                    # We use X and Y from the TRC data, and set confidence to 1.0
                    mb_input = np.zeros((seq_len, 17, 3), dtype=np.float32)
                    for f_idx in range(seq_len):
                        for yolo_idx, trc_marker in enumerate(YOLO_TO_TRC_INDICES):
                            mb_input[f_idx, yolo_idx, 0] = chunk_3d[f_idx, trc_marker*3]     # X
                            mb_input[f_idx, yolo_idx, 1] = chunk_3d[f_idx, trc_marker*3 + 1] # Y
                            mb_input[f_idx, yolo_idx, 2] = 1.0                                # Confidence
                    
                    # Add noise to X and Y to simulate YOLO jitter
                    noise = np.random.normal(0, self.noise_std, (seq_len, 17, 2)).astype(np.float32)
                    mb_input[:, :, :2] += noise
                    
                    self.sequences_input.append(mb_input)
                    self.sequences_target.append(chunk_3d)
                    
            except Exception as e:
                pass
        
        self.sequences_input = np.array(self.sequences_input)
        self.sequences_target = np.array(self.sequences_target)
        print(f"Generated {len(self.sequences_input)} training sequences of {seq_len} frames each.")

    def __len__(self):
        return len(self.sequences_input)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences_input[idx]), torch.tensor(self.sequences_target[idx])

# ==========================================
# 4. THE TRAINING LOOP
# ==========================================
def train_head(trc_folder_path, checkpoint_path, epochs=15, batch_size=8, lr=1e-3, log_every=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- A. Instantiate the DSTformer backbone ---
    print("=" * 60)
    print("  MOTIONBERT TRC HEAD TRAINING")
    print("=" * 60)
    print(f"\n[1/4] Building DSTformer backbone...")
    
    backbone = DSTformer(
        dim_in=3, dim_out=3,
        dim_feat=512, dim_rep=512,
        depth=5, num_heads=8, mlp_ratio=2,
        num_joints=17, maxlen=243,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        att_fuse=True
    )
    
    # --- B. Load pretrained weights ---
    print(f"[2/4] Loading pretrained checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"❌ ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please download from: https://1drv.ms/f/s!AvAdh0LSjEOlgS425shtVi9e5reN?e=6UeBa2")
        print("Place the .bin file at the path above.")
        sys.exit(1)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # MotionBERT checkpoints store weights under 'model_pos' key
    if 'model_pos' in checkpoint:
        state_dict = checkpoint['model_pos']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Strip 'module.' prefix if saved with DataParallel
    clean_state = {}
    for k, v in state_dict.items():
        clean_key = k.replace("module.", "")
        clean_state[clean_key] = v
    
    backbone.load_state_dict(clean_state, strict=False)
    print(f"   ✅ Loaded {len(clean_state)} weight tensors into DSTformer.")
    
    # --- C. Build full model ---
    model = MotionBERT_TRC_Lifter(backbone).to(device)
    
    # Count trainable vs frozen parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"\n   Total parameters:     {total_params:,}")
    print(f"   Frozen (backbone):    {frozen_params:,}")
    print(f"   Trainable (head):     {trainable_params:,}")
    print(f"   Training ratio:       {100*trainable_params/total_params:.2f}%")
    
    # --- D. Build Dataset ---
    # MotionBERT expects maxlen=243 frames, but we can use shorter sequences
    # We use 243 to maximize temporal context
    print(f"\n[3/4] Loading TRC dataset...")
    dataset = MotionBERTTRCDataset(trc_folder=trc_folder_path, seq_len=243, noise_std=0.005)
    
    if len(dataset) == 0:
        print("❌ Dataset is empty. Aborting.")
        return
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # --- E. Optimizer (ONLY on the head) ---
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.trc_head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    total_batches = len(dataloader)
    
    print(f"\n[4/4] Starting training on: {device}")
    print(f"   Dataset size:   {len(dataset)} sequences")
    print(f"   Batch size:     {batch_size}")
    print(f"   Batches/epoch:  {total_batches}")
    print(f"   Epochs:         {epochs}")
    print("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.trc_head.train()
        total_epoch_loss = 0.0
        running_loss = 0.0
        
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            batch_x = batch_x.to(device)  # (B, 243, 17, 3)
            batch_y = batch_y.to(device)  # (B, 243, 192)

            optimizer.zero_grad()
            outputs = model(batch_x)       # (B, 243, 192)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            total_epoch_loss += loss_val
            running_loss += loss_val

            if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == total_batches:
                avg_running = running_loss / min(log_every, batch_idx + 1)
                progress = 100. * (batch_idx + 1) / total_batches
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Batch [{batch_idx+1:04d}/{total_batches}] "
                      f"({progress:05.1f}%) | "
                      f"Loss: {avg_running:.6f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}")
                running_loss = 0.0

        scheduler.step()
        avg_epoch_loss = total_epoch_loss / total_batches
        
        print("-" * 60)
        print(f"✅ Epoch {epoch+1} complete | Avg Loss: {avg_epoch_loss:.6f}")
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'trc_head_state_dict': model.trc_head.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
            }, "motionbert_trc_head_best.pth")
            print(f"💾 New best! Saved to motionbert_trc_head_best.pth (Loss: {best_loss:.6f})")
        print("=" * 60)

    # Final save
    torch.save({
        'trc_head_state_dict': model.trc_head.state_dict(),
        'epoch': epochs,
        'loss': avg_epoch_loss,
    }, "motionbert_trc_head_final.pth")
    print(f"\n💾 Final weights saved to motionbert_trc_head_final.pth")
    print(f"🏆 Best loss achieved: {best_loss:.6f}")

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    # Default checkpoint path — adjust if you placed it elsewhere
    CHECKPOINT = os.path.join("MotionBERT", "checkpoint", "pretrain", "MB_pretrain", "best_epoch.bin")
    
    # You can also try the fine-tuned 3D pose checkpoint for even better initialization:
    # CHECKPOINT = os.path.join("MotionBERT", "checkpoint", "pose3d", "FT_MB_release_MB_ft_h36m", "best_epoch.bin")
    
    train_head(
        trc_folder_path="data/motionsdata",
        checkpoint_path=CHECKPOINT,
        epochs=15,
        batch_size=8,    # MotionBERT is memory-hungry (243 frames × 17 joints × transformer)
        lr=1e-3,
        log_every=5
    )
