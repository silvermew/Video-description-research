import torch
import numpy as np
import os
import sys
import math

# Add parent directory to path to allow running from within 'inference_scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- IMPORTS ---
from models import MotionEncoder, ConditionalDiffusionModel
from utils.scheduler import NoiseScheduler
from utils.data_utils import LazyWindowDataset, parse_trc_robust, load_data_from_trc, save_data_to_trc

# ====================================================================
# 1. CONFIGURATION
# ====================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
ENCODER_PATH = 'checkpoints/best_encoder.pth'
DIFFUSION_PATH = 'checkpoints/best_diffusion.pth'
INPUT_TRC_FILE = 'data/20130712/20130620/cap-003.trc'
OUTPUT_TRC_FILE = 'result/reconstructed_refined.trc'

# Model Hyperparameters
IN_CHANNELS = 3       
CONTEXT_DIM = 256     
MODEL_CHANNELS = 128  
TIME_EMB_DIM = 256
NUM_HEADS = 8
NUM_LAYERS_ENCODER = 1

# Data Params
TRAINING_FPS = 30     
WINDOW_SIZE = 60      
OVERLAP = 10          
R_ASI_IDX, L_ASI_IDX, SACRUM_IDX = 0, 1, 2 

# --- NEW PARAMETER: REFINEMENT STRENGTH ---
# 1.0 = Generate from pure noise (Creative, but desynchronized)
# 0.6 = Clean the existing motion (Keeps timing, fixes jitter)
DENOISING_STRENGTH = 0.6

# ====================================================================
# 2. MAIN FUNCTION
# ====================================================================

def load_and_reconstruct():
    print("\n--- 1. Initializing Models ---")
    
    # Initialize Architecture
    motion_encoder = MotionEncoder(hidden_dim=CONTEXT_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS_ENCODER).to(device)
    diffusion_model = ConditionalDiffusionModel(in_channels=IN_CHANNELS, out_channels=IN_CHANNELS, 
                                                model_channels=MODEL_CHANNELS, context_dim=CONTEXT_DIM, 
                                                time_emb_dim=TIME_EMB_DIM).to(device)

    # Load Weights
    print(f"Loading weights...")
    motion_encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
    diffusion_model.load_state_dict(torch.load(DIFFUSION_PATH, map_location=device))
    motion_encoder.eval()
    diffusion_model.eval()
    
    noise_scheduler = NoiseScheduler(num_train_timesteps=1000)

    # --- 2. Data Preparation ---
    print("\n--- 2. Processing Input Data ---")
    
    # A. Extract Root
    print("Extracting original global trajectory...")
    full_raw_data = parse_trc_robust(INPUT_TRC_FILE) 
    original_root_path = full_raw_data[:, 0:1, :] 
    num_frames_orig = full_raw_data.shape[0]

    # B. Load Normalized Input
    effective_window_size = min(WINDOW_SIZE, num_frames_orig)
    motion_data_tensor, metadata = load_data_from_trc(
        file_path=INPUT_TRC_FILE, target_fps=TRAINING_FPS, window_size=effective_window_size, 
        right_asi_idx=R_ASI_IDX, left_asi_idx=L_ASI_IDX, sacrum_idx=SACRUM_IDX, overlap=OVERLAP
    )
    metadata['original_num_frames'] = num_frames_orig
    motion_data_tensor = motion_data_tensor.to(device) # (B, 60, 64, 3)

    # --- 3. Inference with REFINEMENT ---
    print(f"\n--- 3. Running Diffusion Refinement (Strength: {DENOISING_STRENGTH}) ---")
    
    B = motion_data_tensor.shape[0]
    reconstructed_windows = []
    
    # Calculate starting timestep for Refinement
    start_timestep = int(noise_scheduler.num_train_timesteps * DENOISING_STRENGTH)
    start_timestep = min(start_timestep, noise_scheduler.num_train_timesteps - 1)
    
    print(f"  > Starting from timestep {start_timestep} (Partial Noise)")
    print(f"  > Total Windows to process: {B}")

    for b in range(B):
        # Print progress for the window
        print(f"Processing Window {b+1}/{B}...", end="\r")

        # 1. Get Input
        current_window = motion_data_tensor[b:b+1] # (1, T, M, C)
        
        # 2. Get Context (Condition on the input)
        with torch.no_grad():
            context = motion_encoder(current_window) 
        
        # 3. PREPARE INITIAL NOISY STATE (The "Refinement" Magic)
        if DENOISING_STRENGTH >= 1.0:
            x_t = torch.randn_like(current_window).to(device)
        else:
            noise = torch.randn_like(current_window).to(device)
            timesteps_start = torch.full((1,), start_timestep, device=device, dtype=torch.long)
            x_t = noise_scheduler.add_noise(current_window, noise, timesteps_start)
        
        # 4. Reverse Process Loop
        relevant_timesteps = [t for t in noise_scheduler.timesteps if t <= start_timestep]
        
        for i, t in enumerate(relevant_timesteps):
            timesteps = torch.full((1,), t, device=device, dtype=torch.long)
            
            # --- [ADDED PRINT] Show progression every 100 steps ---
            if (i + 1) % 100 == 0:
                print(f"  Window {b+1}/{B} | Denoising Step {t.item()} / {start_timestep}      ", end="\r")
            
            with torch.no_grad():
                predicted_noise = diffusion_model(x_t, timesteps, context)
            
            # Step backward
            x_t = noise_scheduler.step(predicted_noise, t, x_t).prev_sample
        
        reconstructed_windows.append(x_t)

    print(f"\nAll {B} windows processed.")

    # --- 4. Stitching with COSINE BLENDING ---
    print("\n--- 4. Stitching (Cosine Blending) ---")
    if reconstructed_windows:
        T, M, C = reconstructed_windows[0].shape[1:]
        stride = T - OVERLAP
        num_windows = len(reconstructed_windows)
        total_len = (num_windows - 1) * stride + T
        
        final_pred = torch.zeros((total_len, M, C), device=device)
        weights = torch.zeros((total_len, M, C), device=device)
        
        # --- COSINE WEIGHT CURVE ---
        fade_in = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, OVERLAP, device=device)))
        fade_out = 0.5 * (1 + torch.cos(torch.linspace(0, math.pi, OVERLAP, device=device)))
        
        window_curve = torch.ones(T, device=device)
        window_curve[:OVERLAP] = fade_in
        window_curve[-OVERLAP:] = fade_out
        
        window_curve = window_curve.view(T, 1, 1).expand(T, M, C)

        for i, w in enumerate(reconstructed_windows):
            w_data = w.squeeze(0)
            start_idx = i * stride
            end_idx = start_idx + T
            
            final_pred[start_idx:end_idx] += w_data * window_curve
            weights[start_idx:end_idx] += window_curve
            
        weights[weights == 0] = 1.0 
        full_reconstructed = final_pred / weights
    else:
        return

    # --- 5. Saving ---
    print(f"\n--- 5. Saving to {OUTPUT_TRC_FILE} ---")
    reconstructed_motion_np = full_reconstructed.cpu().numpy()
    save_data_to_trc(reconstructed_motion_np, metadata, OUTPUT_TRC_FILE, insert_root_trajectory=original_root_path) 
    print("✅ Done.")

if __name__ == "__main__":
    load_and_reconstruct()