import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
import subprocess # Added for launching the next script

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Imports
from models.encoder import MotionEncoder
from models.diffusion_model import ConditionalDiffusionModel
from utils.scheduler import NoiseScheduler
from utils.data_utils import LazyWindowDataset

# --- CONFIG ---
PROCESSED_DATA_DIR = "processed_windows"
CHECKPOINT_DIR = 'checkpoints'
CAPTION_SCRIPT_NAME = "train_captioning.py" # The script that will run after training

# Model Hyperparameters
NUM_MARKERS = 64
IN_CHANNELS = 3
CONTEXT_DIM = 256
MODEL_CHANNELS = 128
TIME_EMB_DIM = 256
NUM_HEADS = 8
NUM_LAYERS_ENCODER = 1

# Training Hyperparameters
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200       
LOG_INTERVAL = 100
SAVE_INTERVAL = 10
USE_MIXED_PRECISION = False

if __name__ == "__main__":
    # 1. Setup Data
    if not os.path.exists(PROCESSED_DATA_DIR):
        print(f"ERROR: {PROCESSED_DATA_DIR} does not exist. Run preprocess.py first.")
        sys.exit()

    window_files = [os.path.join(PROCESSED_DATA_DIR, f) for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.npy')]
    print(f"Loaded {len(window_files)} training windows.")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Setup Models
    dataset = LazyWindowDataset(window_files)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, persistent_workers=False)

    encoder = MotionEncoder(hidden_dim=CONTEXT_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS_ENCODER).to(device)
    diffusion = ConditionalDiffusionModel(in_channels=IN_CHANNELS, out_channels=IN_CHANNELS, 
                                          model_channels=MODEL_CHANNELS, context_dim=CONTEXT_DIM, 
                                          time_emb_dim=TIME_EMB_DIM).to(device)
    
    scheduler = NoiseScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(diffusion.parameters()), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() if (USE_MIXED_PRECISION and torch.cuda.is_available()) else None

    # 3. Training Loop
    print("\n--- Starting Training ---")
    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        encoder.train()
        diffusion.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for i, batch in enumerate(loader):
            clean_data = batch.to(device) # (B, 60, 64, 3)
            
            # Forward Diffusion
            noise = torch.randn_like(clean_data)
            t = scheduler.sample_timesteps(clean_data.shape[0]).to(device)
            noisy_data = scheduler.add_noise(clean_data, noise, t)

            # Prediction
            if scaler:
                with torch.cuda.amp.autocast():
                    context = encoder(clean_data)
                    pred_noise = diffusion(noisy_data, t, context)
                    loss = criterion(pred_noise, noise) / GRADIENT_ACCUMULATION_STEPS
                scaler.scale(loss).backward()
            else:
                context = encoder(clean_data)
                pred_noise = diffusion(noisy_data, t, context)
                loss = criterion(pred_noise, noise) / GRADIENT_ACCUMULATION_STEPS
                loss.backward()

            epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

            # Step
            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            if (i + 1) % LOG_INTERVAL == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.6f}")

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.6f}")

        # Save Checkpoints
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_encoder.pth'))
            torch.save(diffusion.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_diffusion.pth'))
            print("  🌟 Saved Best Model")
            
        if (epoch + 1) % SAVE_INTERVAL == 0:
            torch.save(encoder.state_dict(), os.path.join(CHECKPOINT_DIR, f'ckpt_e{epoch+1}_enc.pth'))
            torch.save(diffusion.state_dict(), os.path.join(CHECKPOINT_DIR, f'ckpt_e{epoch+1}_diff.pth'))

    print("Training Complete.")

    # =================================================================
    # 4. LAUNCH DOWNSTREAM TASK (Captioning)
    # =================================================================
    print(f"\n🚀 Phase 1 Complete! Automatically launching Phase 2: {CAPTION_SCRIPT_NAME}...")
    
    # Get the directory of the current script to ensure we find the caption script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    caption_script_path = os.path.join(current_dir, CAPTION_SCRIPT_NAME)
    
    if os.path.exists(caption_script_path):
        try:
            # Run the script and stream the output directly to the terminal
            subprocess.run([sys.executable, caption_script_path], check=True)
            print("\n✅ Entire Pipeline Finished Successfully!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error: The caption training script crashed. Check the logs above. (Error Code: {e.returncode})")
        except KeyboardInterrupt:
            print("\n⚠️ Pipeline interrupted by user.")
    else:
        print(f"\n❌ Error: Could not find '{CAPTION_SCRIPT_NAME}' in the {current_dir} folder.")
        print("Please check the file name and run it manually.")