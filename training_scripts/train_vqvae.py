import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # <--- Progress Bar

# Add parent directory to path to allow running from within 'training_scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# IMPORT YOUR MODULE
from models import MotionVQVAE

# --- CONFIGURATION ---
DATA_DIR = "processed_windows" 
CHECKPOINT_DIR = "checkpoints_vqvae"
BATCH_SIZE = 32
LR = 2e-4
EPOCHS = 200

# --- SIMPLIFIED DATASET ---
class PreProcessedDataset(Dataset):
    def __init__(self, data_root):
        self.files = glob.glob(os.path.join(data_root, "**/*.npy"), recursive=True)
        if len(self.files) == 0:
            print(f"[WARNING] No .npy files found in {data_root}")
        else:
            print(f"[INFO] Found {len(self.files)} pre-processed windows.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        motion_window = np.load(self.files[idx])
        return torch.tensor(motion_window, dtype=torch.float32)

# --- HELPER: PLOT LOSS ---
def plot_training_loss(recon_losses, vq_losses, epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(recon_losses, label='Reconstruction Loss')
    plt.plot(vq_losses, label='VQ Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training Progress (Epoch {epoch})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CHECKPOINT_DIR, "training_loss.png"))
    plt.close()

# --- MAIN TRAINING LOOP ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 1. Data Setup
    dataset = PreProcessedDataset(DATA_DIR)
    if len(dataset) == 0:
        print("❌ Error: No data found.")
        return
        
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 2. Model Setup
    model = MotionVQVAE(input_dim=192, latent_dim=256, num_tokens=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Trackers for plotting
    epoch_recon_losses = []
    epoch_vq_losses = []

    print(f"--- Starting VQ-VAE Training on {device} ---")
    
    for epoch in range(EPOCHS):
        model.train()
        total_recon_loss = 0
        total_vq_loss = 0
        
        # WRAP LOADER WITH TQDM for Progress Bar
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        
        for batch in progress_bar:
            x = batch.to(device) 
            
            optimizer.zero_grad()
            
            # Forward Pass
            vq_loss, x_recon, perplexity = model(x)
            recon_loss = nn.functional.mse_loss(x_recon, x)
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            
            # Update the progress bar description with current loss
            progress_bar.set_postfix({
                "Recon": f"{recon_loss.item():.4f}", 
                "VQ": f"{vq_loss.item():.4f}"
            })
            
        # Logging Average per Epoch
        avg_recon = total_recon_loss / len(loader)
        avg_vq = total_vq_loss / len(loader)
        
        epoch_recon_losses.append(avg_recon)
        epoch_vq_losses.append(avg_vq)
        
        # Save Loss Plot
        plot_training_loss(epoch_recon_losses, epoch_vq_losses, epoch+1)
        
        # Save Checkpoint
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"vqvae_epoch_{epoch+1}.pth"))

    # Save Final
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_vqvae.pth"))
    print("Training Complete. Loss graph saved to checkpoints_vqvae/training_loss.png")

if __name__ == "__main__":
    main()