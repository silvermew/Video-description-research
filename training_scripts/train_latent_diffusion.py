import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path to allow running from within 'training_scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your models
from models import MotionVQVAE, LatentDiffusion1D, TextEncoder
from utils.scheduler import NoiseScheduler
from utils.dataset_caption import MotionCaptionDataset, prepare_caption_pairs
from utils.text_utils import build_vocab

# --- CONFIG ---
DATA_ROOT = "data" 
PROCESSED_WINDOWS_DIR = "processed_windows" 
CHECKPOINT_DIR = "checkpoints_latent_1d" 
VQVAE_WEIGHTS = "checkpoints_vqvae/best_vqvae.pth"
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 200

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. Load Data
    print("Building vocabulary...")
    vocab = build_vocab(DATA_ROOT)
    print(f"Vocabulary size: {len(vocab)}")

    data_pairs = prepare_caption_pairs(PROCESSED_WINDOWS_DIR, DATA_ROOT, window_size=60)
    
    if len(data_pairs) == 0:
        print("❌ Error: No matching text-motion pairs found! Check your folder paths.")
        return

    dataset = MotionCaptionDataset(data_pairs, vocab)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 2. Load VQ-VAE (FROZEN)
    vqvae = MotionVQVAE(input_dim=192, latent_dim=256, num_tokens=512).to(device)
    
    if os.path.exists(VQVAE_WEIGHTS):
        print(f"Loading VQ-VAE weights from {VQVAE_WEIGHTS}")
        vqvae.load_state_dict(torch.load(VQVAE_WEIGHTS, map_location=device))
    else:
        print(f"❌ Error: VQ-VAE weights not found at {VQVAE_WEIGHTS}. Did you finish Phase 1?")
        return
        
    vqvae.eval() 
    for param in vqvae.parameters():
        param.requires_grad = False 

    # 3. Setup Models to Train
    text_encoder = TextEncoder(vocab_size=len(vocab), embed_dim=256).to(device)
    
    # Initialize the 1D Latent Diffusion Model natively
    diffusion = LatentDiffusion1D(
        in_channels=256, 
        model_channels=256, 
        context_dim=256,
        num_blocks=4
    ).to(device)

    optimizer = optim.AdamW(
        list(text_encoder.parameters()) + list(diffusion.parameters()), 
        lr=LR
    )
    scheduler = NoiseScheduler(num_train_timesteps=1000)
    criterion = nn.MSELoss()

    print(f"--- Starting 1D Latent Diffusion Training on {device} ---")

    for epoch in range(EPOCHS):
        text_encoder.train()
        diffusion.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            motions = batch['motion'].to(device) 
            text_idx = batch['text'].to(device)
            
            optimizer.zero_grad()

            # --- STEP A: GET LATENTS ---
            with torch.no_grad():
                b, t, m, c = motions.shape
                motions_flat = motions.view(b, t, m*c).permute(0, 2, 1) 
                
                latents = vqvae.encoder(motions_flat)
                latents = vqvae.pre_vq_conv(latents)
                
                # Quantize BEFORE training the diffusion model!
                quantization_result = vqvae.vq(latents)
                
                if isinstance(quantization_result, tuple):
                    for item in quantization_result:
                        if isinstance(item, torch.Tensor) and item.dim() == 3:
                            latents_quantized = item
                            break
                else:
                    latents_quantized = quantization_result
                    
                # Scale the clean, QUANTIZED latents!
                latents = latents_quantized * 1.41525

            # --- STEP B: GET TEXT CONTEXT ---
            # 10% Classifier-Free Guidance Dropout
            if torch.rand(1).item() < 0.1:
                text_idx = torch.zeros_like(text_idx) # Feed it empty text!
                
            text_context = text_encoder(text_idx)

            # --- STEP C: DIFFUSION FORWARD PASS ---
            latents_4d = latents.unsqueeze(-1)
            noise_4d = torch.randn_like(latents_4d)
            timesteps = scheduler.sample_timesteps(latents.shape[0]).to(device)
            noisy_latents_4d = scheduler.add_noise(latents_4d, noise_4d, timesteps)
            
            noisy_latents = noisy_latents_4d.squeeze(-1)
            noise = noise_4d.squeeze(-1)

            pred_noise = diffusion(noisy_latents, timesteps, text_context)
            loss = criterion(pred_noise, noise)

            # --- STEP D: OPTIMIZE WITH GRADIENT CLIPPING ---
            loss.backward()
            
            # Clip gradients to prevent explosions
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Latent Diffusion Loss: {avg_loss:.6f}")

        # Save weights every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.save(text_encoder.state_dict(), f"{CHECKPOINT_DIR}/text_enc_{epoch+1}.pth")
            torch.save(diffusion.state_dict(), f"{CHECKPOINT_DIR}/latent_diff_{epoch+1}.pth")

    print("Latent Diffusion Training Complete!")

if __name__ == "__main__":
    main()