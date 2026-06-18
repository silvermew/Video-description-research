import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import MotionVQVAE, LatentDiffusion1D, TextEncoder
from utils.text_utils import build_vocab, text_to_indices
from utils.scheduler import NoiseScheduler
from utils.data_utils import save_data_to_trc, MARKER_NAMES_64

# --- CONFIGURATION ---
DATA_ROOT = "data" 
VQVAE_WEIGHTS = "checkpoints_vqvae/best_vqvae.pth"

# [UPDATE] Pointing to your REAL training folder and final epoch!
# (Change the '200' if you decide to stop training early or train longer)
TEXT_ENC_WEIGHTS = "checkpoints_latent_1d/text_enc_200.pth"
DIFFUSION_WEIGHTS = "checkpoints_latent_1d/latent_diff_200.pth"

OUTPUT_DIR = "results/text_to_motion"
MAX_SEQ_LEN = 20

def scheduler_step_math(scheduler, model_output, timestep, sample):
    """Stable HuggingFace 'Predict X0 and Clamp' Math"""
    t = timestep
    device = sample.device
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    
    alpha_prod_t = alphas_cumprod[t]
    alpha_prod_t_prev = alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=device)
    
    beta_prod_t = 1.0 - alpha_prod_t
    beta_prod_t_prev = 1.0 - alpha_prod_t_prev
    alpha_t = alpha_prod_t / alpha_prod_t_prev
    beta_t = 1.0 - alpha_t
    
    # 1. Predict clean latent and CLAMP to prevent mathematical explosion
    pred_original_sample = (sample - torch.sqrt(beta_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
    pred_original_sample = torch.clamp(pred_original_sample, -4.0, 4.0)
    
    # 2. Compute the stable mean
    pred_original_sample_coeff = (torch.sqrt(alpha_prod_t_prev) * beta_t) / beta_prod_t
    current_sample_coeff = torch.sqrt(alpha_t) * beta_prod_t_prev / beta_prod_t
    pred_prev_mean = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
    
    # 3. Add reverse noise
    if t > 0:
        variance = (beta_prod_t_prev / beta_prod_t) * beta_t
        variance = torch.clamp(variance, min=1e-20)
        noise = torch.randn_like(sample)
        pred_prev_sample = pred_prev_mean + torch.sqrt(variance) * noise
    else:
        pred_prev_sample = pred_prev_mean
        
    return pred_prev_sample

def generate_motion(prompt, device):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"1. Loading Vocabulary...")
    vocab = build_vocab(DATA_ROOT)
    
    print("2. Loading Models...")
    vqvae = MotionVQVAE(input_dim=192, latent_dim=256, num_tokens=512).to(device)
    vqvae.load_state_dict(torch.load(VQVAE_WEIGHTS, map_location=device))
    vqvae.eval()

    text_encoder = TextEncoder(vocab_size=len(vocab), embed_dim=256).to(device)
    text_encoder.load_state_dict(torch.load(TEXT_ENC_WEIGHTS, map_location=device))
    text_encoder.eval()

    diffusion = LatentDiffusion1D(
        in_channels=256, 
        model_channels=256, 
        context_dim=256,
        num_blocks=4
    ).to(device)
    diffusion.load_state_dict(torch.load(DIFFUSION_WEIGHTS, map_location=device))
    diffusion.eval()

    scheduler = NoiseScheduler(num_train_timesteps=1000)
    
    print(f"\n--- Generating Motion for: '{prompt}' ---")
    with torch.no_grad():
        # Encode the text prompt
        text_idx = text_to_indices(prompt, vocab, MAX_SEQ_LEN).unsqueeze(0).to(device)
        text_context = text_encoder(text_idx) 

        # Start with pure random noise (Batch=1, Channels=256, Frames=15)
        latents = torch.randn((1, 256, 15), device=device)

        print("Denoising Latents...")
        for t in tqdm(reversed(range(0, scheduler.num_train_timesteps)), total=scheduler.num_train_timesteps):
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            
            # Predict noise natively in 1D
            pred_noise = diffusion(latents, t_tensor, text_context)
            
            # Use the stable explicit math step
            latents = scheduler_step_math(scheduler, pred_noise, t, latents)

        print("Decoding to 3D Skeleton...")
        
        # 1. Reverse your custom magic number
        latents = latents / 1.41525
        
        # 2. Snap to the discrete codebook grid, safely extracting the 3D tensor
        quantization_result = vqvae.vq(latents)
        if isinstance(quantization_result, tuple):
            for item in quantization_result:
                if isinstance(item, torch.Tensor) and item.dim() == 3:
                    latents_quantized = item
                    break
        else:
            latents_quantized = quantization_result
        
        # 3. Decode from the clean, QUANTIZED latents
        motion_flat = vqvae.decoder(latents_quantized) # (1, 192, 60)
        
        # Reshape to (Batch, Time, Markers, Coords) -> (1, 60, 64, 3)
        motion = motion_flat.permute(0, 2, 1).view(1, 60, 64, 3)
        
        # Convert to numpy
        motion_np = motion.cpu().numpy()[0] 
        
        # Sanitize prompt for filename
        safe_name = prompt.replace(" ", "_").replace("/", "")
        npy_path = os.path.join(OUTPUT_DIR, f"{safe_name}.npy")
        trc_path = os.path.join(OUTPUT_DIR, f"{safe_name}.trc")
        
        np.save(npy_path, motion_np)
        
        # --- RESCALE AND LIFT ---
        metadata = {
            'target_fps': 30.0,
            'norm_mean': 0.0,     
            'norm_std': 450.0,    
            'marker_names': MARKER_NAMES_64  
        }
        
        root_traj = np.zeros((motion_np.shape[0], 1, 3))
        root_traj[:, 0, 1] = 950.0  
        
        # Pass the safe numpy array to the writer
        save_data_to_trc(motion_np, metadata, trc_path, insert_root_trajectory=root_traj)
        
        print(f"✅ Success! Motion saved to:")
        print(f"   - {npy_path}")
        print(f"   - {trc_path}")
        
        return trc_path

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # You can change this to test whatever prompt you want!
    my_prompt = "a person is jumping" 
    
    generated_file = generate_motion(my_prompt, device)
    print(f"\nNext step: Use your visualize script to view the generated results!")