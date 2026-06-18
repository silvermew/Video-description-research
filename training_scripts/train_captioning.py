import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import sys

# Add parent directory to path to allow running from within 'training_scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- IMPORTS ---
try:
    from models.caption_model import MotionCaptioner
    from models.encoder import MotionEncoder
    from utils.text_utils import build_vocab
    from utils.dataset_caption import MotionCaptionDataset, prepare_caption_pairs
except ImportError as e:
    print("Error importing modules. Make sure you have 'models' and 'utils' folders with __init__.py files.")
    print(f"Details: {e}")
    sys.exit(1)

# ====================================================================
# 1. CONFIGURATION
# ====================================================================

# Directories
PROCESSED_DATA_DIR = "processed_windows"  
DATA_ROOT = "data"                        
CHECKPOINT_DIR = "checkpoints_caption"    

# Model Paths
BEST_ENCODER_PATH = "checkpoints/ckpt_e60_enc.pth"

# UPGRADED: Training Hyperparameters
BATCH_SIZE = 64        # Increased to utilize 24GB VRAM
NUM_WORKERS = 4        # Load data in parallel
EPOCHS = 100
LEARNING_RATE = 1e-4
LOG_INTERVAL = 10

# Model Architecture
CONTEXT_DIM = 256
NUM_HEADS = 8
NUM_LAYERS_ENCODER = 1

# Text Decoder Params
TEXT_EMBED_DIM = 256
NUM_DECODER_LAYERS = 4


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 2. Build Vocabulary
    annotations_dir = os.path.join(DATA_ROOT, "annotation_rewritten")
    if not os.path.exists(annotations_dir):
        print(f"Error: Annotations folder not found at {annotations_dir}")
        sys.exit(1)
        
    vocab = build_vocab(annotations_dir)
    pad_idx = vocab.word2idx.get('<pad>', 0)
    print(f"Vocabulary built. Size: {len(vocab)} words.")

    # 3. Prepare Data Pairs
    pairs = prepare_caption_pairs(PROCESSED_DATA_DIR, DATA_ROOT)
    if not pairs:
        print("Error: No text-motion pairs found! Check your folder structure and file names.")
        sys.exit(1)
        
    # UPGRADED: Train/Validation Split (90% Train / 10% Val)
    full_dataset = MotionCaptionDataset(pairs, vocab)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # UPGRADED: DataLoaders with parallel workers and pin_memory
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"Data loaded. Training: {len(train_dataset)} | Validation: {len(val_dataset)}")

    # 4. Initialize Models
    print("Initializing models...")
    motion_encoder = MotionEncoder(
        hidden_dim=CONTEXT_DIM, 
        num_heads=NUM_HEADS, 
        num_layers=NUM_LAYERS_ENCODER
    ).to(device)
    
    if os.path.exists(BEST_ENCODER_PATH):
        motion_encoder.load_state_dict(torch.load(BEST_ENCODER_PATH, map_location=device))
        print(f"Loaded pre-trained encoder from {BEST_ENCODER_PATH}")
    else:
        print(f"Warning: Pre-trained encoder not found. Training from scratch.")

    model = MotionCaptioner(
        motion_encoder=motion_encoder,
        vocab_size=len(vocab),
        text_embed_dim=TEXT_EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_DECODER_LAYERS
    ).to(device)

    # 5. UPGRADED: Optimizer, Loss, and Scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 6. Training Loop with Validation
    print(f"\n--- Starting Caption Training for {EPOCHS} Epochs ---")
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            motions = batch["motion"].to(device, non_blocking=True)
            texts = batch["text"].to(device, non_blocking=True)
            
            decoder_input = texts[:, :-1]
            target = texts[:, 1:]
            
            seq_len = decoder_input.size(1)
            tgt_mask = generate_square_subsequent_mask(seq_len).to(device)
            
            predictions = model(motions, decoder_input, tgt_mask)
            
            loss = criterion(
                predictions.reshape(-1, len(vocab)), 
                target.reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            
            # UPGRADED: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
            
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Train Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                motions = batch["motion"].to(device, non_blocking=True)
                texts = batch["text"].to(device, non_blocking=True)
                
                decoder_input = texts[:, :-1]
                target = texts[:, 1:]
                
                seq_len = decoder_input.size(1)
                tgt_mask = generate_square_subsequent_mask(seq_len).to(device)
                
                predictions = model(motions, decoder_input, tgt_mask)
                
                loss = criterion(
                    predictions.reshape(-1, len(vocab)), 
                    target.reshape(-1)
                )
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Step the Learning Rate Scheduler
        scheduler.step()
        
        print(f"--- Epoch {epoch+1} Complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ---")
        
        # UPGRADED: Save only if Validation Loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "caption_model_best_rewritten2.pth")
            torch.save(model.state_dict(), best_path)
            print(f"🌟 New best model saved! (Val Loss: {best_val_loss:.4f})")

    # Final Save
    final_path = os.path.join(CHECKPOINT_DIR, "caption_model_final_rewritten2.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining Complete. Final model saved to {final_path}")
    print(f"Best model based on validation is located at: {os.path.join(CHECKPOINT_DIR, 'caption_model_best_rewritten2.pth')}")

if __name__ == "__main__":
    main()