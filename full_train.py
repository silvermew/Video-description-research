import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- IMPORTS ---
try:
    from models.caption_model import MotionCaptioner
    from models.encoder import MotionEncoder
    from utils.text_utils import build_vocab
    from utils.dataset_caption import MotionCaptionDataset, prepare_caption_pairs
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# ====================================================================
# 1. GLOBAL CONFIGURATION
# ====================================================================
PROCESSED_DATA_DIR = "processed_windows"  
DATA_ROOT = "data"                        
CHECKPOINT_DIR = "checkpoints_caption"    
BEST_ENCODER_PATH = "checkpoints/ckpt_e60_enc.pth" 

# Hardware & Data
BATCH_SIZE = 64
NUM_WORKERS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Phases
OPTUNA_TRIALS = 20       # How many different architectures to test
OPTUNA_EPOCHS = 10       # How long to test each one
FINAL_EPOCHS = 100       # How long to train the final winner

# Fixed Architecture
NUM_HEADS = 8
NUM_LAYERS_ENCODER = 1

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

# ====================================================================
# PHASE 1: OPTUNA SEARCH OBJECTIVE
# ====================================================================
def objective(trial, train_loader, val_loader, vocab, pad_idx):
    
    # 1. Optuna suggests parameters
    context_dim = trial.suggest_categorical("context_dim", [256, 512, 768])
    num_decoder_layers = trial.suggest_int("num_decoder_layers", 2, 8)
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    
    # 2. Build Model
    motion_encoder = MotionEncoder(hidden_dim=context_dim, num_heads=NUM_HEADS, num_layers=NUM_LAYERS_ENCODER).to(device)
    model = MotionCaptioner(motion_encoder, len(vocab), context_dim, NUM_HEADS, num_decoder_layers).to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)

    # 3. Mini-Training Loop
    for epoch in range(OPTUNA_EPOCHS):
        model.train()
        for batch in train_loader:
            motions, texts = batch["motion"].to(device, non_blocking=True), batch["text"].to(device, non_blocking=True)
            dec_in, target = texts[:, :-1], texts[:, 1:]
            
            mask = generate_square_subsequent_mask(dec_in.size(1)).to(device)
            preds = model(motions, dec_in, mask)
            
            loss = criterion(preds.reshape(-1, len(vocab)), target.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                motions, texts = batch["motion"].to(device, non_blocking=True), batch["text"].to(device, non_blocking=True)
                dec_in, target = texts[:, :-1], texts[:, 1:]
                mask = generate_square_subsequent_mask(dec_in.size(1)).to(device)
                preds = model(motions, dec_in, mask)
                val_loss += criterion(preds.reshape(-1, len(vocab)), target.reshape(-1)).item()
                
        avg_val_loss = val_loss / len(val_loader)
        trial.report(avg_val_loss, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()

    return avg_val_loss

# ====================================================================
# PHASE 2: FINAL TRAINING LOOP
# ====================================================================
def train_final_model(best_params, train_loader, val_loader, vocab, pad_idx):
    print(f"\n🚀 STARTING FINAL {FINAL_EPOCHS}-EPOCH TRAINING RUN 🚀")
    print(f"Using Optimal Parameters: {best_params}")
    
    # 1. Build Final Model
    motion_encoder = MotionEncoder(
        hidden_dim=best_params['context_dim'], 
        num_heads=NUM_HEADS, 
        num_layers=NUM_LAYERS_ENCODER
    ).to(device)
    
    # Smart Loading: Only load pre-trained weights if the dimension matches!
    if best_params['context_dim'] == 256 and os.path.exists(BEST_ENCODER_PATH):
        motion_encoder.load_state_dict(torch.load(BEST_ENCODER_PATH, map_location=device))
        print("✅ Restored pre-trained encoder weights (Dim 256 matches).")
    else:
        print("⚠️ Training Encoder from scratch (Dimensions changed or weights not found).")

    model = MotionCaptioner(
        motion_encoder=motion_encoder,
        vocab_size=len(vocab),
        text_embed_dim=best_params['context_dim'],
        num_heads=NUM_HEADS,
        num_layers=best_params['num_decoder_layers']
    ).to(device)

   # 2. Setup Optimizer & Scheduler (WITH DIFFERENTIAL LEARNING RATES)
    
    # Isolate the pre-trained encoder parameters
    encoder_params = list(model.motion_encoder.parameters())
    
    # Isolate the rest of the model (Decoder, Embeddings, Output Layers)
    # We do this by grabbing everything that DOES NOT have 'motion_encoder' in its name
    decoder_params = [p for n, p in model.named_parameters() if 'motion_encoder' not in n]

    # Create the grouped optimizer
    optimizer = torch.optim.AdamW([
        # Group 1: The Encoder (Microscopic adjustments to prevent forgetting)
        {'params': encoder_params, 'lr': 1e-6},
        
        # Group 2: The Decoder (Normal learning rate suggested by Optuna)
        {'params': decoder_params, 'lr': best_params['lr']} 
    ], weight_decay=1e-4)
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    
    # The Scheduler will now automatically scale BOTH learning rates down proportionally
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)

    # 3. Full Training Loop
    best_val_loss = float('inf')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    for epoch in range(FINAL_EPOCHS):
        model.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            motions, texts = batch["motion"].to(device, non_blocking=True), batch["text"].to(device, non_blocking=True)
            dec_in, target = texts[:, :-1], texts[:, 1:]
            
            mask = generate_square_subsequent_mask(dec_in.size(1)).to(device)
            preds = model(motions, dec_in, mask)
            
            loss = criterion(preds.reshape(-1, len(vocab)), target.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                motions, texts = batch["motion"].to(device, non_blocking=True), batch["text"].to(device, non_blocking=True)
                dec_in, target = texts[:, :-1], texts[:, 1:]
                mask = generate_square_subsequent_mask(dec_in.size(1)).to(device)
                preds = model(motions, dec_in, mask)
                total_val_loss += criterion(preds.reshape(-1, len(vocab)), target.reshape(-1)).item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{FINAL_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save Best Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "caption_model_ULTIMATE_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"   🌟 New Best Model Saved! (Val: {best_val_loss:.4f})")

    print(f"\n🎉 ALL DONE! The mathematically optimal model is waiting at: {best_path}")

# ====================================================================
# SYSTEM INITIALIZATION & RUN
# ====================================================================
def main():
    print("Loading Data & Building Vocabulary...")
    vocab = build_vocab(os.path.join(DATA_ROOT, "annotation_rewritten"))
    pad_idx = vocab.word2idx.get('<pad>', 0)
    
    pairs = prepare_caption_pairs(PROCESSED_DATA_DIR, DATA_ROOT)
    full_dataset = MotionCaptionDataset(pairs, vocab)
    
    train_size = int(0.9 * len(full_dataset))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, len(full_dataset)))
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"Data ready. Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # 1. RUN OPTUNA SEARCH
    print("\n--- INITIATING HYPERPARAMETER SEARCH ---")
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, vocab, pad_idx), n_trials=OPTUNA_TRIALS)
    
    print("\n🏆 OPTIMIZATION FINISHED 🏆")
    print(f"Best Validation Loss achieved: {study.best_trial.value:.4f}")
    
    # 2. RUN FULL PIPELINE WITH WINNING PARAMS
    train_final_model(study.best_params, train_loader, val_loader, vocab, pad_idx)

if __name__ == "__main__":
    main()