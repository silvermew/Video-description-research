import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- IMPORTS ---
try:
    from models.caption_model import MotionCaptioner
    from models.encoder import MotionEncoder
    from utils.text_utils import build_vocab
    from utils.data_utils import load_data_from_trc 
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)
    
# ====================================================================
# 1. CONFIGURATION
# ====================================================================

TARGET_TRC_FILE = "data/motionsdata/20150306/cap-005.trc"
OUTPUT_TXT_FILE = "results/motion_to_text/20150306/cap-005.txt"

ANNOTATION_ROOT = "data/annotation_ai_clean"       
CAPTION_MODEL_PATH = "checkpoints_caption/caption_model_final.pth"

CONTEXT_DIM = 256
NUM_HEADS = 8
NUM_LAYERS_ENCODER = 1
TEXT_EMBED_DIM = 256
NUM_DECODER_LAYERS = 4

WINDOW_SIZE = 60
OVERLAP = 0  
TOP_K = 3

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Vocab
    vocab = build_vocab(ANNOTATION_ROOT)
    start_idx = vocab.word2idx.get('<start>', 1)
    end_idx = vocab.word2idx.get('<end>', 2)
    unk_idx = vocab.word2idx.get('<unk>', 3)
    pad_idx = vocab.word2idx.get('<pad>', 0)
    
    # 2. Load Model
    motion_encoder = MotionEncoder(CONTEXT_DIM, NUM_HEADS, NUM_LAYERS_ENCODER).to(device)
    model = MotionCaptioner(motion_encoder, len(vocab), TEXT_EMBED_DIM, NUM_HEADS, NUM_DECODER_LAYERS).to(device)
    
    if os.path.exists(CAPTION_MODEL_PATH):
        model.load_state_dict(torch.load(CAPTION_MODEL_PATH, map_location=device))
        print("✅ Model loaded successfully.")
    else:
        print(f"❌ Model checkpoint not found at {CAPTION_MODEL_PATH}!")
        sys.exit(1)

    model.eval()

    # 3. Load Full TRC
    print(f"\nProcessing Full Motion File: {TARGET_TRC_FILE}")
    windows_tensor, metadata = load_data_from_trc(
        file_path=TARGET_TRC_FILE, 
        target_fps=30.0, 
        window_size=WINDOW_SIZE, 
        right_asi_idx=0, left_asi_idx=0, sacrum_idx=0, 
        overlap=OVERLAP
    )
    num_windows = windows_tensor.shape[0]

    os.makedirs(os.path.dirname(OUTPUT_TXT_FILE), exist_ok=True)
    print(f"\nGenerating Timeline...")
    
    # 4. Run Inference
    with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as f:
        f.write("Frame,Rank,Probability,Sentence\n")
        
        with torch.no_grad():
            for w_idx in range(num_windows):
                start_frame = w_idx * (WINDOW_SIZE - OVERLAP)
                center_frame = start_frame + (WINDOW_SIZE // 2)
                
                real_motion = windows_tensor[w_idx].unsqueeze(0).to(device)
                
                context_4d = model.motion_encoder(real_motion)
                context = context_4d.mean(dim=2) 
                
                # --- Step 1: Branch into Top-K starting words ---
                tgt_tensor = torch.LongTensor([[start_idx]]).to(device)
                mask = generate_square_subsequent_mask(1).to(device)
                
                text_emb = model.embedding(tgt_tensor) + model.text_pos_embedding[:, :1, :]
                output = model.decoder(tgt=text_emb, memory=context, tgt_mask=mask)
                logits = model.fc_out(output[:, -1, :])
                
                # [CRITICAL FIX] Ban the AI from predicting <unk> or <pad>
                logits[0, unk_idx] = -float('Inf')
                logits[0, pad_idx] = -float('Inf')
                
                probs = torch.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, TOP_K, dim=-1)
                
                branches = []
                for k in range(TOP_K):
                    token = top_indices[0][k].item()
                    prob = top_probs[0][k].item()
                    branches.append({
                        'tokens': [start_idx, token],
                        'prob': prob,
                        'ended': (token == end_idx)
                    })
                
                # --- Step 2: Greedily decode the rest of each branch ---
                for branch in branches:
                    for _ in range(20): # Max length
                        if branch['ended']:
                            break
                            
                        curr_tgt = torch.LongTensor([branch['tokens']]).to(device)
                        seq_len = curr_tgt.shape[1]
                        mask = generate_square_subsequent_mask(seq_len).to(device)
                        
                        text_emb = model.embedding(curr_tgt) + model.text_pos_embedding[:, :seq_len, :]
                        out = model.decoder(tgt=text_emb, memory=context, tgt_mask=mask)
                        next_logits = model.fc_out(out[:, -1, :])
                        
                        # [CRITICAL FIX] Ban <unk> and <pad> on every step
                        next_logits[0, unk_idx] = -float('Inf')
                        next_logits[0, pad_idx] = -float('Inf')
                        
                        next_probs = torch.softmax(next_logits, dim=-1)
                        best_prob, best_idx = torch.max(next_probs, dim=-1)
                        
                        branch['tokens'].append(best_idx.item())
                        branch['prob'] *= best_prob.item() 
                        
                        if best_idx.item() == end_idx:
                            branch['ended'] = True

                # --- Step 3: Format and write to file ---
                branches.sort(key=lambda x: x['prob'], reverse=True)
                
                for rank, branch in enumerate(branches, 1):
                    words = [vocab.idx2word[idx] for idx in branch['tokens']]
                    
                    if words[0] == '<start>': words = words[1:]
                    if words[-1] == '<end>': words = words[:-1]
                    
                    final_sentence = " ".join(words)
                    f.write(f"{center_frame},{rank},{branch['prob']:.4f},{final_sentence}\n")

    print(f"✅ Finished! Results saved to: {OUTPUT_TXT_FILE}")

if __name__ == "__main__":
    main()