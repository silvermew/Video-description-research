import torch
import torch.nn as nn
import os
import sys
import numpy as np
import math

# Add parent directory to path to allow running from within 'inference_scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- IMPORTS ---
from models.caption_model import MotionCaptioner
from models.encoder import MotionEncoder
from utils.text_utils import build_vocab

# [NEW] Import your specific normalization utils
from utils.data_utils import normalize_and_center_data, parse_trc_robust

# ====================================================================
# 1. CONFIGURATION
# ====================================================================
# Input: Can be .trc (Raw) or .npy (Raw)
INPUT_FILE = "data/motionsdata/20150306/cap-005.trc" 
OUTPUT_FILE = "frame_captions2.txt"

# Paths
ANNOTATION_ROOT = "data/annotation_ai_clean"
CHECKPOINT_PATH = "checkpoints_caption/caption_model_final.pth"

# Preprocessing Params (Must match your training logic)
ROOT_INDEX = 0  # Index of pHipOrigin for centering

# Model Params
CONTEXT_DIM = 256
NUM_HEADS = 8
NUM_LAYERS_ENCODER = 1
TEXT_EMBED_DIM = 256
NUM_DECODER_LAYERS = 4

# Inference Params
WINDOW_SIZE = 60    # 60 frames context
STRIDE = 60          # 1 = Every frame
BEAM_SIZE = 3       # Top 3 sentences
MAX_LEN = 15        # Max sentence length

# ====================================================================
# 2. THE BRIDGE (Safety Wrapper)
# ====================================================================
class CaptioningBridge(nn.Module):
    def __init__(self, base_model, device):
        super().__init__()
        self.base_model = base_model
        self.device = device
        
        # Dynamic Layer Finding
        self.embed_layer = getattr(base_model, 'text_embedding', getattr(base_model, 'embedding', None))
        self.pos_emb_param = getattr(base_model, 'text_pos_embedding', None)
        self.out_layer = getattr(base_model, 'fc_out', getattr(base_model, 'linear', None))
        self.scale = getattr(base_model, 'scale', None)

    def encode_motion(self, motions):
        # Input expected: (Batch, Time, Markers, 3)
        if motions.dim() == 3: 
            # If (Batch, Time, Features) -> Reshape to (Batch, Time, 64, 3)
            b, w, f = motions.shape
            if f == 192: # 64 * 3
                motions = motions.view(b, w, 64, 3)

        if motions.dim() == 4: # (Batch, Time, Markers, 3)
            context = self.base_model.motion_encoder(motions)
            # Squash markers to get (Batch, Time, Hidden) for Decoder
            return context.mean(dim=2)
        
        return None

# ====================================================================
# 3. BEAM SEARCH (With Diversity Filter)
# ====================================================================
def calculate_overlap(sent1, sent2):
    """Calculates how similar two sentences are based on word overlap."""
    # Convert to sets of words (ignoring "a", "the", "he", etc. helps but keep it simple first)
    set1 = set(sent1.split())
    set2 = set(sent2.split())
    
    # Jaccard Similarity: Intersection / Union
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0: return 0.0
    return intersection / union

def beam_search_decode(model, context, vocab, device, beam_size=3, max_len=15):
    # We generate MORE candidates initially so we can filter duplicates later
    SEARCH_SIZE = 20 
    
    start_token = vocab('<start>')
    beam = [(0.0, [start_token])] 
    completed_candidates = []

    for _ in range(max_len):
        candidates = []
        for score, seq in beam:
            if vocab.idx2word[seq[-1]] == '<end>':
                # If finished, add to completed list
                completed_candidates.append((score, seq))
                continue
            
            # Predict next word
            tgt_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
            sz = tgt_tensor.size(1)
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)

            tgt_emb = model.embed_layer(tgt_tensor)
            if model.scale: tgt_emb = tgt_emb * model.scale
            
            seq_len_pos = tgt_emb.size(1)
            if model.pos_emb_param is not None:
                tgt_emb = tgt_emb + model.pos_emb_param[:, :seq_len_pos, :]
            
            output = model.base_model.decoder(tgt=tgt_emb, memory=context, tgt_mask=mask)
            logits = model.out_layer(output[:, -1, :])
            
            log_probs = torch.log_softmax(logits, dim=-1)
            top_log_probs, top_indices = torch.topk(log_probs, SEARCH_SIZE) 
            
            for k in range(len(top_indices[0])):
                token_idx = top_indices[0][k].item()
                word = vocab.idx2word[token_idx]
                
                if word == '<unk>': continue # Skip unk
                
                next_score = score + top_log_probs[0][k].item()
                next_seq = seq + [token_idx]
                candidates.append((next_score, next_seq))
        
        # Sort and keep top SEARCH_SIZE for next step
        ordered = sorted(candidates, key=lambda x: x[0], reverse=True)
        beam = ordered[:SEARCH_SIZE]

    # --- DIVERSITY FILTERING ---
    # Convert all found paths to strings
    all_results = []
    
    # Add any paths still in beam (even if not finished with <end>)
    potential_finalists = completed_candidates + beam 
    
    for score, seq in potential_finalists:
        probability = math.exp(score)
        words = [vocab.idx2word[idx] for idx in seq if vocab.idx2word[idx] not in ['<start>', '<end>', '<pad>', '<unk>']]
        sentence = " ".join(words)
        all_results.append((probability, sentence))

    # Sort by probability (highest first)
    all_results.sort(key=lambda x: x[0], reverse=True)

    # Pick Top 3 Distinct Sentences
    final_top_k = []
    
    for prob, sent in all_results:
        is_duplicate = False
        
        # Check against already picked sentences
        for _, existing_sent in final_top_k:
            # If overlap > 60%, consider it a "duplicate meaning"
            if calculate_overlap(sent, existing_sent) > 0.6: 
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_top_k.append((prob, sent))
        
        if len(final_top_k) >= beam_size:
            break
            
    # Fallback: If filter removed too many, just fill with next best
    if len(final_top_k) < beam_size:
        for prob, sent in all_results:
             # Just add unique strings regardless of similarity to fill the list
             if not any(sent == s for p, s in final_top_k):
                 final_top_k.append((prob, sent))
             if len(final_top_k) >= beam_size: break

    return final_top_k

# ====================================================================
# 4. MAIN
# ====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    vocab = build_vocab(ANNOTATION_ROOT)
    
    # Initialize Models
    motion_encoder = MotionEncoder(hidden_dim=CONTEXT_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS_ENCODER).to(device)
    base_model = MotionCaptioner(motion_encoder, len(vocab), TEXT_EMBED_DIM, NUM_HEADS, NUM_DECODER_LAYERS).to(device)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from {CHECKPOINT_PATH}")
        base_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device), strict=False)
    else:
        print("Model checkpoint not found.")
        sys.exit(1)

    model = CaptioningBridge(base_model, device).to(device)
    model.eval()

    # --- LOAD & NORMALIZE DATA (Using your util) ---
    print(f"Loading motion data from {INPUT_FILE}...")
    try:
        # 1. Parse Data
        if INPUT_FILE.endswith('.trc'):
            raw_data = parse_trc_robust(INPUT_FILE)
            print("Parsed TRC file.")
        else:
            # Assume .npy is RAW data (un-normalized)
            raw_data = np.load(INPUT_FILE)
            print("Loaded NPY file.")

        # 2. Normalize and Center (CRITICAL STEP)
        # This aligns the data to match the training distribution exactly
        print("Applying normalization...")
        processed_data, stats = normalize_and_center_data(raw_data, root_idx=ROOT_INDEX)
        
        # 3. Convert to Tensor
        motion_sequence = torch.from_numpy(processed_data).float()
        print(f"Data ready. Shape: {motion_sequence.shape}") # Should be (Time, 64, 3)

    except Exception as e:
        print(f"Could not load or process file: {e}")
        sys.exit(1)

    num_frames = motion_sequence.shape[0]
    results_buffer = []

    print(f"\nProcessing {num_frames} frames...")
    print("-" * 60)

    for i in range(0, num_frames - WINDOW_SIZE + 1, STRIDE):
        # Window: (60, 64, 3) -> Add Batch -> (1, 60, 64, 3)
        window = motion_sequence[i : i + WINDOW_SIZE].unsqueeze(0).to(device) 
        
        with torch.no_grad():
            context = model.encode_motion(window)
            top_sentences = beam_search_decode(model, context, vocab, device, beam_size=BEAM_SIZE, max_len=MAX_LEN)
            
            center_frame = i + (WINDOW_SIZE // 2)
            
            results_buffer.append({
                "frame": center_frame,
                "captions": top_sentences
            })

            print(f"Frame {center_frame:03d}:")
            for rank, (prob, text) in enumerate(top_sentences):
                print(f"   {rank+1}. [{prob:.1%}] {text}")

    # Save Results
    with open(OUTPUT_FILE, "w") as f:
        f.write("Frame,Rank,Probability,Sentence\n")
        for res in results_buffer:
            frame = res['frame']
            for rank, (prob, text) in enumerate(res['captions']):
                f.write(f"{frame},{rank+1},{prob:.4f},{text}\n")
    
    print("-" * 60)
    print(f"Results saved to {OUTPUT_FILE}")
    print("Done.")

if __name__ == "__main__":
    main()