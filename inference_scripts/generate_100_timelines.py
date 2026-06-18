import os
import sys
import glob
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

TRC_DATA_DIR = "data/motionsdata"
OUTPUT_ROOT = "results/motion_to_text_ULTIMATE"
NUM_TO_PROCESS = 100

ANNOTATION_ROOT = "data/annotation_rewritten"   
CAPTION_MODEL_PATH = "checkpoints_caption/caption_model_ULTIMATE_best.pth"

CONTEXT_DIM = 768
NUM_HEADS = 8
NUM_LAYERS_ENCODER = 1
TEXT_EMBED_DIM = 768
NUM_DECODER_LAYERS = 7

WINDOW_SIZE = 60
OVERLAP = 0  
TOP_K = 3

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def clean_sentence(sentence):
    return sentence.strip()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Vocab
    vocab = build_vocab(ANNOTATION_ROOT)
    start_idx = vocab.word2idx.get('<start>', 1)
    end_idx = vocab.word2idx.get('<end>', 2)
    unk_idx = vocab.word2idx.get('<unk>', 3)
    pad_idx = vocab.word2idx.get('<pad>', 0)
    
    # 2. Load Model ONCE
    print("Loading Model...")
    motion_encoder = MotionEncoder(CONTEXT_DIM, NUM_HEADS, NUM_LAYERS_ENCODER).to(device)
    model = MotionCaptioner(motion_encoder, len(vocab), TEXT_EMBED_DIM, NUM_HEADS, NUM_DECODER_LAYERS).to(device)
    
    if os.path.exists(CAPTION_MODEL_PATH):
        model.load_state_dict(torch.load(CAPTION_MODEL_PATH, map_location=device))
        print("✅ Model loaded successfully.")
    else:
        print(f"❌ Model checkpoint not found at {CAPTION_MODEL_PATH}!")
        sys.exit(1)

    model.eval()

    # 3. Find all TRC files
    # This looks for files like "data/motionsdata/20130621/cap-001.trc"
    trc_files = glob.glob(os.path.join(TRC_DATA_DIR, "*", "*.trc"))
    trc_files = sorted(trc_files)[:NUM_TO_PROCESS]
    
    print(f"\nFound {len(trc_files)} files. Starting batch generation...")

    # 4. Loop through files
    for count, trc_path in enumerate(trc_files, 1):
        # Extract folder (date) and filename (cap-xxx)
        path_parts = trc_path.split(os.sep)
        date_folder = path_parts[-2]                # e.g., "20130621"
        file_name = path_parts[-1].replace('.trc', '') # e.g., "cap-001"
        
        # Define output path
        out_dir = os.path.join(OUTPUT_ROOT, date_folder)
        os.makedirs(out_dir, exist_ok=True)
        out_txt_path = os.path.join(out_dir, f"{file_name}.txt")
        
        print(f"[{count}/{len(trc_files)}] Processing {date_folder}/{file_name}.trc ...")
        
        try:
            windows_tensor, _ = load_data_from_trc(
                file_path=trc_path, 
                target_fps=30.0, 
                window_size=WINDOW_SIZE, 
                right_asi_idx=0, left_asi_idx=0, sacrum_idx=0, 
                overlap=OVERLAP
            )
        except Exception as e:
            print(f"  ⚠️ Skipping file due to load error: {e}")
            continue
            
        num_windows = windows_tensor.shape[0]

        # 5. Run Inference and Save to CSV format
        with open(out_txt_path, 'w', encoding='utf-8') as f:
            f.write("Frame,Rank,Probability,Sentence\n")
            
            with torch.no_grad():
                for w_idx in range(num_windows):
                    start_frame = w_idx * (WINDOW_SIZE - OVERLAP)
                    center_frame = start_frame + (WINDOW_SIZE // 2)
                    
                    real_motion = windows_tensor[w_idx].unsqueeze(0).to(device)
                    
                    context_4d = model.motion_encoder(real_motion)
                    context = context_4d.mean(dim=2) 
                    
                    # Initialize with a single empty beam
                    beams = [{'tokens': [start_idx], 'log_prob': 0.0, 'ended': False}]
                    
                    # True Beam Search Loop
                    import math
                    for _ in range(20): # Max sequence length
                        new_beams = []
                        for beam in beams:
                            if beam['ended']:
                                new_beams.append(beam)
                                continue
                                
                            curr_tgt = torch.LongTensor([beam['tokens']]).to(device)
                            seq_len = curr_tgt.shape[1]
                            mask = generate_square_subsequent_mask(seq_len).to(device)
                            
                            text_emb = model.embedding(curr_tgt) + model.text_pos_embedding[:, :seq_len, :]
                            out = model.decoder(tgt=text_emb, memory=context, tgt_mask=mask)
                            logits = model.fc_out(out[:, -1, :])
                            
                            # Ban <unk> and <pad>
                            logits[0, unk_idx] = -float('Inf')
                            logits[0, pad_idx] = -float('Inf')
                            
                            # Use log_softmax to prevent 0.0000 probability collapse
                            log_probs = torch.log_softmax(logits, dim=-1)
                            top_log_probs, top_indices = torch.topk(log_probs, TOP_K, dim=-1)
                            
                            for k in range(TOP_K):
                                token = top_indices[0][k].item()
                                log_p = top_log_probs[0][k].item()
                                
                                new_beam = {
                                    'tokens': beam['tokens'] + [token],
                                    'log_prob': beam['log_prob'] + log_p,
                                    'ended': (token == end_idx)
                                }
                                new_beams.append(new_beam)
                                
                        # Sort by length-normalized log probability to prevent favoring tiny sentences
                        new_beams.sort(key=lambda x: x['log_prob'] / len(x['tokens']), reverse=True)
                        
                        # Keep only the absolute best TOP_K branches overall
                        beams = new_beams[:TOP_K] 
                        
                        # Stop early if all our top beams have generated the <end> token
                        if all(b['ended'] for b in beams):
                            break

                    # Sort and write to file
                    for rank, branch in enumerate(beams, 1):
                        words = [vocab.idx2word[idx] for idx in branch['tokens']]
                        
                        if words[0] == '<start>': words = words[1:]
                        if words[-1] == '<end>': words = words[:-1]
                        
                        final_sentence = clean_sentence(" ".join(words))
                        
                        # Convert log_prob back to a human-readable 0.0 to 1.0 confidence score
                        normalized_confidence = math.exp(branch['log_prob'] / len(branch['tokens']))
                        
                        f.write(f"{center_frame},{rank},{normalized_confidence:.4f},{final_sentence}\n")

    print(f"\n✅ All 100 files processed and saved to '{OUTPUT_ROOT}'")

if __name__ == "__main__":
    main()