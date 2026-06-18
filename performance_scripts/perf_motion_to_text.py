import os
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
GENERATED_DIR = "results/motion_to_text_ULTIMATE"
ANNOTATION_DIR = "data/annotation_rewritten" 
CHECK_TOP_K = 3

def main():
    print("1. Loading Semantic Evaluation Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    print(f"\n2. Scanning for files and matching frames (Evaluating Top-{CHECK_TOP_K})...")
    gen_files = glob.glob(os.path.join(GENERATED_DIR, "*", "*.txt"))
    
    all_generated_candidates = [] 
    all_ground_truth_captions = []
    
    files_processed = 0
    pairs_found = 0

    for gen_path in gen_files:
        path_parts = gen_path.split(os.sep)
        date_folder = path_parts[-2]
        gen_filename = path_parts[-1]
        
        ant_filename = gen_filename.replace('cap-', 'annotation-').replace('.txt', '.ant')
        ant_path = os.path.join(ANNOTATION_DIR, date_folder, ant_filename)
        
        if not os.path.exists(ant_path):
            continue
            
        # A. Load Generated Top-K Captions
        gen_dict = {}
        with open(gen_path, 'r', encoding='utf-8') as f:
            next(f) # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    frame_idx = int(parts[0])
                    rank = int(parts[1])
                    
                    # [THE FIX] Safely rejoin any sentences that contained commas!
                    sentence = ",".join(parts[3:]).strip()
                    
                    if rank <= CHECK_TOP_K:
                        if frame_idx not in gen_dict:
                            gen_dict[frame_idx] = {}
                        gen_dict[frame_idx][rank] = sentence
                    
        if not gen_dict:
            continue

        # B. Load Ground Truth and Match using Action Boundaries!
        raw_annotations = []
        with open(ant_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    raw_annotations.append((int(parts[0]), parts[1].strip()))
                    
        if not raw_annotations:
            continue
            
        raw_annotations.sort(key=lambda x: x[0])
        
        # 1. Convert timestamps to Start/End segments
        segments = []
        for i in range(len(raw_annotations)):
            start_frame = raw_annotations[i][0]
            gt_text = raw_annotations[i][1]
            if i + 1 < len(raw_annotations):
                end_frame = raw_annotations[i+1][0]
            else:
                end_frame = float('inf') # Lasts until the end of the video
            segments.append((start_frame, end_frame, gt_text))
            
        # 2. Check which segment each AI generation falls into
        for gen_frame, ranks_dict in gen_dict.items():
            for s_start, s_end, gt_text in segments:
                if s_start <= gen_frame < s_end:
                    all_generated_candidates.append(ranks_dict)
                    all_ground_truth_captions.append(gt_text)
                    pairs_found += 1
                    break # We found the match for this frame, move to the next!
                        
        files_processed += 1

    print(f"✅ Processed {files_processed} files.")
    print(f"✅ Found {pairs_found} temporally matched pairs.")
    
    if pairs_found == 0:
        print("❌ No pairs found! Check your folder paths.")
        return

    print(f"\n3. Calculating Semantic Similarity across all {CHECK_TOP_K} ranks...")
    
    # Get ground truth embeddings
    gt_embeddings = model.encode(all_ground_truth_captions, convert_to_tensor=True, batch_size=128)
    
    all_rank_scores = []
    all_rank_sentences = []
    
    # Evaluate each rank independently
    for k in range(1, CHECK_TOP_K + 1):
        rank_k_sents = [cands.get(k, cands.get(1, "")) for cands in all_generated_candidates]
        rank_k_embs = model.encode(rank_k_sents, convert_to_tensor=True, batch_size=128)
        
        # [THE FIX] Calculate 1-to-1 similarity natively to save gigabytes of VRAM
        sims = F.cosine_similarity(rank_k_embs, gt_embeddings, dim=1).cpu().numpy()
        
        all_rank_scores.append(sims)
        all_rank_sentences.append(rank_k_sents)
        
    all_rank_scores = np.array(all_rank_scores)
    
    best_scores = np.max(all_rank_scores, axis=0)
    best_ranks = np.argmax(all_rank_scores, axis=0) 
    
    best_sentences = [all_rank_sentences[best_ranks[i]][i] for i in range(len(best_ranks))]

    print("\n--- 5 RANDOM MATCH EXAMPLES ---")
    random_indices = random.sample(range(len(best_scores)), min(5, len(best_scores)))
    for idx in random_indices:
        winning_rank = best_ranks[idx] + 1
        print(f"\nExample {idx}:")
        print(f"  Ground Truth: '{all_ground_truth_captions[idx]}'")
        print(f"  AI Generated: '{best_sentences[idx]}' (Selected Rank {winning_rank})")
        print(f"  Score:        {best_scores[idx]:.4f} / 1.0000")

    avg_score = np.mean(best_scores)
    
    print("\n=========================================")
    print("             FINAL EVALUATION            ")
    print("=========================================")
    print(f"Total Evaluated Captions: {pairs_found}")
    print(f"Evaluation Mode:          Top-{CHECK_TOP_K}")
    print(f"Average Semantic Score:   {avg_score:.4f} / 1.0000")
    print("=========================================")

if __name__ == "__main__":
    main()