# -*- coding: utf-8 -*-
import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser(description="Evaluate a generated _predicted.txt against an annotation .ant file.")
    parser.add_argument("--predicted", type=str, required=True, help="Path to the generated text file (e.g., cam-001_predicted.txt)")
    parser.add_argument("--annotation", type=str, required=True, help="Path to the corresponding ground truth .ant file")
    parser.add_argument("--top_k", type=int, default=3, help="Evaluate up to Top-K ranks (default: 3)")
    args = parser.parse_args()

    if not os.path.exists(args.predicted):
        print("[Error] Predicted file not found at '{}'".format(args.predicted))
        return
    if not os.path.exists(args.annotation):
        print("[Error] Annotation file not found at '{}'".format(args.annotation))
        return

    print("1. Loading Semantic Evaluation Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    print("\n2. Parsing files and matching frames (Evaluating Top-{})...".format(args.top_k))
    
    # A. Load Generated Top-K Captions
    gen_dict = {}
    with open(args.predicted, 'r', encoding='utf-8') as f:
        header = next(f) # Skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    frame_idx = int(parts[0])
                    rank = int(parts[1])
                except ValueError:
                    continue # skip invalid rows
                
                # Rejoin any commas that were part of the sentence
                sentence = ",".join(parts[3:]).strip()
                
                if rank <= args.top_k:
                    if frame_idx not in gen_dict:
                        gen_dict[frame_idx] = {}
                    gen_dict[frame_idx][rank] = sentence
                    
    if not gen_dict:
        print("[Error] No valid generated captions found in the predicted file.")
        return

    # B. Load Ground Truth
    raw_annotations = []
    with open(args.annotation, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    frame = int(parts[0])
                except ValueError:
                    continue
                raw_annotations.append((frame, parts[1].strip()))
                
    if not raw_annotations:
        print("[Error] No valid annotations found in the .ant file.")
        return
        
    raw_annotations.sort(key=lambda x: x[0])
    
    # Match each annotation to the closest generated frame
    all_generated_candidates = [] 
    all_ground_truth_captions = []
    pairs_found = 0
    
    generated_frames = sorted(list(gen_dict.keys()))
    
    for ant_frame, gt_text in raw_annotations:
        # Find the closest generated frame
        closest_gen_frame = min(generated_frames, key=lambda x: abs(x - ant_frame))
        
        # We only consider it a match if it's reasonably close (within 60 frames)
        if abs(closest_gen_frame - ant_frame) <= 60:
            all_generated_candidates.append(gen_dict[closest_gen_frame])
            all_ground_truth_captions.append(gt_text)
            pairs_found += 1
            
    print("[Success] Found {} matches near annotation timestamps.".format(pairs_found))
    
    if pairs_found == 0:
        print("[Error] No matches found! Ensure the frames in the predicted file overlap with the annotations.")
        return

    print("\n3. Calculating Semantic Similarity across all {} ranks...".format(args.top_k))
    
    # Get ground truth embeddings
    gt_embeddings = model.encode(all_ground_truth_captions, convert_to_tensor=True, batch_size=128)
    
    all_rank_scores = []
    all_rank_sentences = []
    
    # Evaluate each rank independently
    for k in range(1, args.top_k + 1):
        # Fallback to rank 1 if a specific rank is missing for a frame
        rank_k_sents = [cands.get(k, cands.get(1, "")) for cands in all_generated_candidates]
        rank_k_embs = model.encode(rank_k_sents, convert_to_tensor=True, batch_size=128)
        
        # Calculate 1-to-1 similarity natively to save gigabytes of VRAM
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
        print("\nExample {}:".format(idx))
        print("  Ground Truth: '{}'".format(all_ground_truth_captions[idx]))
        print("  AI Generated: '{}' (Selected Rank {})".format(best_sentences[idx], winning_rank))
        print("  Score:        {:.4f} / 1.0000".format(best_scores[idx]))

    avg_score = np.mean(best_scores)
    
    print("\n=========================================")
    print("             FINAL EVALUATION            ")
    print("=========================================")
    print("Evaluated File:           {}".format(os.path.basename(args.predicted)))
    print("Ground Truth File:        {}".format(os.path.basename(args.annotation)))
    print("Total Evaluated Captions: {} (Sparse Match)".format(pairs_found))
    print("Evaluation Mode:          Top-{}".format(args.top_k))
    print("Average Semantic Score:   {:.4f} / 1.0000".format(avg_score))
    print("=========================================")

if __name__ == "__main__":
    main()
