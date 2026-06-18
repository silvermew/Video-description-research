import os
import glob
import random
import json
import requests
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
GENERATED_DIR = "results/motion_to_text_ULTIMATE"
ANNOTATION_DIR = "data/annotation_rewritten" 
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:70b"
CHECK_TOP_K = 3
EVAL_LIMIT = 100 # Number of random samples to evaluate (set to None for all)

# SYSTEM PROMPT for Llama evaluation
SYSTEM_PROMPT = """You are an expert evaluator for motion-to-text systems.
Your task is to compare a 'Ground Truth' motion description with an 'AI Generated' description.

Task:
Determine if the 'AI Generated' description is similar enough to the 'Ground Truth' that the overall movement is correctly understood and captured.

Criteria for 1 (Success):
- The core action is the same (e.g., 'walks' vs 'is walking').
- The subject and main objects are correctly identified.
- minor differences in wording or synonyms are acceptable.

Criteria for 0 (Failure):
- The core action is different (e.g., 'sits' vs 'stands').
- The description is nonsensical or unrelated to the ground truth.
- Critical details are hallucinated.

Output ONLY the digit '1' or '0'. Do not provide any explanation.
"""

def evaluate_with_llama(gt_text, gen_text):
    prompt = f"Ground Truth: \"{gt_text}\"\nAI Generated: \"{gen_text}\"\n\nJudgment (1 or 0):"
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
        "stream": False,
        "options": {
            "temperature": 0.0,
            "stop": ["\n"]
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json().get("response", "").strip()
        if '1' in result:
            return 1
        elif '0' in result:
            return 0
        return None
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

def main():
    print(f"1. Scanning for files in {GENERATED_DIR}...")
    gen_files = glob.glob(os.path.join(GENERATED_DIR, "*", "*.txt"))
    
    all_matched_pairs = []
    
    for gen_path in gen_files:
        path_parts = gen_path.split(os.sep)
        date_folder = path_parts[-2]
        gen_filename = path_parts[-1]
        
        ant_filename = gen_filename.replace('cap-', 'annotation-').replace('.txt', '.ant')
        ant_path = os.path.join(ANNOTATION_DIR, date_folder, ant_filename)
        
        if not os.path.exists(ant_path):
            continue
            
        gen_dict = {}
        with open(gen_path, 'r', encoding='utf-8') as f:
            next(f) # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    frame_idx = int(parts[0])
                    rank = int(parts[1])
                    sentence = ",".join(parts[3:]).strip()
                    if rank <= CHECK_TOP_K:
                        if frame_idx not in gen_dict:
                            gen_dict[frame_idx] = sentence
                    
        if not gen_dict:
            continue

        raw_annotations = []
        with open(ant_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    raw_annotations.append((int(parts[0]), parts[1].strip()))
                    
        if not raw_annotations:
            continue
            
        raw_annotations.sort(key=lambda x: x[0])
        
        segments = []
        for i in range(len(raw_annotations)):
            start_frame = raw_annotations[i][0]
            gt_text = raw_annotations[i][1]
            end_frame = raw_annotations[i+1][0] if i + 1 < len(raw_annotations) else float('inf')
            segments.append((start_frame, end_frame, gt_text))
            
        for gen_frame, gen_text in gen_dict.items():
            for s_start, s_end, gt_text in segments:
                if s_start <= gen_frame < s_end:
                    all_matched_pairs.append({
                        "gt": gt_text,
                        "gen": gen_text,
                        "file": gen_filename,
                        "frame": gen_frame
                    })
                    break 

    print(f"✅ Found {len(all_matched_pairs)} temporally matched pairs.")
    
    if EVAL_LIMIT and len(all_matched_pairs) > EVAL_LIMIT:
        print(f"-> Subsampling to {EVAL_LIMIT} random pairs...")
        all_matched_pairs = random.sample(all_matched_pairs, EVAL_LIMIT)
    
    print(f"\n2. Evaluating with {OLLAMA_MODEL}...")
    scores = []
    results_detail = []

    for pair in tqdm(all_matched_pairs):
        score = evaluate_with_llama(pair['gt'], pair['gen'])
        if score is not None:
            scores.append(score)
            results_detail.append({
                "gt": pair['gt'],
                "gen": pair['gen'],
                "score": score
            })

    if not scores:
        print("❌ No valid evaluations received from LLM.")
        return

    accuracy = (sum(scores) / len(scores)) * 100
    
    print("\n--- EVALUATION SAMPLES ---")
    random_samples = random.sample(results_detail, min(10, len(results_detail)))
    for res in random_samples:
        status = "✅ UNDERSTOOD" if res['score'] == 1 else "❌ MISMATCH"
        print(f"\n[{status}]")
        print(f"  Ground Truth: '{res['gt']}'")
        print(f"  AI Generated: '{res['gen']}'")

    print("\n=========================================")
    print("             LLM EVALUATION              ")
    print("=========================================")
    print(f"Total Evaluated: {len(scores)}")
    print(f"Understanding Score (Acc): {accuracy:.2f}%")
    print("=========================================")

if __name__ == "__main__":
    main()
