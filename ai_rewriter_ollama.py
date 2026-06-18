import os
import sys
import json
import numpy as np
import requests
from tqdm import tqdm
from llama_cpp import Llama
from sklearn.cluster import AgglomerativeClustering
import contextlib

# ====================================================================
# CONFIGURATION
# ====================================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:70b"
EMBED_MODEL_PATH = "all-MiniLM-L6-v2.Q4_K_M.gguf"
INPUT_ROOT = "data/annotation"
OUTPUT_ROOT = "data/annotation_rewritten"
CACHE_FILE = "rewrite_cache_clustered.json"
N_CLUSTERS = 2000

# SYSTEM PROMPT for Llama 3.1 70B
SYSTEM_PROMPT = """You are an expert in motion analysis and descriptive linguistics. 
Your task is to rewrite short motion descriptions into a standardized, clean, and professional format.

Rules:
1. Simplify descriptions to focus on the action and the subject (e.g., "a person walks").
2. Standardize subjects: replace specific roles (teacher, actor, student) with "a person" unless the specific role is critical to the movement itself.
3. Fix any spelling or grammatical errors.
4. Ensure the tone is neutral and descriptive.
5. Use ONLY lowercase letters.
6. Do NOT use any trailing punctuation (like periods).
7. Output ONLY the rewritten sentence. Do not include any explanations or conversational text.

Examples:
- Input: "a studen is walking in the classroom" -> Output: "a person walks"
- Input: "he is runing very fast" -> Output: "a person runs quickly"
- Input: "the teacher sits down on a chair" -> Output: "a person sits down"
"""

@contextlib.contextmanager
def suppress_stderr():
    try:
        with open(os.devnull, "w") as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            yield
            sys.stderr = old_stderr
    except Exception:
        yield

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=4, ensure_ascii=False)

def rewrite_sentence(sentence, cache):
    if sentence in cache:
        return cache[sentence]

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\nInput: \"{sentence}\" -> Output:",
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "stop": ["\n", "->"]
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        rewritten = response.json().get("response", "").strip().strip('"').strip("'").lower().rstrip('.')
        cache[sentence] = rewritten
        return rewritten
    except Exception as e:
        print(f"\n❌ Error rewriting sentence '{sentence}': {e}")
        return None

def get_embeddings(sentences):
    print(f"-> Computing embeddings for {len(sentences)} sentences...")
    with suppress_stderr():
        llm = Llama(model_path=EMBED_MODEL_PATH, embedding=True, verbose=False)
    
    embeddings = []
    for text in tqdm(sentences, desc="Embedding"):
        embed = llm.create_embedding(text)['data'][0]['embedding']
        embeddings.append(embed)
    return np.array(embeddings)

def cluster_and_get_representatives(sentences, embeddings, n_clusters):
    print(f"-> Clustering into {n_clusters} groups...")
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norm + 1e-9)
    
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average'
    )
    clustering.fit(normalized_embeddings)
    
    labels = clustering.labels_
    cluster_groups = {}
    for idx, label in enumerate(labels):
        if label not in cluster_groups:
            cluster_groups[label] = []
        cluster_groups[label].append(sentences[idx])
    
    # Map every sentence to its cluster representative
    sentence_to_rep = {}
    representatives = []
    
    for label, group in cluster_groups.items():
        # Representative: choose the shortest one as it's usually the simplest
        rep = min(group, key=len)
        representatives.append(rep)
        for s in group:
            sentence_to_rep[s] = rep
            
    return sentence_to_rep, representatives

def process_annotations():
    print(f"Step 1: Reading annotations from {INPUT_ROOT}...")
    all_files_data = []
    unique_sentences = set()

    for root, _, files in os.walk(INPUT_ROOT):
        for file in files:
            if file.endswith(".ant"):
                path = os.path.join(root, file)
                file_lines = []
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            text = parts[1].strip()
                            file_lines.append((parts[0], text))
                            unique_sentences.add(text)
                all_files_data.append((path, file_lines))

    sentences = sorted(list(unique_sentences))
    print(f"-> Found {len(all_files_data)} files and {len(sentences)} unique sentences.")
    
    # STAGE 1: Clustering
    embeddings = get_embeddings(sentences)
    sentence_to_rep, representatives = cluster_and_get_representatives(sentences, embeddings, min(N_CLUSTERS, len(sentences)))
    
    # STAGE 2: Rewriting representatives
    cache = load_cache()
    to_rewrite = sorted(list(set(representatives)))
    actual_to_rewrite = [s for s in to_rewrite if s not in cache]
    
    if actual_to_rewrite:
        print(f"Step 2: Rewriting {len(actual_to_rewrite)} cluster representatives using {OLLAMA_MODEL}...")
        for i, sentence in enumerate(tqdm(actual_to_rewrite, desc="Rewriting")):
            rewritten = rewrite_sentence(sentence, cache)
            if rewritten:
                if (i + 1) % 10 == 0:
                    save_cache(cache)
        save_cache(cache)
    else:
        print("Step 2: All cluster representatives already in cache.")

    print(f"Step 3: Saving rewritten files to {OUTPUT_ROOT}...")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    for src_path, file_lines in tqdm(all_files_data, desc="Saving files"):
        rel_path = os.path.relpath(src_path, INPUT_ROOT)
        dest_path = os.path.join(OUTPUT_ROOT, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        with open(dest_path, 'w', encoding='utf-8') as fout:
            for frame, original_text in file_lines:
                rep = sentence_to_rep.get(original_text, original_text)
                rewritten_text = cache.get(rep, rep)
                fout.write(f"{frame}\t{rewritten_text}\n")

    print(f"✅ Done! Reduced to {len(set(cache.values()))} unique rewritten sentences.")
    print(f"Rewritten annotations are in: {OUTPUT_ROOT}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_ROOT):
        print(f"❌ Error: Input directory '{INPUT_ROOT}' not found.")
    elif not os.path.exists(EMBED_MODEL_PATH):
        print(f"❌ Error: Embedding model '{EMBED_MODEL_PATH}' not found.")
    else:
        process_annotations()
