import numpy as np
import os
import shutil
import sys
from utils.data_utils import parse_trc_robust, normalize_and_center_data

# --- CONFIGURATION ---
DATA_DIRECTORY = 'data'            
OUTPUT_DIR = 'processed_windows'   
WINDOW_SIZE = 60
OVERLAP = 10
ROOT_INDEX = 0  # Index of pHipOrigin

if __name__ == "__main__":
    print(f"--- STARTING PREPROCESSING ---")
    
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    # Recursive search
    trc_files = []
    for root, dirs, files in os.walk(DATA_DIRECTORY):
        for file in files:
            if file.lower().endswith('.trc'):
                trc_files.append(os.path.join(root, file))
    
    print(f"Found {len(trc_files)} TRC files.")
    
    count = 0
    for trc_file in trc_files:
        try:
            # 1. Parse
            raw_data = parse_trc_robust(trc_file)
            
            # 2. Normalize
            processed_data, stats = normalize_and_center_data(raw_data, root_idx=ROOT_INDEX)
            
            # 3. Window & Save
            num_frames = processed_data.shape[0]
            stride = WINDOW_SIZE - OVERLAP
            
            if num_frames >= WINDOW_SIZE:
                num_windows = (num_frames - WINDOW_SIZE) // stride + 1
                for i in range(num_windows):
                    start = i * stride
                    end = start + WINDOW_SIZE
                    window = processed_data[start:end]
                    
                    # Naming: FolderName_FileName_w0.npy
                    folder_name = os.path.basename(os.path.dirname(trc_file))
                    file_name = os.path.basename(trc_file).replace('.trc', '')
                    out_name = f"{folder_name}_{file_name}_w{i}.npy"
                    
                    np.save(os.path.join(OUTPUT_DIR, out_name), window)
                    count += 1
        except Exception as e:
            print(f"Error processing {trc_file}: {e}")

    print(f"\n✅ Created {count} windows in '{OUTPUT_DIR}'")