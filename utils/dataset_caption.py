import os
import torch
import numpy as np
from torch.utils.data import Dataset
from .text_utils import text_to_indices

class MotionCaptionDataset(Dataset):
    def __init__(self, data_pairs, vocab, max_len=20):
        self.data_pairs = data_pairs
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        npy_path, text = self.data_pairs[idx]
        
        # 1. Load Motion
        motion_np = np.load(npy_path)
        motion_tensor = torch.from_numpy(motion_np.astype(np.float32))
        
        # 2. Tokenize Text
        text_tensor = text_to_indices(text, self.vocab, self.max_len)
        
        # [CHANGED HERE] Return a dictionary so the training loop can easily unpack it
        return {
            "motion": motion_tensor,
            "text": text_tensor
        }

def prepare_caption_pairs(processed_windows_dir, data_root, window_size=60, overlap=10):
    """
    Production-Grade Matching Logic: Uses a 'Purity Threshold' to ensure 
    windows are only assigned text if they cleanly match the action.
    """
    pairs = []
    stride = window_size - overlap
    
    # We require the window to be 85% filled by a single action (51 out of 60 frames)
    # This completely eliminates messy "transition" windows.
    PURITY_THRESHOLD = 0.85 
    
    window_files = [f for f in os.listdir(processed_windows_dir) if f.endswith('.npy')]
    print(f"Scanning {len(window_files)} windows for pure text matches...")
    
    annotations_root = os.path.join(data_root, "annotation_ai_clean")
    
    from collections import defaultdict
    files_by_sequence = defaultdict(list)
    for w_file in window_files:
        name_parts = w_file.replace('.npy', '').split('_')
        if len(name_parts) >= 3:
            file_id = name_parts[-2]
            folder_name = "_".join(name_parts[:-2])
            files_by_sequence[(folder_name, file_id)].append(w_file)

    for (folder_name, file_id), w_files in files_by_sequence.items():
        file_id_fixed = file_id.replace('cap-', 'annotation-')
        anno_path = os.path.join(annotations_root, folder_name, f"{file_id_fixed}.ant")
        
        if not os.path.exists(anno_path):
            continue
            
        # 1. Read annotations and sort chronologically
        raw_annotations = []
        with open(anno_path, 'r') as f:
            for line in f:
                line_parts = line.strip().split('\t')
                if len(line_parts) >= 2:
                    try:
                        raw_annotations.append((int(line_parts[0]), line_parts[1]))
                    except ValueError:
                        continue
        
        if len(raw_annotations) == 0:
            continue
            
        raw_annotations.sort(key=lambda x: x[0])
        
        # 2. Convert timestamps into strict Start/End Segments
        # Example: Action 1 goes from its timestamp to the timestamp of Action 2.
        segments = []
        for i in range(len(raw_annotations)):
            start_frame = raw_annotations[i][0]
            text = raw_annotations[i][1]
            
            if i + 1 < len(raw_annotations):
                end_frame = raw_annotations[i+1][0]
            else:
                end_frame = float('inf') # The final action lasts until the video ends
                
            segments.append((start_frame, end_frame, text))

        # 3. Evaluate every window using the Purity Rule
        for w_file in w_files:
            window_part = w_file.replace('.npy', '').split('_')[-1]
            try:
                window_idx = int(window_part.replace('w', ''))
            except ValueError:
                continue
                
            w_start = window_idx * stride
            w_end = w_start + window_size
            
            best_text = None
            max_overlap = 0
            
            # Check how many frames of this window overlap with each segment
            for s_start, s_end, text in segments:
                overlap_start = max(w_start, s_start)
                overlap_end = min(w_end, s_end)
                
                overlap_frames = max(0, overlap_end - overlap_start)
                
                if overlap_frames > max_overlap:
                    max_overlap = overlap_frames
                    best_text = text
            
            # THE MOMENT OF TRUTH: Does the best matching action cover at least 85% of the window?
            purity_ratio = max_overlap / window_size
            
            if purity_ratio >= PURITY_THRESHOLD:
                pairs.append((os.path.join(processed_windows_dir, w_file), best_text))
                            
    print(f"✅ Strict Filtering complete. Retained {len(pairs)} highly-pure text-motion pairs.")
    return pairs