import torch
from torch.utils.data import Dataset
import numpy as np
import os
from typing import Tuple, Dict, Any

# --- 1. CONSTANTS ---
MARKER_NAMES_64 = [
    'pHipOrigin', 'pRightASI', 'pLeftASI', 'pRightCSI', 'pLeftCSI', 'pRightIschialTub',
    'pLeftIschialTub', 'pSacrum', 'pL5SpinalProcess', 'pL3SpinalProcess', 'pT12SpinalProcess',
    'pPX', 'pIJ', 'pT4SpinalProcess', 'pT8SpinalProcess', 'pC7SpinalProcess', 'pTopOfHead',
    'pRightAuricularis', 'pLeftAuricularis', 'pBackOfHead', 'pRightAcromion', 'pLeftAcromion',
    'pRightArmLatEpicondyle', 'pRightArmMedEpicondyle', 'pLeftArmLatEpicondyle',
    'pLeftArmMedEpicondyle', 'pRightUlnarStyloid', 'pRightRadialStyloid', 'pRightOlecranon',
    'pLeftUlnarStyloid', 'pLeftRadialStyloid', 'pLeftOlecranon', 'pRightTopOfHand',
    'pRightPinky', 'pRightBallHand', 'pLeftTopOfHand', 'pLeftPinky', 'pLeftBallHand',
    'pRightGreaterTrochanter', 'pRightKneeLatEpicondyle', 'pRightKneeMedEpicondyle',
    'pRightPatella', 'pLeftGreaterTrochanter', 'pLeftKneeLatEpicondyle', 'pLeftKneeMedEpicondyle',
    'pLeftPatella', 'pRightLatMalleolus', 'pRightMedMalleolus', 'pRightTibialTub',
    'pLeftLatMalleolus', 'pLeftMedMalleolus', 'pLeftTibialTub', 'pRightHeelFoot',
    'pRightFirstMetatarsal', 'pRightFifthMetatarsal', 'pRightPivotFoot', 'pRightHeelCenter',
    'pRightToe', 'pLeftHeelFoot', 'pLeftFirstMetatarsal', 'pLeftFifthMetatarsal',
    'pLeftPivotFoot', 'pLeftHeelCenter', 'pLeftToe'
]

# --- 2. DATASET CLASS ---
class LazyWindowDataset(Dataset):
    """
    Loads window data from .npy files on disk only when requested.
    """
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        window_path = self.file_paths[idx]
        window_np = np.load(window_path) 
        clean_data = torch.from_numpy(window_np.astype(np.float32))
        return clean_data

# --- 3. ROBUST PARSING ---
def parse_trc_robust(file_path):
    """
    Robustly parses a TRC file by searching for the 'Frame#' header.
    Returns: numpy array of shape (Frames, Markers, 3)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    header_index = 0
    found_header = False
    for i, line in enumerate(lines):
        if line.strip().startswith("Frame#"):
            header_index = i
            found_header = True
            break
            
    if not found_header:
        raise ValueError(f"Could not find 'Frame#' header in {file_path}")

    # Data usually starts 2 lines after "Frame#"
    data_start_index = header_index + 2
    
    # Check marker count from header
    header_parts = [p for p in lines[header_index].split('\t') if p.strip()]
    num_markers = len(header_parts) - 2 
    
    raw_data = []
    
    for i in range(data_start_index, len(lines)):
        line = lines[i].strip()
        if not line: continue
        
        parts = line.split('\t')
        clean_parts = [p for p in parts if p.strip()]
        
        try:
            # Skip Frame (0) and Time (1), take coords
            coords = [float(x) for x in clean_parts[2:]]
            if len(coords) == num_markers * 3:
                raw_data.append(coords)
        except ValueError:
            continue

    flat_data = np.array(raw_data, dtype=np.float32)
    
    if flat_data.size == 0:
        raise ValueError(f"No valid data found in {file_path}")

    n_frames = flat_data.shape[0]
    shaped_data = flat_data.reshape(n_frames, num_markers, 3)
    
    return shaped_data

# --- 4. NORMALIZATION LOGIC ---
def normalize_and_center_data(data_array: np.ndarray, root_idx: int = 0):
    """
    Centers the skeleton on the root joint and standardizes variance.
    Returns: (Normalized Data, Stats Dictionary)
    """
    # Extract Root (e.g., Hip) -> Shape: (T, 1, 3)
    root_trajectory = data_array[:, root_idx:root_idx+1, :].copy() 
    
    # 1. Center: Subtract root from everything
    centered_data = data_array - root_trajectory
    
    # 2. Normalize: Scale by Global Mean/Std of the CENTERED data
    data_mean = np.mean(centered_data)
    data_std = np.std(centered_data)
    
    normalized_data = (centered_data - data_mean) / (data_std + 1e-8)
    
    return normalized_data, {
        "mean": data_mean, 
        "std": data_std, 
        "root_trajectory": root_trajectory
    }

def denormalize_data(normalized_data, mean, std, root_trajectory):
    """
    Reverses normalization to get back mm coordinates.
    """
    # 1. Rescale
    centered_data = (normalized_data * std) + mean
    # 2. Add Root back
    original_data = centered_data + root_trajectory
    return original_data

# --- 5. WRITING TRC FILES ---
def write_trc_file_manually(file_path: str, data_array: np.ndarray, metadata: Dict[str, Any]):
    """
    Writes a numpy array to a .trc file format.
    """
    if data_array.ndim != 3 or data_array.shape[2] != 3:
        raise ValueError(f"Data array must be [Frames, Markers, 3]. Found: {data_array.shape}")

    num_frames, num_markers, _ = data_array.shape
    
    # Defaults
    data_rate = metadata.get('target_fps', 30.0)
    marker_names = metadata.get('marker_names', MARKER_NAMES_64)
    original_num_frames = metadata.get('original_num_frames', num_frames)
    base_file_name = os.path.basename(file_path).replace('.trc', '')

    with open(file_path, 'w') as f:
        # Header L1
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{base_file_name}\n")
        # Header L2
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        # Header L3
        f.write(f"{data_rate:.2f}\t{data_rate:.2f}\t{original_num_frames}\t{num_markers}\tmm\t{data_rate:.2f}\t1\t{original_num_frames}\n")
        # Header L4 (Names)
        f.write("Frame#\tTime\t" + "\t".join(marker_names) + "\n")
        # Header L5 (XYZ subheader)
        coord_line = [""] * 2
        for _ in range(num_markers): coord_line.extend(["X", "Y", "Z"])
        f.write("\t".join(coord_line) + "\n\n")

        # Data Rows
        for i in range(num_frames):
            frame_num = i + 1
            time_sec = float(i) / data_rate
            # Flatten X,Y,Z for this frame
            row_data = data_array[i].flatten()
            row_str = "\t".join([f"{x:.4f}" for x in row_data])
            f.write(f"{frame_num}\t{time_sec:.6f}\t{row_str}\n")

    print(f"✅ TRC file successfully written to: {file_path}")

# --- 6. MAIN PIPELINES (Used by inference.py and preprocess.py) ---

def load_data_from_trc(file_path, target_fps, window_size, right_asi_idx, left_asi_idx, sacrum_idx, overlap):
    """
    Parses, Normalizes, and Windows the data.
    Note: 'right_asi_idx', 'left_asi_idx', 'sacrum_idx' are kept for compatibility 
    with inference scripts, but the robust normalizer now defaults to using 
    Index 0 (Hip) as the root.
    """
    # 1. Parse
    raw_data = parse_trc_robust(file_path)
    
    # 2. Normalize (Center on Hip, Index 0)
    processed_data, stats = normalize_and_center_data(raw_data, root_idx=0)
    
    # 3. Windowing
    num_frames = processed_data.shape[0]
    stride = window_size - overlap
    windows = []
    
    if num_frames >= window_size:
        for i in range(0, num_frames - window_size + 1, stride):
            windows.append(processed_data[i:i+window_size])
    else:
        windows.append(processed_data) 

    if not windows:
        final_input = np.zeros((1, window_size, 64, 3)) 
    else:
        final_input = np.array(windows)

    # 4. Metadata
    metadata = {
        'original_trc_path': file_path,
        'target_fps': target_fps,
        'marker_names': MARKER_NAMES_64,
        'norm_mean': stats['mean'],
        'norm_std': stats['std'],
        'original_num_frames': num_frames
    }
    
    return torch.from_numpy(final_input).float(), metadata

def load_semantic_data_from_trc(file_path, target_fps, right_asi_idx=0, left_asi_idx=0, sacrum_idx=0):
    """
    Parses, Normalizes, and Windows the data using Semantic Kinematic Chunking.
    Returns a list of tensors (each representing a variable-length chunk) and chunk boundaries.
    """
    from utils.segmentation import KinematicSegmenter
    
    # 1. Parse
    raw_data = parse_trc_robust(file_path)
    
    # 2. Normalize (Center on Hip, Index 0)
    processed_data, stats = normalize_and_center_data(raw_data, root_idx=0)
    
    # 3. Semantic Windowing
    segmenter = KinematicSegmenter()
    chunk_boundaries = segmenter.segment(processed_data)
    
    windows = []
    for start, end in chunk_boundaries:
        chunk_data = processed_data[start:end]
        # Shape: (1, chunk_size, 64, 3)
        windows.append(torch.from_numpy(chunk_data).unsqueeze(0).float())
        
    num_frames = processed_data.shape[0]

    # 4. Metadata
    metadata = {
        'original_trc_path': file_path,
        'target_fps': target_fps,
        'marker_names': MARKER_NAMES_64,
        'norm_mean': stats['mean'],
        'norm_std': stats['std'],
        'original_num_frames': num_frames,
        'chunk_boundaries': chunk_boundaries
    }
    
    return windows, metadata

def save_data_to_trc(reconstructed_data_tensor, metadata, file_path, insert_root_trajectory=None):
    """
    Denormalizes, adds root trajectory back, and saves to TRC.
    """
    # 1. To Numpy
    if isinstance(reconstructed_data_tensor, torch.Tensor):
        data = reconstructed_data_tensor.detach().cpu().numpy()
    else:
        data = reconstructed_data_tensor
        
    if data.ndim == 4: data = data[0] # Remove batch dim if present

    # 2. Prepare Root
    if insert_root_trajectory is not None:
        print("🔄 Restoring global movement...")
        
        # --- NEW: VELOCITY SCALING HACK ---
        # If your feet are sliding "backwards" (Moonwalk), the root is too fast.
        # Try multiplying by 0.9 or 0.85 to slow down the global movement.
        VELOCITY_SCALE = 0.90  
        
        # Calculate velocity (difference between frames)
        velocity = np.diff(insert_root_trajectory, axis=0, prepend=insert_root_trajectory[0:1])
        
        # Scale velocity and integrate back to position
        scaled_velocity = velocity * VELOCITY_SCALE
        scaled_root = np.cumsum(scaled_velocity, axis=0)
        
        # Use this NEW root instead of the original
        # -----------------------------------
        
        min_len = min(len(data), len(scaled_root))
        root_to_add = scaled_root[:min_len]
        data = data[:min_len]

    # 3. Denormalize
    final_data = denormalize_data(
        data, 
        metadata['norm_mean'], 
        metadata['norm_std'], 
        root_to_add
    )

    # 4. Save
    write_trc_file_manually(file_path, final_data, metadata)