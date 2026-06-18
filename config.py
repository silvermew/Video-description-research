
import os

# --- CONFIGURATION ---
# Default Checkpoints
CHECKPOINT_ENCODER = "checkpoints/best_encoder.pth"
CHECKPOINT_DIFFUSION = "checkpoints/best_diffusion.pth"
CHECKPOINT_CAPTION = "checkpoints_caption/caption_model_final.pth"

# Visualization
SKELETON_EDGES = [
    [38, 7],  [42, 7],       # Hips
    [7, 8], [8, 9], [9, 10], [10, 13], [13, 14], [14, 15], # Spine
    [18, 16], [17, 16], [15, 17], [15, 18],   # Head
    [15, 20], [15, 21],                       # Shoulders
    [20, 22], [22, 28], [28, 26], [26, 32],   # Right Arm
    [21, 24], [24, 31], [31, 29], [29, 35],   # Left Arm
     [38, 39], [39, 46], [46, 56], # Right Leg
     [42, 43], [43, 49], [49, 62]  # Left Leg
]
DATA_ROOT = "data/annotation_ai_clean"
CONTEXT_DIM = 256
NUM_HEADS = 8
NUM_LAYERS_ENCODER = 1
TEXT_EMBED_DIM = 256
NUM_DECODER_LAYERS = 4
INPUT_FEATURE_DIM = 263
TOP_K = 3
