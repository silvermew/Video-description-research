import os
import sys
import torch
import cv2
import math
import argparse
from tqdm import tqdm
from ultralytics import YOLO

# --- Import our custom modules ---
try:
    from mp4_to_trc_motionbert import MP4toTRCMotionBERT
    from models.caption_model import MotionCaptioner
    from models.encoder import MotionEncoder
    from utils.text_utils import build_vocab
    from utils.data_utils import load_data_from_trc, load_semantic_data_from_trc
    
    from yolov10_spatial_context import YOLOSpatialContext
    from llm_orchestrator import LLMOrchestrator
    from utils.interaction_detector import InteractionDetector
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def clean_sentence(sentence):
    return sentence.strip()

def main():
    parser = argparse.ArgumentParser(description="Ultimate E2E Pipeline: Video -> MotionBERT TRC -> PyTorch + YOLOv10 -> Llama 3")
    parser.add_argument("input_video", type=str, nargs='?', help="Path to the input MP4 video.")
    args = parser.parse_args()

    video_path = args.input_video

    if not video_path:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        print("Please select an MP4 file in the dialog window...")
        video_path = filedialog.askopenfilename(
            title="Select an MP4 video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if not video_path:
            print("No file selected. Exiting.")
            sys.exit()

    if not os.path.exists(video_path):
        print(f"Error: The input video '{video_path}' does not exist.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_trc_path = f"{base_name}_predicted.trc"
    out_transcript_path = f"{base_name}_narrative_transcript.txt"

    print("==================================================")
    print("      INITIALIZING THE ULTIMATE PIPELINE          ")
    print("==================================================")

    # 1. Initialize YOLO & Interaction Detector
    print("[1/3] Loading YOLOv10...")
    yolo_detector = YOLOSpatialContext(model_path="yolov10n.pt", conf_threshold=0.5)
    interaction_detector = InteractionDetector(padding_pixels=20)

    # 2. Initialize LLM Orchestrator & Critic
    print("[2/3] Connecting to Llama 3 via Ollama...")
    llm = LLMOrchestrator(model_name="llama3:8b")
    
    from llm_critic import LLMQualityCritic
    print("      Connecting Quality QA Critic...")
    llm_critic = LLMQualityCritic(model_name="llama3:8b")

    # 3. Initialize PyTorch Captioner (ULTIMATE Config)
    print("[3/3] Loading PyTorch MotionCaptioner...")
    CONTEXT_DIM = 768
    NUM_HEADS = 8
    NUM_LAYERS_ENCODER = 1
    TEXT_EMBED_DIM = 768
    NUM_DECODER_LAYERS = 7
    WINDOW_SIZE = 60
    OVERLAP = 0  
    TOP_K = 3

    vocab = build_vocab("data/annotation_rewritten")
    start_idx = vocab.word2idx.get('<start>', 1)
    end_idx = vocab.word2idx.get('<end>', 2)
    unk_idx = vocab.word2idx.get('<unk>', 3)
    pad_idx = vocab.word2idx.get('<pad>', 0)

    motion_encoder = MotionEncoder(CONTEXT_DIM, NUM_HEADS, NUM_LAYERS_ENCODER).to(device)
    caption_model = MotionCaptioner(motion_encoder, len(vocab), TEXT_EMBED_DIM, NUM_HEADS, NUM_DECODER_LAYERS).to(device)
    caption_model.load_state_dict(torch.load("checkpoints_caption/caption_model_ULTIMATE_best.pth", map_location=device))
    caption_model.eval()
    
    # 4. Initialize YOLO Pose for Single-Frame coordinate extraction (spatial context)
    pose_estimator = YOLO("yolov8n-pose.pt")

    print("\n==================================================")
    print("      STEP 1: EXTRACTING MOTION (VIDEO -> TRC)    ")
    print("      (MotionBERT DSTformer + TRC Head)            ")
    print("==================================================")
    
    try:
        MB_BACKBONE = os.path.join("MotionBERT", "checkpoint", "pretrain", "MB_pretrain", "best_epoch.bin")
        converter = MP4toTRCMotionBERT(
            backbone_checkpoint=MB_BACKBONE,
            head_checkpoint="motionbert_trc_head_best.pth"
        )
        converter.convert_video(video_path, out_trc_path)
    except Exception as e:
        print(f"Motion extraction failed: {e}")
        sys.exit(1)

    print("\n==================================================")
    print("      STEP 2: RUNNING MASTER NARRATIVE LOOP       ")
    print("==================================================")
    
    # Load TRC Windows using Semantic Chunking
    windows_list, metadata = load_semantic_data_from_trc(
        file_path=out_trc_path, target_fps=30.0,
        right_asi_idx=0, left_asi_idx=0, sacrum_idx=0
    )
    num_windows = len(windows_list)
    chunk_boundaries = metadata['chunk_boundaries']

    # Open Video for Frame Extraction
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Processing {num_windows} windows (each {WINDOW_SIZE/fps:.2f} seconds)...")

    with open(out_transcript_path, 'w', encoding='utf-8') as f:
        f.write(f"Narrative Transcript for {base_name}\n")
        f.write("="*50 + "\n\n")

        with torch.no_grad():
            for w_idx in tqdm(range(num_windows), desc="Generating Narrative"):
                start_frame, end_frame = chunk_boundaries[w_idx]
                center_frame = start_frame + ((end_frame - start_frame) // 2)
                
                time_range_str = f"[{start_frame/fps:.1f}s - {end_frame/fps:.1f}s]"
                
                # --- A. PYTORCH MOTION INFERENCE ---
                real_motion = windows_list[w_idx].to(device)
                context_4d = caption_model.motion_encoder(real_motion)
                context = context_4d.mean(dim=2) 
                
                beams = [{'tokens': [start_idx], 'log_prob': 0.0, 'ended': False}]
                
                for _ in range(20):
                    new_beams = []
                    for beam in beams:
                        if beam['ended']:
                            new_beams.append(beam)
                            continue
                            
                        curr_tgt = torch.LongTensor([beam['tokens']]).to(device)
                        seq_len = curr_tgt.shape[1]
                        mask = generate_square_subsequent_mask(seq_len).to(device)
                        
                        text_emb = caption_model.embedding(curr_tgt) + caption_model.text_pos_embedding[:, :seq_len, :]
                        out = caption_model.decoder(tgt=text_emb, memory=context, tgt_mask=mask)
                        logits = caption_model.fc_out(out[:, -1, :])
                        
                        logits[0, unk_idx] = -float('Inf')
                        logits[0, pad_idx] = -float('Inf')
                        
                        log_probs = torch.log_softmax(logits, dim=-1)
                        top_log_probs, top_indices = torch.topk(log_probs, TOP_K, dim=-1)
                        
                        for k in range(TOP_K):
                            new_beams.append({
                                'tokens': beam['tokens'] + [top_indices[0][k].item()],
                                'log_prob': beam['log_prob'] + top_log_probs[0][k].item(),
                                'ended': (top_indices[0][k].item() == end_idx)
                            })
                            
                    new_beams.sort(key=lambda x: x['log_prob'] / len(x['tokens']), reverse=True)
                    beams = new_beams[:TOP_K] 
                    if all(b['ended'] for b in beams):
                        break

                best_branch = beams[0]
                best_words = [vocab.idx2word[idx] for idx in best_branch['tokens']]
                if best_words[0] == '<start>': best_words = best_words[1:]
                if best_words[-1] == '<end>': best_words = best_words[:-1]
                pytorch_text = clean_sentence(" ".join(best_words))

                # --- B. YOLO SPATIAL CONTEXT ---
                # Seek to center frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, center_frame)
                ret, frame = cap.read()
                
                yolo_data = []
                if ret:
                    pose_results = pose_estimator(frame, conf=0.5, verbose=False)
                    
                    if len(pose_results[0].boxes) > 0 and pose_results[0].keypoints is not None:
                        keypoints_xyn = pose_results[0].keypoints.xyn[0]
                        keypoints_conf = pose_results[0].keypoints.conf[0]
                        
                        # Grab Left Hip (index 11) as reference
                        l_hip_xyn = keypoints_xyn[11]
                        ref_x = int(l_hip_xyn[0].item() * frame_width)
                        ref_y = int(l_hip_xyn[1].item() * frame_height)
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        yolo_data = yolo_detector.process_frame(frame_rgb, (ref_x, ref_y))
                        
                        # Explicit Interaction Heuristic
                        interaction_flags = interaction_detector.detect_interactions(
                            yolo_data, keypoints_xyn, keypoints_conf, frame_width, frame_height
                        )
                        
                        # Append the interaction context to the motion string for the LLM
                        if interaction_flags and interaction_flags[0] != "[NO INTERACTION]":
                            pytorch_text += " " + " ".join(interaction_flags)

                # --- C. LLM SYNTHESIS (PASS 1) ---
                draft_sentence = llm.generate_narrative(pytorch_text, yolo_data)
                
                # --- D. LLM CRITIC (PASS 2) ---
                critic_result = llm_critic.evaluate_draft(pytorch_text, yolo_data, draft_sentence)
                narrative_sentence = critic_result.get('corrected_sentence', draft_sentence)
                
                # --- E. SAVE AND PRINT ---
                f.write(f"{time_range_str}\n")
                f.write(f"Motion: {pytorch_text}\n")
                if yolo_data:
                    objects_str = ", ".join([f"{o['object']} ({o['position']})" for o in yolo_data])
                    f.write(f"Objects: {objects_str}\n")
                f.write(f"Narrative: {narrative_sentence}\n\n")
                f.flush()

    cap.release()
    print(f"\n✅ All done! Full transcript saved to: {out_transcript_path}")

if __name__ == "__main__":
    main()
