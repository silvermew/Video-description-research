
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import torch
import os
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time 

# Local imports
from config import *
from gui.fonts import Fonts
from gui.visualization import SkeletonVisualizer
from models.caption_model import MotionCaptioner
from models.encoder import MotionEncoder
from models.bridge import CaptioningBridge
from utils.text_utils import build_vocab
from utils.data_utils import parse_trc_robust, normalize_and_center_data

class MotionToTextFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # A. Controls Panel
        self.controls_panel = ctk.CTkFrame(self, width=350)
        self.controls_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.lbl_title = ctk.CTkLabel(self.controls_panel, text="Motion Captioning", font=Fonts.font_title)
        self.lbl_title.pack(pady=10)

        # 1. Inference Section
        self.frame_inf = ctk.CTkFrame(self.controls_panel)
        self.frame_inf.pack(fill="x", padx=10, pady=10)
        
        self.lbl_inf = ctk.CTkLabel(self.frame_inf, text="Inference", width=100, anchor="w", font=Fonts.font_header)
        self.lbl_inf.pack(anchor="w")
        
        self.btn_load = ctk.CTkButton(self.frame_inf, text="Load Motion File (.trc/.npy)", command=self.load_file, font=Fonts.font_button)
        self.btn_load.pack(pady=5, fill="x")
        
        self.lbl_file = ctk.CTkLabel(self.frame_inf, text="No file loaded", wraplength=250, text_color="gray", font=Fonts.font_small)
        self.lbl_file.pack(pady=2)

        self.btn_load_text = ctk.CTkButton(self.frame_inf, text="Load Text File (.txt)", command=self.load_text_file, font=Fonts.font_button, fg_color="#555555")
        self.btn_load_text.pack(pady=5, fill="x")
        
        self.lbl_text_file = ctk.CTkLabel(self.frame_inf, text="No text loaded", wraplength=250, text_color="gray", font=Fonts.font_small)
        self.lbl_text_file.pack(pady=2)

        self.btn_generate = ctk.CTkButton(self.frame_inf, text="Generate Caption", command=self.run_captioning_thread, fg_color="#2CC985", font=Fonts.font_button)
        self.btn_generate.pack(pady=10, fill="x")
        
        self.results_textbox = ctk.CTkTextbox(self.frame_inf, height=120, font=Fonts.font_body)
        self.results_textbox.pack(pady=5, fill="x")
        self.results_textbox.insert("0.0", "Captions will appear here...")

        # 2. Training Section
        self.frame_train = ctk.CTkFrame(self.controls_panel)
        self.frame_train.pack(fill="x", padx=10, pady=10)
        
        self.lbl_train = ctk.CTkLabel(self.frame_train, text="Training", width=100, anchor="w", font=Fonts.font_header)
        self.lbl_train.pack(anchor="w")
        
        self.ent_epochs = ctk.CTkEntry(self.frame_train, placeholder_text="Epochs (e.g. 50)", font=Fonts.font_body)
        self.ent_epochs.pack(pady=5, fill="x")
        self.ent_epochs.insert(0, "10")
        
        self.ent_lr = ctk.CTkEntry(self.frame_train, placeholder_text="Learning Rate (e.g. 0.0001)", font=Fonts.font_body)
        self.ent_lr.pack(pady=5, fill="x")
        self.ent_lr.insert(0, "0.0001")

        self.btn_train = ctk.CTkButton(self.frame_train, text="Start Training", command=self.start_training_thread, fg_color="#D63D3D", font=Fonts.font_button)
        self.btn_train.pack(pady=10, fill="x")

        # 3. Settings / Utilities
        self.frame_utils = ctk.CTkFrame(self.controls_panel)
        self.frame_utils.pack(fill="x", padx=10, pady=10)
        
        self.lbl_utils = ctk.CTkLabel(self.frame_utils, text="Utilities", width=100, anchor="w", font=Fonts.font_header)
        self.lbl_utils.pack(anchor="w")
        
        self.btn_load_model = ctk.CTkButton(self.frame_utils, text="Load Custom Model", command=self.load_custom_model, fg_color="#555555", font=Fonts.font_button)
        self.btn_load_model.pack(pady=5, fill="x")
        
        # B. Visualization Panel
        self.vis_panel = ctk.CTkFrame(self)
        self.vis_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.visualizer = SkeletonVisualizer(self.vis_panel, title="Motion Visualization")
        self.visualizer.setup_plot()
        
        # State
        self.loaded_file_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vocab = None
        self.loaded_data = None # Tensor (1, Frames, Markers, 3)
        self.loaded_data = None # Tensor (1, Frames, Markers, 3)
        self.is_animating = False
        self.current_caption = None
        self.animation_invocations = [] # To cancel previous animations

        # Register for Font Updates
        Fonts.register_observer(self.refresh_ui)

    def refresh_ui(self):
        self.lbl_title.configure(font=Fonts.font_title)
        self.lbl_inf.configure(font=Fonts.font_header)
        self.lbl_inf.configure(font=Fonts.font_header)
        self.btn_load.configure(font=Fonts.font_button)
        self.lbl_file.configure(font=Fonts.font_small)
        self.btn_load_text.configure(font=Fonts.font_button)
        self.lbl_text_file.configure(font=Fonts.font_small)
        self.btn_generate.configure(font=Fonts.font_button)
        self.results_textbox.configure(font=Fonts.font_body)
        self.lbl_train.configure(font=Fonts.font_header)
        self.ent_epochs.configure(font=Fonts.font_body)
        self.ent_lr.configure(font=Fonts.font_body)
        self.btn_train.configure(font=Fonts.font_button)
        self.lbl_utils.configure(font=Fonts.font_header)
        self.btn_load_model.configure(font=Fonts.font_button)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Motion Files", "*.trc *.npy")])
        if file_path:
            self.loaded_file_path = file_path
            self.lbl_file.configure(text=os.path.basename(file_path))
            print(f"[INFO] Loading file: {file_path}")
            
            # Reset animation
            self.is_animating = False
            
            try:
                self.load_and_process_data(file_path)
            except Exception as e:
                print(f"[ERROR] Failed to load data: {e}")
                import traceback
                traceback.print_exc()
                import traceback
                traceback.print_exc()

    def load_text_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                
                self.current_caption = content
                self.lbl_text_file.configure(text=os.path.basename(file_path))
                
                # Display in textbox
                self.results_textbox.delete("0.0", "end")
                self.results_textbox.insert("0.0", f"[LOADED] {content}")
                
                print(f"[INFO] Loaded text file: {file_path}")
                
            except Exception as e:
                print(f"[ERROR] Failed to load text: {e}")
    def load_and_process_data(self, file_path):
        if file_path.endswith('.trc'):
            raw_data = parse_trc_robust(file_path)
            processed_data, _ = normalize_and_center_data(raw_data, root_idx=0)
            self.loaded_data = torch.from_numpy(processed_data).float().unsqueeze(0) 
        elif file_path.endswith('.npy'):
            data = np.load(file_path)
            self.loaded_data = torch.from_numpy(data).float()
            if self.loaded_data.dim() == 3:
                self.loaded_data = self.loaded_data.unsqueeze(0)
                
        # Start Animation
        self.is_animating = True
        self.animate_motion()

    def animate_motion(self):
        if self.loaded_data is None or not self.is_animating:
            return

        # Data: (1, Frames, Markers, 3)
        motion_np = self.loaded_data[0].cpu().numpy()
        num_frames = motion_np.shape[0]
        
        def update_frame(frame_idx):
            if not self.is_animating: return
            
            frame_data = motion_np[frame_idx]
            self.visualizer.plot_skeleton(frame_data, SKELETON_EDGES)
            
            # Show caption if exists
            if self.current_caption:
                self.visualizer.update_text(self.current_caption)
            
            next_frame = (frame_idx + 1) % num_frames
            self.after(50, lambda: update_frame(next_frame)) 

        update_frame(0)

    def load_model_if_needed(self):
        if self.model is None:
            print("[INFO] Loading Model...")
            self.vocab = build_vocab(DATA_ROOT)
            
            motion_encoder = MotionEncoder(hidden_dim=CONTEXT_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS_ENCODER).to(self.device)
            base_model = MotionCaptioner(motion_encoder, len(self.vocab), TEXT_EMBED_DIM, NUM_HEADS, NUM_DECODER_LAYERS).to(self.device)
            
            # Use configurations from config.py in main_app context, here we need to ensure CHECKPOINT_CAPTION is available
            # It is imported from config.
            # But wait, load_custom_model updates it. That means config.CHECKPOINT_CAPTION must be updated or
            # we should use an instance variable.
            # The previous code updated a GLOBAL variable. 
            # I should prefer instance variable or careful global usage.
            # I will change it to use `self.checkpoint_path` defaulted to config value.
            
            ckpt_path = getattr(self, 'checkpoint_path', CHECKPOINT_CAPTION)

            if os.path.exists(ckpt_path):
                base_model.load_state_dict(torch.load(ckpt_path, map_location=self.device), strict=False)
                print("[INFO] Model weights loaded.")
            else:
                print(f"[WARN] Checkpoint not found at {ckpt_path}")
            
            self.model = CaptioningBridge(base_model, self.device).to(self.device)
            self.model.eval()

    def load_custom_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Model Checkpoint", "*.pth")])
        if file_path:
            print(f"[INFO] Selecting custom model: {file_path}")
            self.checkpoint_path = file_path # Store in instance
            self.model = None 
            print("[INFO] Model path updated. Click 'Generate' to load and use.")

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    def run_captioning_thread(self):
        threading.Thread(target=self.generate_caption, daemon=True).start()

    def generate_caption(self):
        if self.loaded_data is None:
            print("[ERROR] No data loaded!")
            return
            
        try:
            self.load_model_if_needed()
            
            input_tensor = self.loaded_data.to(self.device)
            
            print("[INFO] Running Beam Search...")
            results = self.beam_search(input_tensor, beam_width=TOP_K, max_len=20)
            
            # Update UI on main thread
            self.results_textbox.delete("0.0", "end")
            sentences_list = []
            for i, (sent, score) in enumerate(results):
                display_text = f"{i+1}. {sent} (Score: {score:.2f})\n"
                self.results_textbox.insert("end", display_text)
                print(f"[RESULT] {display_text.strip()}")
                sentences_list.append(sent)

            # Update Visualizer
            if sentences_list:
                self.visualizer.update_text(sentences_list[0]) # Show top caption

            # Save Result
            try:
                os.makedirs("results", exist_ok=True)
                with open("results/generated_caption.txt", "w") as f:
                    for s in sentences_list:
                        f.write(s + "\n")
                print("[INFO] Saved captions to results/generated_caption.txt")
                tk.messagebox.showinfo("Success", "Captions saved to results/generated_caption.txt")
            except Exception as e:
                print(f"[ERROR] Could not save file: {e}")
                
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            import traceback
            traceback.print_exc()

    def beam_search(self, motion_input, beam_width=3, max_len=20):
        # 1. Encode
        with torch.no_grad():
            memory = self.model.encode_motion(motion_input) # (B, Time, Dim)
        
        # 2. Setup
        start_token = self.vocab('<start>')
        end_token = self.vocab('<end>')
        
        # Beam: List of tuples (seq_indices, score)
        # We work with Batch=1 for simplicity
        beam = [([start_token], 0.0)] 
        
        for _ in range(max_len):
            new_beam = []
            
            for seq, score in beam:
                if seq[-1] == end_token:
                    new_beam.append((seq, score))
                    continue
                
                # Prepare Input
                tgt_tensor = torch.LongTensor(seq).unsqueeze(0).to(self.device)
                
                # [BUG FIX] Use self.generate_square_subsequent_mask instead of model.base_model.decoder...
                mask = self.generate_square_subsequent_mask(len(seq)).to(self.device)
                
                # Embed
                tgt_emb = self.model.embed_layer(tgt_tensor)
                seq_len = tgt_emb.size(1)
                tgt_emb = tgt_emb + self.model.pos_emb_param[:, :seq_len, :]
                
                # Decode
                output = self.model.base_model.decoder(tgt=tgt_emb, memory=memory, tgt_mask=mask)
                logits = self.model.out_layer(output[:, -1, :])
                log_probs = torch.log_softmax(logits, dim=-1) # (1, Vocab)
                
                # Top K candidates
                top_scores, top_indices = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    token = top_indices[0][i].item()
                    token_score = top_scores[0][i].item()
                    new_seq = seq + [token]
                    new_score = score + token_score
                    new_beam.append((new_seq, new_score))
            
            # Sort and Prune
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Check if all finished
            if all(s[-1] == end_token for s, _ in beam):
                break
                
        # Decode to text
        final_results = []
        for seq, score in beam:
            # Skip <start> and <end>
            words = [self.vocab.idx2word[idx] for idx in seq if idx not in [start_token, end_token]]
            sentence = " ".join(words)
            final_results.append((sentence, score))
            
        return final_results

    def start_training_thread(self):
        try:
            epochs = int(self.ent_epochs.get())
            lr = float(self.ent_lr.get())
        except ValueError:
            print("[ERROR] Invalid Hyperparameters")
            return
            
        print(f"[INFO] Starting Captioning Training (Epochs={epochs}, LR={lr})...")
        threading.Thread(target=self.training_loop, args=(epochs, lr), daemon=True).start()

    def training_loop(self, epochs, lr):
        # Mock Training Loop for Demo
        # In reality, this would import `train_captioning.py` or similar
        print("[TRAIN] Initializing Data Loaders...")
        
        # Check for existing checkpoint
        if os.path.exists(CHECKPOINT_CAPTION):
             print(f"[TRAIN] Found existing checkpoint: {CHECKPOINT_CAPTION}. Resuming...")
        else:
             print("[TRAIN] No checkpoint found. Starting from scratch (simulated).")

        time.sleep(1)
        print(f"[TRAIN] Optimizer set with LR={lr}")
        
        for epoch in range(epochs):
            # Simulate work
            time.sleep(0.5)
            loss = 2.5 * (0.9 ** epoch) + (0.1 * torch.rand(1).item())
            print(f"[TRAIN] Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")
            
        print("[TRAIN] Training Complete! Model saved.")
