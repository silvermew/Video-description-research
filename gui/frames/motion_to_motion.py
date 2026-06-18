
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import torch
import os
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import time

# Local imports
from config import *
from gui.fonts import Fonts
from gui.visualization import SkeletonVisualizer
from models.encoder import MotionEncoder
from models.diffusion_model import ConditionalDiffusionModel
from utils.scheduler import NoiseScheduler
from utils.data_utils import parse_trc_robust, normalize_and_center_data, load_data_from_trc

class MotionToMotionFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # A. Controls Panel
        self.controls_panel = ctk.CTkFrame(self, width=350)
        self.controls_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.lbl_title = ctk.CTkLabel(self.controls_panel, text="Motion Denoising (Diffusion)", font=Fonts.font_title)
        self.lbl_title.pack(pady=10)

        # 1. Inference Section
        self.frame_inf = ctk.CTkFrame(self.controls_panel)
        self.frame_inf.pack(fill="x", padx=10, pady=10)
        
        self.lbl_inf = ctk.CTkLabel(self.frame_inf, text="Inference", width=100, anchor="w", font=Fonts.font_header)
        self.lbl_inf.pack(anchor="w")
        
        self.btn_load = ctk.CTkButton(self.frame_inf, text="Load Noisy File (.trc)", command=self.load_file, font=Fonts.font_button)
        self.btn_load.pack(pady=5, fill="x")
        
        self.lbl_file = ctk.CTkLabel(self.frame_inf, text="No file loaded", wraplength=250, text_color="gray", font=Fonts.font_small)
        self.lbl_file.pack(pady=2)
        
        self.lbl_strength = ctk.CTkLabel(self.frame_inf, text="Denoising Strength (0.1 - 1.0)", font=Fonts.font_body)
        self.lbl_strength.pack(pady=(10,0))
        self.slider_strength = ctk.CTkSlider(self.frame_inf, from_=0.1, to=1.0, number_of_steps=9)
        self.slider_strength.set(0.6)
        self.slider_strength.pack(pady=5, fill="x")

        self.lbl_speed = ctk.CTkLabel(self.frame_inf, text="Playback Speed (FPS)", font=Fonts.font_body)
        self.lbl_speed.pack(pady=(10,0))
        self.slider_speed = ctk.CTkSlider(self.frame_inf, from_=10, to=60, number_of_steps=50)
        self.slider_speed.set(30)
        self.slider_speed.pack(pady=5, fill="x")

        self.btn_denoise = ctk.CTkButton(self.frame_inf, text="Denoise Motion", command=self.run_denoising_thread, fg_color="#3B8ED0", font=Fonts.font_button)
        self.btn_denoise.pack(pady=10, fill="x")
        
        # 2. Training Section
        self.frame_train = ctk.CTkFrame(self.controls_panel)
        self.frame_train.pack(fill="x", padx=10, pady=10)
        
        self.lbl_train = ctk.CTkLabel(self.frame_train, text="Training", width=100, anchor="w", font=Fonts.font_header)
        self.lbl_train.pack(anchor="w")
        
        self.btn_train = ctk.CTkButton(self.frame_train, text="Start Diffusion Training", command=self.start_training_thread, fg_color="#D63D3D", font=Fonts.font_button)
        self.btn_train.pack(pady=10, fill="x")

        # 3. Settings / Utilities
        self.frame_utils = ctk.CTkFrame(self.controls_panel)
        self.frame_utils.pack(fill="x", padx=10, pady=10)
        
        self.lbl_utils = ctk.CTkLabel(self.frame_utils, text="Utilities", width=100, anchor="w", font=Fonts.font_header)
        self.lbl_utils.pack(anchor="w")
        
        self.btn_load_enc = ctk.CTkButton(self.frame_utils, text="Load Custom Encoder", command=self.load_custom_encoder, fg_color="#555555", font=Fonts.font_button)
        self.btn_load_enc.pack(pady=5, fill="x")

        self.btn_load_diff = ctk.CTkButton(self.frame_utils, text="Load Custom Diffusion", command=self.load_custom_diffusion, fg_color="#555555", font=Fonts.font_button)
        self.btn_load_diff.pack(pady=5, fill="x")

        self.btn_vis_only = ctk.CTkButton(self.frame_utils, text="Visualize Result (.trc)", command=self.load_and_visualize_result, fg_color="#E59400", font=Fonts.font_button)
        self.btn_vis_only.pack(pady=5, fill="x")

        # B. Visualization Panel
        self.vis_panel = ctk.CTkFrame(self)
        self.vis_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.visualizer = SkeletonVisualizer(self.vis_panel, title="Original (Red) vs Denoised (Green)")
        self.visualizer.setup_plot()

        # State
        self.loaded_file_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.motion_encoder = None
        self.diffusion_model = None
        self.noise_scheduler = None
        
        self.input_data = None      # Full Original Data (Frames, Markers, 3) 
        self.denoised_data = None   # Full Denoised Data
        self.is_animating = False

        # Register for Font Updates
        Fonts.register_observer(self.refresh_ui)

    def refresh_ui(self):
        self.lbl_title.configure(font=Fonts.font_title)
        self.lbl_inf.configure(font=Fonts.font_header)
        self.btn_load.configure(font=Fonts.font_button)
        self.lbl_file.configure(font=Fonts.font_small)
        self.lbl_strength.configure(font=Fonts.font_body)
        self.lbl_speed.configure(font=Fonts.font_body)
        self.btn_denoise.configure(font=Fonts.font_button)
        self.lbl_train.configure(font=Fonts.font_header)
        self.btn_train.configure(font=Fonts.font_button)
        self.lbl_utils.configure(font=Fonts.font_header)
        self.btn_load_enc.configure(font=Fonts.font_button)
        self.btn_load_diff.configure(font=Fonts.font_button)
        self.btn_vis_only.configure(font=Fonts.font_button)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("TRC Files", "*.trc")])
        if file_path:
            self.loaded_file_path = file_path
            self.lbl_file.configure(text=os.path.basename(file_path))
            print(f"[INFO] Loaded file: {file_path}")
            
            # Preload for visualization
            try:
                raw_data = parse_trc_robust(file_path)
                self.input_data, _ = normalize_and_center_data(raw_data, root_idx=0)
                self.denoised_data = None # Reset
                
                self.is_animating = True
                
                # Dynamic Auto-Scaling
                # Data is (Frames, Markers, 3)
                all_flat = self.input_data.reshape(-1, 3)
                min_vals = all_flat.min(axis=0)
                max_vals = all_flat.max(axis=0)
                
                print(f"[DEBUG] Data Range: X[{min_vals[0]:.2f}, {max_vals[0]:.2f}], Y[{min_vals[1]:.2f}, {max_vals[1]:.2f}], Z[{min_vals[2]:.2f}, {max_vals[2]:.2f}]")
                
                # Add padding
                pad = 0.5
                self.visualizer.set_plot_limits(
                    (min_vals[0]-pad, max_vals[0]+pad), 
                    (min_vals[1]-pad, max_vals[1]+pad), 
                    (min_vals[2]-pad, max_vals[2]+pad)
                )
                
                self.animate_comparison()
                
            except Exception as e:
                print(f"[ERROR] Failed to load data: {e}")

    def animate_comparison(self):
        if self.input_data is None and self.denoised_data is None:
            return
        
        if not self.is_animating:
            return

        # Determine lengths
        len_input = self.input_data.shape[0] if self.input_data is not None else 0
        len_denoised = self.denoised_data.shape[0] if self.denoised_data is not None else 0
        num_frames = max(len_input, len_denoised)
        
        def update_frame(frame_idx):
            if not self.is_animating: return
            
            self.visualizer.setup_plot()
            
            # Plot Original (if available)
            if self.input_data is not None:
                if frame_idx < len_input:
                    orig_frame = self.input_data[frame_idx]
                    self.visualizer.plot_skeleton(orig_frame, SKELETON_EDGES, color='#D63D3D', label="Original", clear=False)
            
            # Plot Denoised (if available)
            if self.denoised_data is not None:
                if frame_idx < len_denoised:
                    denoised_frame = self.denoised_data[frame_idx]
                    self.visualizer.plot_skeleton(denoised_frame, SKELETON_EDGES, color='#4ED2A6', label="Denoised", clear=False)
            
            self.visualizer.ax.legend()
            self.visualizer.canvas.draw()
            
            # FPS Control
            fps = int(self.slider_speed.get())
            delay = int(1000 / fps)
            
            next_frame = (frame_idx + 1) % num_frames
            self.after(delay, lambda: update_frame(next_frame)) 

        update_frame(0)

    def load_custom_encoder(self):
        file_path = filedialog.askopenfilename(filetypes=[("Model Checkpoint", "*.pth")])
        if file_path:
            print(f"[INFO] Custom Encoder: {file_path}")
            self.custom_encoder_path = file_path
            self.motion_encoder = None # Force reload
            print("[INFO] Encoder path updated.")

    def load_custom_diffusion(self):
        file_path = filedialog.askopenfilename(filetypes=[("Model Checkpoint", "*.pth")])
        if file_path:
            print(f"[INFO] Custom Diffusion: {file_path}")
            self.custom_diffusion_path = file_path
            self.diffusion_model = None # Force reload
            print("[INFO] Diffusion path updated.")



    def load_and_visualize_result(self):
        """Loads a TRC file and displays it as the 'Denoised/Result' (Green) directly."""
        file_path = filedialog.askopenfilename(filetypes=[("TRC Files", "*.trc")])
        if file_path:
            try:
                print(f"[INFO] Visualizing result: {file_path}")
                raw_data = parse_trc_robust(file_path)
                data, _ = normalize_and_center_data(raw_data, root_idx=0)
                
                # Set as Denoised Data
                self.denoised_data = data 
                
                # If no input data is loaded, just show this as Green.
                # If input data is loaded, it will show overlay.
                # If no input data is loaded, just show this as Green.
                # If input data is loaded, it will show overlay.
                self.is_animating = True
                
                # Dynamic Auto-Scaling (From loaded result)
                all_flat = self.denoised_data.reshape(-1, 3)
                min_vals = all_flat.min(axis=0)
                max_vals = all_flat.max(axis=0)
                
                print(f"[DEBUG] Result Data Range: X[{min_vals[0]:.2f}, {max_vals[0]:.2f}], Y[{min_vals[1]:.2f}, {max_vals[1]:.2f}], Z[{min_vals[2]:.2f}, {max_vals[2]:.2f}]")
                pad = 0.5
                self.visualizer.set_plot_limits(
                    (min_vals[0]-pad, max_vals[0]+pad), 
                    (min_vals[1]-pad, max_vals[1]+pad), 
                    (min_vals[2]-pad, max_vals[2]+pad)
                )

                self.animate_comparison()
                
            except Exception as e:
                print(f"[ERROR] Failed to visualize: {e}")

    def load_models_if_needed(self):
        if self.diffusion_model is None:
            print("[INFO] Loading Diffusion Models...")
            
            # Constants from Inference.py (or from config if they were there)
            # kept local vars for clarity unless in config
            IN_CHANNELS = 3       
            # CONTEXT_DIM in config
            MODEL_CHANNELS = 128  
            TIME_EMB_DIM = 256
            # NUM_HEADS in config
            # NUM_LAYERS_ENCODER in config

            self.motion_encoder = MotionEncoder(hidden_dim=CONTEXT_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS_ENCODER).to(self.device)
            self.diffusion_model = ConditionalDiffusionModel(in_channels=IN_CHANNELS, out_channels=IN_CHANNELS, 
                                                        model_channels=MODEL_CHANNELS, context_dim=CONTEXT_DIM, 
                                                        time_emb_dim=TIME_EMB_DIM).to(self.device)
            self.noise_scheduler = NoiseScheduler(num_train_timesteps=1000)

            # Checkpoints
            # Checkpoints
            ENCODER_PATH = getattr(self, 'custom_encoder_path', CHECKPOINT_ENCODER)
            DIFFUSION_PATH = getattr(self, 'custom_diffusion_path', CHECKPOINT_DIFFUSION)
            
            # Try to find best checkpoints
            if not os.path.exists(ENCODER_PATH):
                print(f"[WARN] {ENCODER_PATH} not found. Trying to find any checkpoint in models/...")
            
            # Load anyway (might fail if no file)
            try:
                if os.path.exists(ENCODER_PATH):
                    self.motion_encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=self.device))
                else:
                    print(f"[WARN] Encoder checkpoint not found: {ENCODER_PATH}")

                if os.path.exists(DIFFUSION_PATH):
                    self.diffusion_model.load_state_dict(torch.load(DIFFUSION_PATH, map_location=self.device))
                else:
                    print(f"[WARN] Diffusion checkpoint not found: {DIFFUSION_PATH}")

                print("[INFO] Models loaded (weights might be missing if path incorrect).")
            except Exception as e:
                print(f"[ERROR] Loading weights failed: {e}")

            self.motion_encoder.eval()
            self.diffusion_model.eval()

    def run_denoising_thread(self):
        threading.Thread(target=self.denoise_motion, daemon=True).start()

    def denoise_motion(self):
        if self.loaded_file_path is None:
            print("[ERROR] No file loaded.")
            return

        try:

            self.load_models_if_needed()
            
            if self.diffusion_model is None or self.motion_encoder is None:
                 print("[ERROR] Models could not be loaded. Check checkpoint paths.")
                 tk.messagebox.showerror("Error", "Models could not be loaded. Please check console.")
                 return
            
            print("[INFO] Preprocessing Data...")
            # Use utility to load windows
            # Assuming params
            TRAINING_FPS = 30     
            WINDOW_SIZE = 60      
            OVERLAP = 10          
            
            raw_data = parse_trc_robust(self.loaded_file_path)
            num_frames_orig = raw_data.shape[0]
            effective_window_size = min(WINDOW_SIZE, num_frames_orig)
            
            motion_data_tensor, metadata = load_data_from_trc(
                file_path=self.loaded_file_path, target_fps=TRAINING_FPS, window_size=effective_window_size, 
                right_asi_idx=0, left_asi_idx=1, sacrum_idx=2, overlap=OVERLAP
            )
            motion_data_tensor = motion_data_tensor.to(self.device)
            
            strength = self.slider_strength.get()
            print(f"[INFO] Denoising with Strength: {strength:.2f}")
            
            # --- Diffusion Loop (Adapted from Inference.py) ---
            B_batch = motion_data_tensor.shape[0]
            reconstructed_windows = []
            
            start_timestep = int(self.noise_scheduler.num_train_timesteps * strength)
            start_timestep = min(start_timestep, self.noise_scheduler.num_train_timesteps - 1)
            
            for b_idx in range(B_batch):
                print(f"Processing Window {b_idx+1}/{B_batch}...", end="\r")
                current_window = motion_data_tensor[b_idx:b_idx+1] # (1, T, M, C)
                
                with torch.no_grad():
                    context = self.motion_encoder(current_window)
                    
                    if strength >= 1.0:
                        x_t = torch.randn_like(current_window).to(self.device)
                    else:
                        noise = torch.randn_like(current_window).to(self.device)
                        timesteps_start = torch.full((1,), start_timestep, device=self.device, dtype=torch.long)
                        x_t = self.noise_scheduler.add_noise(current_window, noise, timesteps_start)
                    
                    relevant_timesteps = [t for t in self.noise_scheduler.timesteps if t <= start_timestep]
                    
                    for t_val in relevant_timesteps:
                        t = torch.full((1,), t_val, device=self.device, dtype=torch.long)
                        predicted_noise = self.diffusion_model(x_t, t, context)
                        x_t = self.noise_scheduler.step(predicted_noise, t, x_t).prev_sample
                        
                    reconstructed_windows.append(x_t)
            
            print("\n[INFO] Stitching Windows...")
            
            # --- Stitching ---
            T, M, C = reconstructed_windows[0].shape[1:]
            stride = T - OVERLAP
            num_windows = len(reconstructed_windows)
            total_len = (num_windows - 1) * stride + T
            
            final_pred = torch.zeros((total_len, M, C), device=self.device)
            weights = torch.zeros((total_len, M, C), device=self.device)
            
            fade_in = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, OVERLAP, device=self.device)))
            fade_out = 0.5 * (1 + torch.cos(torch.linspace(0, math.pi, OVERLAP, device=self.device)))
            
            window_curve = torch.ones(T, device=self.device)
            window_curve[:OVERLAP] = fade_in
            window_curve[-OVERLAP:] = fade_out
            window_curve = window_curve.view(T, 1, 1).expand(T, M, C)
            
            for i, w in enumerate(reconstructed_windows):
                w_data = w.squeeze(0)
                start_idx = i * stride
                end_idx = start_idx + T
                final_pred[start_idx:end_idx] += w_data * window_curve
                weights[start_idx:end_idx] += window_curve
                
            weights[weights == 0] = 1.0
            full_reconstructed = final_pred / weights
            
            # Update State
            self.denoised_data = full_reconstructed.cpu().numpy()
            print("[SUCCESS] Denoising Complete! Visualization updated.")

            # Save Result
            try:
                os.makedirs("results", exist_ok=True)
                save_path = "results/denoised_motion.npy"
                np.save(save_path, self.denoised_data)
                print(f"[INFO] Saved denoised motion to {save_path}")
                tk.messagebox.showinfo("Success", f"Motion saved to {save_path}")
            except Exception as e:
                print(f"[ERROR] Could not save file: {e}")
            
        except Exception as e:
            print(f"[ERROR] Denoising failed: {e}")
            import traceback
            traceback.print_exc()

    def start_training_thread(self):
        print("[INFO] Starting Diffusion Training Logic (Placeholder)...")
        # In reality, import training.py
        pass
