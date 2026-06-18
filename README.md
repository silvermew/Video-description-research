# MotionToMotion: Video-to-Text Narrative Pipeline

MotionToMotion is an advanced, end-to-end computer vision and natural language processing pipeline. It takes raw MP4 video, extracts high-precision 3D human motion, analyzes spatial object interactions, and synthesizes a coherent, contextual narrative of the scene using a local LLM.

## Architecture

The system is built on a robust multi-stage architecture:

1. **Motion Extraction (Video to 3D TRC)**
   - **2D Pose Estimation**: Uses YOLOv8-Pose to extract initial 2D keypoints.
   - **3D Temporal Lifting**: Leverages a frozen **MotionBERT** (DSTformer) backbone to lift 2D poses to 3D, providing excellent occlusion resistance through temporal context.
   - **Custom TRC Head**: A trained linear projection head maps the 17-joint MotionBERT output into a dense 64-marker `TRC` kinematic format.

2. **Spatial & Object Context**
   - **YOLO-World / YOLOv10**: Analyzes the scene to detect objects and their proximity to the human subjects, filtering out background noise.
   - **Interaction Detection**: Heuristics detect interactions between the human and nearby objects.

3. **Motion-to-Text Captioning**
   - **PyTorch Transformer**: A custom-trained encoder-decoder Transformer (`MotionCaptioner`) translates the 3D kinematic TRC data into initial text descriptions.

4. **Narrative Orchestration**
   - **Llama 3 (via Ollama)**: The `LLMOrchestrator` fuses the raw motion captions with the spatial object context to generate a fluid narrative.
   - **LLM Quality Critic**: A secondary LLM pass ensures the final narrative is accurate, natural, and grounded in the data.

## Key Files

*   `ultimate_pipeline.py`: The master script that runs the entire end-to-end pipeline.
*   `mp4_to_trc_motionbert.py`: The inference script for lifting MP4 video to 3D TRC format.
*   `motionBert_train_head.py`: The script used to train the custom TRC projection head on top of the MotionBERT backbone.
*   `llm_orchestrator.py` & `llm_critic.py`: Manage the integration with the local Llama 3 instance.
*   `yolov10_spatial_context.py`: Handles spatial awareness and proximity gating for object detection.

## Setup & Usage

1.  **Dependencies**: Requires `torch`, `ultralytics`, `mediapipe`, `scipy`, `pandas`, and a local instance of Ollama running `llama3:8b`.
2.  **Pretrained Weights**: The pipeline requires the MotionBERT backbone (`best_epoch.bin`) and the custom TRC head (`motionbert_trc_head_best.pth`).
3.  **Run Pipeline**:
    ```bash
    python3 ultimate_pipeline.py cam-001.mp4
    ```

## Note on Repository Content
Due to size constraints, datasets, trained model weights (`.pt`, `.pth`, `.bin`), large video files, and generated output artifacts are excluded from this repository.

## Acknowledgements & Citations

This project builds upon the incredible work of the open-source computer vision community. We would like to acknowledge and cite the following projects:

*   **MotionBERT**: The core 3D lifting backbone of our motion extraction pipeline.
    > Zhu, W., Ma, X., Liu, Z., Liu, L., Wu, W., & Wang, Y. (2023). *MotionBERT: A Unified Perspective on Learning Human Motion Representations*. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). [GitHub](https://github.com/Walter0807/MotionBERT)

*   **Ultralytics YOLO**: Used for 2D human pose estimation (YOLOv8-Pose) and spatial object context (YOLOv10 / YOLO-World).
    > Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLO*. [GitHub](https://github.com/ultralytics/ultralytics)
