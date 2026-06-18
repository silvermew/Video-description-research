import json
import requests
from typing import List, Dict

class LLMOrchestrator:
    """
    Orchestrates the synthesis of motion descriptions and spatial object data 
    into a single fluent narrative sentence using a local LLM via Ollama.
    """
    
    def __init__(self, model_name: str = "llama3:8b", api_url: str = "http://localhost:11434/api/generate"):
        """
        Initializes the LLM orchestrator.
        
        Args:
            model_name (str): The name of the model in Ollama.
            api_url (str): The endpoint for Ollama's generate API.
        """
        self.model_name = model_name
        self.api_url = api_url
        self.context_memory: List[str] = []
        self.max_memory = 2 # Store only the last 2 generated sentences

    def _build_prompt(self, motion_text: str, yolo_objects: List[Dict]) -> str:
        """
        Constructs the strict prompt for the LLM based on current context.
        """
        # Format YOLO objects
        if yolo_objects:
            objects_str = ", ".join([f"{obj['object']} ({obj['position']})" for obj in yolo_objects])
        else:
            objects_str = "None."

        prompt = f"""You are an automated, clinical robotic sensor. Your ONLY job is to merge a raw motion log and an object log into a single, highly objective sentence.

STRICT RULES:

Use clinical, literal language only.

DO NOT use adjectives like 'peaceful', 'serene', 'effortlessly', or 'gentle'.

DO NOT hallucinate background details like 'grassy' unless explicitly stated in the Object Log.

State only the physical truth.

EXAMPLE 1:
Motion: 'Person sits down on the ground.'
Objects: 'None.'
Output: 'The person sits down on the ground.'

EXAMPLE 2:
Motion: 'Person raises right arm.'
Objects: 'Cup at right hand.'
Output: 'The person raises their right arm toward the cup.'

Current Input:
Motion: '{motion_text}'
Objects: '{objects_str}'
Output:"""
        return prompt

    def generate_narrative(self, motion_text: str, yolo_objects: List[Dict]) -> str:
        """
        Sends the gathered context to the local Ollama LLM and updates memory.
        
        Args:
            motion_text (str): The raw text output from the PyTorch motion captioner.
            yolo_objects (list): List of dictionaries from the YOLO spatial detector.
            
        Returns:
            str: The synthesized narrative sentence.
        """
        prompt = self._build_prompt(motion_text, yolo_objects)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status() # Raise an exception for bad status codes
            
            result = response.json()
            generated_sentence = result.get("response", "").strip()
            
            # Clean up potential LLM bad habits (e.g., quotes)
            if generated_sentence.startswith('"') and generated_sentence.endswith('"'):
                generated_sentence = generated_sentence[1:-1]
            
            # Update memory
            self.context_memory.append(generated_sentence)
            if len(self.context_memory) > self.max_memory:
                self.context_memory.pop(0)
                
            return generated_sentence
            
        except requests.exceptions.RequestException as e:
            print(f"❌ LLM API Error: Failed to connect to Ollama at {self.api_url}. Is it running?")
            print(f"Details: {e}")
            return f"[{motion_text}] (API Error)"


# ==========================================
# MOCK MAIN ROLLING LOOP
# ==========================================
if __name__ == "__main__":
    import time
    
    print("Initializing LLM Orchestrator...")
    orchestrator = LLMOrchestrator(model_name="llama3:8b")
    
    # Mock Video Metadata
    total_frames = 270 # 9 seconds at 30 fps
    chunk_size = 90    # 3 seconds per chunk
    
    final_transcript = []
    
    print("\nStarting Video Processing Loop...")
    for start_frame in range(0, total_frames, chunk_size):
        end_frame = start_frame + chunk_size
        print(f"\n--- Processing Chunk: Frames {start_frame} to {end_frame} ---")
        
        # 1. Mock PyTorch MotionCaptioner Output
        # In reality, you pass frames to MediaPipe -> TRC -> PyTorch Model
        if start_frame == 0:
            mock_motion = "a person walks forward and stops"
            mock_yolo = [{'object': 'chair', 'position': 'near right of human'}]
        elif start_frame == 90:
            mock_motion = "a person sits down"
            mock_yolo = [{'object': 'chair', 'position': 'near exactly at human'}, 
                         {'object': 'laptop', 'position': 'near above human'}]
        else:
            mock_motion = "a person types on a keyboard"
            mock_yolo = [{'object': 'laptop', 'position': 'near center of human'}]
            
        print(f"  [PyTorch] Detected Motion: '{mock_motion}'")
        print(f"  [YOLOv10] Detected Objects: {len(mock_yolo)}")
        
        # 2. Call LLM Orchestrator
        print("  [LLM] Synthesizing narrative...")
        start_time = time.time()
        
        synthesized_sentence = orchestrator.generate_narrative(mock_motion, mock_yolo)
        
        elapsed = time.time() - start_time
        print(f"  [LLM] Output ({elapsed:.2f}s): {synthesized_sentence}")
        
        # 3. Append to transcript
        final_transcript.append({
            'time_range': f"{start_frame/30.0:.1f}s - {end_frame/30.0:.1f}s",
            'text': synthesized_sentence
        })
        
    print("\n=========================================")
    print("             FINAL TRANSCRIPT            ")
    print("=========================================")
    for entry in final_transcript:
        print(f"[{entry['time_range']}] {entry['text']}")
    print("=========================================")
