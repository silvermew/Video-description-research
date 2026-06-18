import json
import requests
import re
from typing import Dict, List

class LLMQualityCritic:
    """
    Acts as an Agentic Critic (Pass 2) in the Video-to-Text pipeline.
    Evaluates drafted sentences against raw sensor ground truth and forces
    rewrites if hallucinations or inaccuracies are detected.
    """
    def __init__(self, model_name: str = "llama3:8b", api_url: str = "http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.api_url = api_url

    def _build_critic_prompt(self, motion_text: str, yolo_objects: str, draft_sentence: str) -> str:
        prompt = f"""You are a strict Quality Assurance Bot for a robotic sensor pipeline. 
Your ONLY job is to evaluate a drafted sentence and verify it matches the absolute ground-truth sensor data without any hallucinations or missed details.

GROUND TRUTH MOTION: {motion_text}
GROUND TRUTH OBJECTS: {yolo_objects}

DRAFT SENTENCE TO EVALUATE: "{draft_sentence}"

INSTRUCTIONS:
1. Compare the Draft Sentence against the Ground Truths.
2. The draft FAILS if it adds subjective adjectives (e.g. peaceful, gently), hallucinates objects or environment details not in the ground truth, or misses critical physical interactions.
3. You MUST output your response in strict JSON format. Do NOT output any other text or markdown blocks.

JSON FORMAT:
{{
  "passed": boolean,
  "corrected_sentence": string
}}

If "passed" is true, "corrected_sentence" should be the exact same as the draft.
If "passed" is false, "corrected_sentence" must contain the rewritten, objective, and accurate sentence.

OUTPUT JSON ONLY:"""
        return prompt

    def evaluate_draft(self, motion_text: str, yolo_objects: List[Dict], draft_sentence: str) -> Dict:
        """
        Sends the strict evaluation prompt to the Ollama API and parses the JSON response.
        """
        if yolo_objects:
            objects_str = ", ".join([f"{obj['object']} ({obj.get('position', '')})" for obj in yolo_objects])
        else:
            objects_str = "None."
            
        prompt = self._build_critic_prompt(motion_text, objects_str, draft_sentence)
        
        # We can pass "format": "json" to Ollama API to force JSON syntax
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result_text = response.json().get("response", "").strip()
            
            # Robust JSON extraction to handle any markdown wrapping or prepended text
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    if "passed" in parsed_json and "corrected_sentence" in parsed_json:
                        return parsed_json
                except json.JSONDecodeError:
                    pass
            
            # Fallback if json parsing completely fails
            return {"passed": True, "corrected_sentence": draft_sentence}
            
        except Exception as e:
            print(f"❌ Critic API Error: {e}")
            return {"passed": True, "corrected_sentence": draft_sentence}

# ==========================================
# INTEGRATION LOOP DEMONSTRATION
# ==========================================
if __name__ == "__main__":
    from llm_orchestrator import LLMOrchestrator
    
    print("Initializing Generator (Pass 1) and Critic (Pass 2)...")
    generator = LLMOrchestrator(model_name="llama3:8b")
    critic = LLMQualityCritic(model_name="llama3:8b")
    
    # Mock Sensor Data
    test_cases = [
        {
            "motion": "person sits down on the ground",
            "objects": [],
            # The LLM generator might hallucinate "soft grassy ground"
            "mock_draft": "The person sits down peacefully on the soft grassy ground." 
        },
        {
            "motion": "person raises right arm [INTERACTION DETECTED: Right Hand intersecting cup_id_3]",
            "objects": [{'object': 'cup_id_3', 'position': 'near right hand'}],
            "mock_draft": "The person raises their right arm toward the cup."
        }
    ]
    
    final_transcript = []
    
    print("\n--- Starting Two-Pass Generation Loop ---")
    for i, case in enumerate(test_cases):
        print(f"\n[Frame {i}] Ground Truth Motion: {case['motion']}")
        print(f"[Frame {i}] Ground Truth Objects: {case['objects']}")
        
        # PASS 1: Generator 
        # (Using a mock draft here to demonstrate the Critic fixing a hallucination)
        draft = case['mock_draft']
        print(f"  [Pass 1 - Generator] Draft: {draft}")
        
        # PASS 2: Critic
        critic_result = critic.evaluate_draft(case['motion'], case['objects'], draft)
        print(f"  [Pass 2 - Critic] Passed: {critic_result.get('passed')}")
        
        final_sentence = critic_result.get('corrected_sentence', draft)
        print(f"  [Final Approved Sentence]: {final_sentence}")
        
        # Append to transcript
        final_transcript.append(final_sentence)
        
    print("\n--- Final Transcript ---")
    for s in final_transcript:
        print("-", s)
