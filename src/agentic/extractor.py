import sys
import os
import json

# Ensure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from language_engine import LanguageEngine, VLLM_CONFIG
import io_agents as i_o
from prompts import ExtractorPrompts

class ExtractorAgent:
    def __init__(self, engine: LanguageEngine):
        self.engine = engine

    def run(self, 
            doc_text: str, 
            aspect_name: str, 
            aspect_definition: str = None, 
            extraction_cues: list[str] = None,
            correction_instruction: str = None) -> i_o.ExtractorOutput:
        """
        Executes the extraction phase.
        
        Args:
            doc_text: The full text of the document.
            aspect_name: The target aspect to extract.
            plan: (Optional) Output from the Planner. If None, runs in 'Ablated' mode (Zero-Shot).
            feedback: (Optional) Error description from the Verifier. If present, runs in 'Refinement' mode.
        """
        
        # 1. Construct Input Object
        # We map the Plan -> Input fields here.
        extractor_input = i_o.ExtractorInput(
            document_text=doc_text,
            aspect_name=aspect_name,
            
            # --- ABLATION LOGIC ---
            # If plan exists, use the definition/cues. 
            # If None, the prompt template will trigger the "Infer definition" fallback.
            aspect_definition=aspect_definition,
            extraction_cues=extraction_cues,
            
            # --- REFINEMENT LOGIC ---
            # If feedback exists, the prompt template will inject the "CRITICAL FEEDBACK" block.
            correction_instruction=correction_instruction
        )

        # 2. Render Prompt 
        system_prompt, user_prompt = ExtractorPrompts.render(extractor_input)

        # Logging for transparency
        mode = "Standard (Guided)"
        if not aspect_definition and not extraction_cues: mode = "Ablation (no Planner)"
        if correction_instruction: mode = "Refinement (Feedback Loop)"
        
        print(f"\n--- [Extractor] Running in [{mode}] Mode ---")
        print(f"    Aspect: '{aspect_name}'")

        # 3. Inference (Structured Output)
        # The engine enforces the i_o.ExtractorOutput schema (evidence_set list)
        output = self.engine.generate_structured(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            pydantic_model=i_o.ExtractorOutput,
            enable_thinking=False 
        )

        return output
    

# Just for testing, later we use the data_utils module to load JSONL files
def flatten_source_text(record: dict) -> str:
    """
    Converts the structured 'source_text' dictionary into a single Markdown-formatted string.
    """
    source_dict = record.get("source_text", {})
    if isinstance(source_dict, str): return source_dict # Handle edge case where it's already a string
    
    flattened_parts = []
    for section_header, content in source_dict.items():
        # Clean up header and content
        header = section_header.strip()
        text = content.strip() if content else ""
        
        # Format as Markdown Header + Text
        flattened_parts.append(f"## {header}\n{text}")
    
    return "\n\n".join(flattened_parts)

# --- RUNNABLE PROTOTYPE ---
if __name__ == "__main__":
    # 1. Setup Configuration
    config = VLLM_CONFIG.copy()
    
    # 2. Model Selection (High Logic Required)
    # Extraction requires understanding nuance, so we stick to the 32B model.
    config = VLLM_CONFIG.copy()
    config["model_path"] = "Qwen/Qwen3-32B" 
    config["tensor_parallel_size"] = 2 
    config["gpu_memory_utilization"] = 0.95
    config["max_tokens"] = 32000
    config["max_model_len"] = 70000
    config["temperature"] = 0.1

    # 3. Initialize Engine
    print(f"Initializing vLLM Engine with {config['model_path']}...")
    engine = LanguageEngine(config)
    extractor = ExtractorAgent(engine)

    # Input Data 
    with open("data/test.jsonl", "r", encoding="utf-8") as f:
        record = json.loads(f.readline())

    # planner output data/planner_output_debug.jsonl
    with open("data/planner_output_debug.jsonl", "r", encoding="utf-8") as f:
        plan_record = json.loads(f.readline())

    aspect_definition = plan_record.get("aspect_definition")
    extraction_cues = plan_record.get("extraction_cues")


    full_doc_text = flatten_source_text(record)
    aspect = "Methodology"


    # --- SCENARIO 1: Full Pipeline (With Mock Plan) ---
    print("\n" + "="*60)
    print("SCENARIO 1: Full Pipeline (Plan Provided)")
    print("="*60)

    # No feedback for first run
    try:
        result = extractor.run(full_doc_text, aspect, aspect_definition=aspect_definition, extraction_cues=extraction_cues)
        print(json.dumps(result.model_dump(), indent=2))
        with open("data/extractor_output_1.jsonl", "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result.model_dump()) + "\n")
    except Exception as e:
        print(f"Error: {e}")

    # --- SCENARIO 2: Refinement (With Feedback) ---
    print("\n" + "="*60)
    print("SCENARIO 2: Refinement (Fixing a mistake)")
    print("="*60)
    
    # Simulating the Verifier rejecting the first attempt
    correction_instruction = "The extraction missed the specific threshold value used for discarding documents."
    
    try:
        result = extractor.run(full_doc_text, aspect, aspect_definition=aspect_definition, extraction_cues=extraction_cues, correction_instruction=correction_instruction)
        print(json.dumps(result.model_dump(), indent=2))
        with open("data/extractor_output_2.jsonl", "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result.model_dump()) + "\n")
    except Exception as e:
        print(f"Error: {e}")