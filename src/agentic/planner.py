import sys
import os
import json

# Ensure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from language_engine import LanguageEngine, VLLM_CONFIG
import io_agents as i_o
from prompts import PlannerPrompts

class PlannerAgent:
    def __init__(self, engine: LanguageEngine):
        self.engine = engine

    def run(self, doc_text: str, aspect_name: str, text_type: str = "Scientific Paper") -> i_o.PlannerOutput:
        """
        Executes the planning phase: Deconstructs the aspect into extraction cues and criteria.
        """
        # 1. Prepare Input Data (Pydantic)
        planner_input = i_o.PlannerInput(
            document_text=doc_text,
            aspect_name=aspect_name,
            text_type=text_type
        )

        # 2. Render Prompt (Jinja2)
        system_prompt, user_prompt = PlannerPrompts.render(planner_input)

        print(f"\n--- [Planner] Generating Plan for Aspect: '{aspect_name}' ---")
        
        # 3. Inference (Structured Output)
        plan = self.engine.generate_structured(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            pydantic_model=i_o.PlannerOutput,
            enable_thinking=True
        )

        return plan

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
    config["model_path"] = "Qwen/Qwen3-32B" 
    config["tensor_parallel_size"] = 2 
    config["gpu_memory_utilization"] = 0.95
    config["max_tokens"] = 4096
    config["max_model_len"] = 70000
    config["temperature"] = 0.1

    # 2. Initialize Engine
    print(f"Initializing vLLM Engine with {config['model_path']}...")
    engine = LanguageEngine(config)
    planner = PlannerAgent(engine)

    # 3. Load Your JSON Record (for testing)
    with open("data/test.jsonl", "r", encoding="utf-8") as f:
        line = f.readline()
        test_record = json.loads(line)

    # 4. Pre-process Data
    # Flatten the text
    full_doc_text = flatten_source_text(test_record)
    # Extract type
    doc_type = test_record.get("source_type", "Scientific Paper")
    
    # 5. Define Aspect (Taking one from your summary keys)
    target_aspect = "Purpose"  

    # 6. Run Planner
    try:
        print(f"Processing Document Type: {doc_type}")
        result = planner.run(full_doc_text, target_aspect, text_type=doc_type)
        
        # 7. Pretty Print Result
        print("\n" + "="*60)
        print(f"PLAN GENERATED FOR: {target_aspect}")
        print("="*60)
        print(json.dumps(result.model_dump(), indent=2))
        print("="*60)

        # #ave to file jsonl
        with open("data/planner_output_debug.jsonl", "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result.model_dump()) + "\n")

    except Exception as e:
        print(f"Error during planning: {e}")