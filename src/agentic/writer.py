import sys
import os

# Ensure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from language_engine import LanguageEngine, VLLM_CONFIG
import io_agents as i_o
from prompts import WriterPrompts


class WriterAgent:
	def __init__(self, engine: LanguageEngine):
		self.engine = engine

	def run(
		self,
		aspect_name: str,
		evidence_set: list[i_o.EvidenceItem],
		summary_length: str,
		plan: i_o.PlannerOutput = None,
		aspect_definition: str = None,
		acceptance_criteria: list[str] = None,
		feedback: str = None,
		prior_summary: str = None,
	) -> i_o.WriterOutput:
		"""
		Executes the writing phase.

		Args:
			aspect_name: The target aspect to summarize.
			evidence_set: Evidence extracted by the Extractor.
			summary_length: Length instruction (e.g., "200 words", "2 sentences").
			plan: (Optional) Output from the Planner.
			aspect_definition: (Optional) Override for aspect definition.
			acceptance_criteria: (Optional) Override for acceptance criteria.
			feedback: (Optional) Error description from the Verifier.
			prior_summary: (Optional) Latest summary draft for revise_summary.
		"""

		if plan is not None:
			if aspect_definition is None:
				aspect_definition = plan.aspect_definition
			if acceptance_criteria is None:
				acceptance_criteria = plan.acceptance_criteria

		writer_input = i_o.WriterInput(
			aspect_name=aspect_name,
			summary_length=summary_length,
			evidence_set=evidence_set,
			aspect_definition=aspect_definition,
			acceptance_criteria=acceptance_criteria,
			correction_instruction=feedback,
			prior_summary=prior_summary,
		)

		system_prompt, user_prompt = WriterPrompts.render(writer_input)

		mode = "Standard (Guided)"
		if plan is None:
			mode = "Ablated (Zero-Shot)"
		if feedback:
			mode = "Refinement (Feedback Loop)"

		print(f"\n--- [Writer] Running in [{mode}] Mode ---")
		print(f"    Aspect: '{aspect_name}'")
		print(f"    Summary Length: {summary_length}")

		output = self.engine.generate_structured(
			user_prompt=user_prompt,
			system_prompt=system_prompt,
			pydantic_model=i_o.WriterOutput,
			enable_thinking=False,
		)

		return output


if __name__ == "__main__":
	# 1. Setup Configuration
	config = VLLM_CONFIG.copy()

	# 2. Model Selection
	config["model_path"] = "Qwen/Qwen3-4B"
	config["tensor_parallel_size"] = 1
	config["gpu_memory_utilization"] = 0.9

	# 3. Initialize Engine
	print(f"Initializing vLLM Engine with {config['model_path']}...")
	engine = LanguageEngine(config)
	writer = WriterAgent(engine)

	# --- Shared Setup ---
	aspect = "Methodology"
	summary_length = "2 sentences"
	plan = i_o.PlannerOutput(
		aspect_definition="Implementation details and procedures used in the study.",
		extraction_cues=["Methods", "we implement", "architecture"],
		acceptance_criteria=[
			"Must mention the model architecture",
			"Must mention the training data",
			"Must mention any thresholds or hyperparameters",
		],
	)

	# --- COMMON PATHWAY 1: STANDARD (GUIDED) ---
	print("\n" + "=" * 60)
	print("COMMON 1: Standard (Guided)")
	print("=" * 60)

	evidence_guided = [
			i_o.EvidenceItem(
				span="We fine-tune a BART-large model using 50k PubMed abstracts. Training is performed for 3 epochs with a learning rate of 3e-5.",
			),
			i_o.EvidenceItem(
				span="We discard samples with confidence below 0.7.",
			),
	]

	try:
		result = writer.run(
			aspect_name=aspect,
			evidence_set=evidence_guided,
			summary_length=summary_length,
			plan=plan,
		)
		print(result.model_dump())
	except Exception as e:
		print(f"Error: {e}")

	# --- COMMON PATHWAY 2: REWRITE WITH SAME EVIDENCE ---
	print("\n" + "=" * 60)
	print("COMMON 2: Refine with Feedback")
	print("=" * 60)

	feedback = "Include all relevant hyperparameters and keep it to two sentences."
	try:
		result = writer.run(
			aspect_name=aspect,
			evidence_set=evidence_guided,
			summary_length=summary_length,
			plan=plan,
			feedback=feedback,
		)
		print(result.model_dump())
	except Exception as e:
		print(f"Error: {e}")
