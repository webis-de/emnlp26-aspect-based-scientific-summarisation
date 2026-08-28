import sys
import os

# Ensure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from language_engine import LanguageEngine, VLLM_CONFIG
import io_agents as i_o
from prompts import VerifierPrompts


class VerifierAgent:
	def __init__(self, engine: LanguageEngine):
		self.engine = engine

	def run(
		self,
		aspect_name: str,
		summary_text: str,
		evidence_set: list[i_o.EvidenceItem],
		plan: i_o.PlannerOutput = None,
		aspect_definition: str = None,
		acceptance_criteria: list[str] = None,
	) -> i_o.VerifierOutput:
		"""
		Executes the verification phase.

		Args:
			aspect_name: The target aspect for verification.
			summary_text: The generated summary to verify.
			evidence_set: The evidence set used by the Writer.
			plan: (Optional) Output from the Planner.
			aspect_definition: (Optional) Override for aspect definition.
			acceptance_criteria: (Optional) Override for acceptance criteria.
		"""

		if plan is not None:
			if aspect_definition is None:
				aspect_definition = plan.aspect_definition
			if acceptance_criteria is None:
				acceptance_criteria = plan.acceptance_criteria

		verifier_input = i_o.VerifierInput(
			aspect_name=aspect_name,
			summary_text=summary_text,
			evidence_set=evidence_set,
			aspect_definition=aspect_definition,
			acceptance_criteria=acceptance_criteria,
		)

		system_prompt, user_prompt = VerifierPrompts.render(verifier_input)

		mode = "Standard (Guided)"
		if plan is None:
			mode = "Ablated (Unguided)"

		print(f"\n--- [Verifier] Running in [{mode}] Mode ---")
		print(f"    Aspect: '{aspect_name}'")

		output = self.engine.generate_structured(
			user_prompt=user_prompt,
			system_prompt=system_prompt,
			pydantic_model=i_o.VerifierOutput,
			enable_thinking=True,
		)

		return output


if __name__ == "__main__":
	# 1. Setup Configuration
	config = VLLM_CONFIG.copy()

	# 2. Model Selection
	config["model_path"] = "Qwen/Qwen3-4B"
	config["tensor_parallel_size"] = 1
	config["gpu_memory_utilization"] = 0.9
	# config["max_tokens"] = 1024
	# config["max_model_len"] = 8192 / 2  # Half-length context for verification

	# 3. Initialize Engine
	print(f"Initializing vLLM Engine with {config['model_path']}...")
	engine = LanguageEngine(config)
	verifier = VerifierAgent(engine)

	# --- Shared Setup ---
	aspect = "Methodology"
	plan = i_o.PlannerOutput(
		aspect_definition="Implementation details and procedures used in the study.",
		extraction_cues=["Methods", "we implement", "architecture"],
		acceptance_criteria=[
			"Must mention the model architecture",
			"Must mention the training data",
			"Must mention any thresholds or hyperparameters",
		],
	)

	# --- COMMON PATHWAY 1: ACCEPT ---
	print("\n" + "=" * 60)
	print("COMMON 1: Accept (Aligned, Faithful, Covered)")
	print("=" * 60)

	evidence_accept = [
			i_o.EvidenceItem(
				span="We fine-tune a BART-large model using 50k PubMed abstracts. Training is performed for 3 epochs with a learning rate of 3e-5.",
			),
			i_o.EvidenceItem(
				span="We discard samples with confidence below 0.7.",
			),
	]
	summary_accept = (
		"The method fine-tunes BART-large on 50k PubMed abstracts and applies a 0.7 confidence threshold. "
		"Training uses three epochs at 3e-5 learning rate."
	)

	try:
		result = verifier.run(
			aspect_name=aspect,
			summary_text=summary_accept,
			evidence_set=evidence_accept,
			plan=plan,
		)
		print(result.model_dump())
	except Exception as e:
		print(f"Error: {e}")

	# --- COMMON PATHWAY 2: REWRITE WITH SAME EVIDENCE ---
	print("\n" + "=" * 60)
	print("COMMON 2: Revise Summary (Rewrite with Same Evidence)")
	print("=" * 60)

	evidence_rewrite = [
			i_o.EvidenceItem(
				span="We use a Transformer-base architecture with 12 layers. The model is trained on the ACLSum dataset.",
			),
			i_o.EvidenceItem(
				span="Training uses a 0.2 dropout rate.",
			),
	]
	summary_rewrite = (
		"The study trains a 24-layer Transformer on the ACLSum dataset with 0.2 dropout."
	)

	try:
		result = verifier.run(
			aspect_name=aspect,
			summary_text=summary_rewrite,
			evidence_set=evidence_rewrite,
			plan=plan,
		)
		print(result.model_dump())
	except Exception as e:
		print(f"Error: {e}")

	# --- EDGE CASE 1: SPARSE (MISSING CRITERIA) ---
	print("\n" + "=" * 60)
	print("EDGE 1: Sparse Evidence (Extract More)")
	print("=" * 60)

	evidence_sparse = [
			i_o.EvidenceItem(
				span="We fine-tune a Longformer model on the PMC dataset.",
			),
	]
	summary_sparse = "We fine-tune Longformer on PMC for the method." 

	try:
		result = verifier.run(
			aspect_name=aspect,
			summary_text=summary_sparse,
			evidence_set=evidence_sparse,
			plan=plan,
		)
		print(result.model_dump())
	except Exception as e:
		print(f"Error: {e}")

	# --- EDGE CASE 2: OFF-ASPECT EVIDENCE ---
	print("\n" + "=" * 60)
	print("EDGE 2: Off-Aspect Evidence (Revise Evidence)")
	print("=" * 60)

	evidence_off_aspect = [
			i_o.EvidenceItem(
				span="The model achieves 92.3% ROUGE-L on the test set. We observe consistent gains over baselines.",
			),
	]
	summary_off_aspect = "The method achieves 92.3% ROUGE-L and outperforms baselines."

	try:
		result = verifier.run(
			aspect_name=aspect,
			summary_text=summary_off_aspect,
			evidence_set=evidence_off_aspect,
			plan=plan,
		)
		print(result.model_dump())
	except Exception as e:
		print(f"Error: {e}")
