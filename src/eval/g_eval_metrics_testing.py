import os
import math
import sys
from openai import AzureOpenAI
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from eval_prompts.GEVAL_ASP_CON import GEVAL_ASP_CON_PROMPT
from eval_prompts.GEVAL_ASP_REL import GEVAL_ASP_REL_PROMPT

load_dotenv()

# ==========================================
# 1. Configuration
# ==========================================
AZURE_ENDPOINT = os.getenv("GPT_4_1_MINI_AZURE_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("GPT_4_1_MINI_AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("GPT_4_1_MINI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT_NAME = os.getenv("GPT_4_1_MINI_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# ==========================================
# 2. The Metric Class
# ==========================================
class GEvalScientificMetric:
    def __init__(self):
        missing = []
        if not AZURE_ENDPOINT:
            missing.append("GPT_4_1_MINI_AZURE_ENDPOINT")
        if not API_KEY or "your-key-here" in API_KEY:
            missing.append("GPT_4_1_MINI_AZURE_OPENAI_API_KEY")
        if not API_VERSION:
            missing.append("GPT_4_1_MINI_API_VERSION")
        if not DEPLOYMENT_NAME:
            missing.append("GPT_4_1_MINI_DEPLOYMENT_NAME")
        if missing:
            raise ValueError(f"Missing Azure OpenAI configuration: {', '.join(missing)}")
            
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=API_KEY,
            api_version=API_VERSION,
        )
    
    def evaluate_aspect(self, aspect_name: str, gold_summary: str, generated_summary: str, dimension: str) -> float:
        """
        Calculates the weighted Alignment Score (1-5) using Logprobs.
        """
        if dimension == "consistency":
            prompt_template = GEVAL_ASP_CON_PROMPT
        elif dimension == "relevance":
            prompt_template = GEVAL_ASP_REL_PROMPT
        else:
            raise ValueError("Dimension must be either 'consistency' or 'relevance'.")
        
        prompt = prompt_template.format(
            aspect=aspect_name,
            reference=gold_summary,
            summary=generated_summary
        )

        try:
            # Call Azure with Logprobs enabled
            response = self.client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1,     # We only need the score digit
                logprobs=True,    # REQUIRED
                top_logprobs=5    # Capture probability spread
            )

            if not response.choices or not response.choices[0].logprobs:
                print("[Error] No logprobs returned. Check deployment settings.")
                return 0.0

            top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            
            score_sum = 0.0
            total_prob = 0.0
            
            # Calculate Expected Value
            for token_data in top_logprobs:
                token_str = token_data.token.strip() # Remove leading space (e.g., " 5")
                
                if token_str.isdigit():
                    val = int(token_str)
                    if 1 <= val <= 5:
                        prob = math.exp(token_data.logprob)
                        score_sum += prob * val
                        total_prob += prob

            if total_prob > 0:
                return score_sum / total_prob
            
            # Fallback: exact match if logprobs failed to capture digits
            content = response.choices[0].message.content.strip()
            return float(content) if content.isdigit() else 0.0

        except Exception as e:
            print(f"API Error: {e}")
            return 0.0

# ==========================================
# 3. Usage Example
# ==========================================
if __name__ == "__main__":
    evaluator = GEvalScientificMetric()

    aspect = "Efficiency"
    
    # ---------------------------------------------------------
    # Context: A Paper about a new Efficient Transformer (FlashAttention)
    # ---------------------------------------------------------
    gold_summary = (
        "This paper introduces FlashAttention, an IO-aware exact attention algorithm. "
        "It uses tiling to reduce memory reads/writes between GPU HBM and SRAM. "
        "The method achieves 2-4x speedup over standard attention and reduces memory complexity from quadratic to linear."
    )

    # ---------------------------------------------------------
    # Case 1: High Quality Summary (Should score ~4.5 - 5.0)
    # ---------------------------------------------------------
    good_summary = (
        "The authors propose FlashAttention to optimize attention mechanisms by accounting for IO costs. "
        "By utilizing tiling techniques to manage memory access between HBM and SRAM, they achieve significant speedups "
        "and linear memory complexity compared to standard attention."
    )
    
    print("--- Evaluation 1: High Quality ---")
    print("\n-- Aspect Relevance --")
    score_1 = evaluator.evaluate_aspect(aspect, gold_summary, good_summary, dimension="relevance")
    print(f"Score: {score_1:.4f}")
    print("\n-- Aspect Consistency --")
    score_1c = evaluator.evaluate_aspect(aspect, gold_summary, good_summary, dimension="consistency")
    print(f"Score: {score_1c:.4f}")

    # ---------------------------------------------------------
    # Case 2: Low Quality / Missing Key Aspect (Should score ~2.0 - 3.0)
    # ---------------------------------------------------------
    bad_summary = (
        "This paper discusses a new attention mechanism called FlashAttention. "
        "It is very fast and runs on GPUs. It helps with training large language models."
        # MISSING: Tiling, HBM/SRAM details, Memory complexity reduction
    )

    print("\n--- Evaluation 2: Low Quality ---")
    print("\n-- Aspect Relevance --")
    score_2 = evaluator.evaluate_aspect(aspect, gold_summary, bad_summary, dimension="relevance")
    print(f"Score: {score_2:.4f}")
    print("\n-- Aspect Consistency --")
    score_2 = evaluator.evaluate_aspect(aspect, gold_summary, bad_summary, dimension="consistency")
    print(f"Score: {score_2:.4f}")