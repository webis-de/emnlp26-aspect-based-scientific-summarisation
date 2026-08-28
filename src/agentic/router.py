import sys
import os
from typing import Literal

# Ensure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import io_agents as i_o # Matches your import name

class RouterAgent:
    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def route(self, 
              verifier_output: i_o.VerifierOutput, 
              current_iteration: int) -> i_o.RouterOutput:
        """
        Decides the next step based on the Verifier's judgment and iteration count.
        """
        judgment = verifier_output.judgment
        
        # --- SCENARIO 1: SUCCESS ---
        if judgment == 'accept':
            return i_o.RouterOutput(
                next_role='stop',
                termination_reason='success',
                correction_instruction=None # No instruction needed for success
            )

        # --- SCENARIO 2: MAX RETRIES REACHED ---
        if current_iteration >= self.max_retries:
            return i_o.RouterOutput(
                next_role='stop',
                termination_reason='max_retries_exceeded',
                correction_instruction=None # Stop regardless of what needs fixing
            )

        # --- SCENARIO 3: REVISION LOOP ---
        if judgment == 'revise_evidence':
            next_role = 'extractor' # Go all the way back
        elif judgment == 'revise_summary':
            next_role = 'writer'    # Just fix the prose

        # Pass the instruction through!
        # The Verifier wrote specific instructions (e.g., "Find the sample size").
        # The Router MUST carry this message to the next agent.
        return i_o.RouterOutput(
            next_role=next_role,
            correction_instruction=verifier_output.correction_instruction
        )

# --- RUNNABLE PROTOTYPE ---
if __name__ == "__main__":
    # Test Data setup
    router = RouterAgent(max_retries=3)

    print("--- TEST 1: Success Case ---")
    mock_success = i_o.VerifierOutput(
        judgment='accept',
        issue_description="None",
        action_hint="none"
    )
    decision = router.route(mock_success, current_iteration=1)
    print(decision.model_dump())

    print("\n--- TEST 2: Revision Case (Writer Fault) ---")
    mock_rewrite = i_o.VerifierOutput(
        judgment='revise_summary',
        issue_description="Hallucinated a date",
        correction_instruction="Remove the date 2023 as it does not appear in evidence.",
        action_hint="rewrite_with_same_evidence"
    )
    decision = router.route(mock_rewrite, current_iteration=1)
    print(decision.model_dump())

    print("\n--- TEST 3: Revision Case (Extractor Fault) ---")
    mock_reextract = i_o.VerifierOutput(
        judgment='revise_evidence',
        issue_description="Missing sample size",
        correction_instruction="Find the specific N number for the user study.",
        action_hint="extract_additional_evidence"
    )
    decision = router.route(mock_reextract, current_iteration=1)
    print(decision.model_dump())

    print("\n--- TEST 4: Max Retries Fail ---")
    # Even though judgment is 'revise', we should STOP because iter=3 (>= max_retries)
    decision = router.route(mock_reextract, current_iteration=3)
    print(decision.model_dump())