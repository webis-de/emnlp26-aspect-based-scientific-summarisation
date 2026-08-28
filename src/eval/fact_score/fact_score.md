# Claim-Based Fact Score Metrics

This file explains the fact-score metrics implemented in `src/eval/fact_score/fact_score.py`: claim recall, claim precision, and claim F1. The implementation evaluates factual overlap between a generated aspect summary and its gold reference by decomposing both summaries into atomic claims and checking entailment in both directions.

## Overview

For each evaluated example, the input row is expected to contain:

- `gold_aspect_summary`: the reference aspect summary,
- `generated_aspect_summary`: the system output,
- optionally `unique_id`, `aspect_name`, `dataset`, and other metadata.

The metric pipeline has two model-based components:

1. Claim decomposition: an instruction-tuned LLM decomposes each summary into atomic factual claims.
2. Entailment scoring: the TRUE NLI model checks whether one summary entails each claim from the other summary.

The final per-example scores are written back into the JSONL rows as:

- `fact_claim_recall`,
- `fact_claim_precision`,
- `fact_claim_f1`.

## Claim Decomposition

Both gold and generated summaries are decomposed into atomic claims using `ClaimDecomposer`. The decomposer calls the local vLLM `LanguageEngine` with the schema in `io_agents.py`:

```json
{
  "claims": ["claim 1", "claim 2"]
}
```

The decomposition prompt asks the model to extract only explicit factual claims, keep them atomic and non-overlapping, avoid inference, remove duplicates, and return an empty list if no factual claims exist.

After decomposition, claims are normalized by:

- collapsing whitespace,
- dropping empty claims,
- deduplicating claims case-insensitively while preserving order.

Gold claim decompositions are cached by `unique_id` because the same reference summary may be reused across evaluations. Generated summaries are decomposed for the current input file.

## Entailment Model

Entailment is computed with `google/t5_xxl_true_nli_mixture` through `TrueEntailmentScorer`. Each pair is formatted as:

```text
premise: <premise text> hypothesis: <claim text>
```

The generated TRUE output is converted to a binary label:

- `1` if the model output indicates entailment,
- `0` otherwise.

The implementation deduplicates premise-claim pairs globally before running TRUE, so repeated pairs are scored once and reused.

## Claim Recall

Claim recall measures how many reference claims are supported by the generated summary. Let:

- `G = {g_1, ..., g_m}` be the decomposed gold claims,
- `P` be the generated summary text,
- `ENTAILS(P, g_i)` return 1 if the generated summary entails gold claim `g_i`.

The implemented recall is:

$$
\mathrm{ClaimRecall} = \frac{1}{|G|} \sum_{g_i \in G} \mathrm{ENTAILS}(P, g_i).
$$

In code, this is stored as `fact_claim_recall`.

Interpretation:

- High recall means the generated summary covers many of the factual claims present in the gold summary.
- Low recall means the generated summary omits gold-reference information.

Implementation details:

- If either the gold summary or generated summary is empty or equal to `Unknown`, recall is set to `0.0`.
- If the gold summary is non-empty but decomposes to zero claims, recall is also set to `0.0` because the denominator would otherwise be empty.

## Claim Precision

Claim precision measures how many generated claims are supported by the gold reference summary. Let:

- `H = {h_1, ..., h_n}` be the decomposed generated claims,
- `R` be the gold summary text,
- `ENTAILS(R, h_j)` return 1 if the gold summary entails generated claim `h_j`.

The implemented precision is:

$$
\mathrm{ClaimPrecision} = \frac{1}{|H|} \sum_{h_j \in H} \mathrm{ENTAILS}(R, h_j).
$$

In code, this is stored as `fact_claim_precision`.

Interpretation:

- High precision means most generated factual claims are supported by the reference.
- Low precision means the generated summary contains claims not entailed by the gold summary, which may indicate hallucination, over-specificity, or unsupported content relative to the reference.

Implementation details:

- If either the gold summary or generated summary is empty or equal to `Unknown`, precision is set to `0.0`.
- If the generated summary is non-empty but decomposes to zero claims, precision is also set to `0.0`.
- The implementation does not treat an empty generated claim list as perfectly precise.

## Claim F1

Claim F1 combines claim recall and claim precision with the harmonic mean:

$$
\mathrm{ClaimF1} =
\begin{cases}
0, & \text{if } \mathrm{ClaimRecall} + \mathrm{ClaimPrecision} = 0 \\
\frac{2 \cdot \mathrm{ClaimRecall} \cdot \mathrm{ClaimPrecision}}
{\mathrm{ClaimRecall} + \mathrm{ClaimPrecision}}, & \text{otherwise.}
\end{cases}
$$

In code, this is stored as `fact_claim_f1`.

Interpretation:

- High F1 requires both coverage of gold claims and support for generated claims.
- A system can have high recall but low precision if it covers gold information while adding unsupported claims.
- A system can have high precision but low recall if it is conservative and only states supported information while omitting many gold claims.

## Per-Example Output Fields

For each row, the evaluator adds:

- `fact_claim_recall`: claim recall score,
- `fact_claim_precision`: claim precision score,
- `fact_claim_f1`: harmonic mean of precision and recall,
- `fact_ref_claim_count`: number of decomposed gold claims,
- `fact_pred_claim_count`: number of decomposed generated claims,
- `fact_ref_claims_entailed`: number of gold claims entailed by the generated summary,
- `fact_pred_claims_entailed`: number of generated claims entailed by the gold summary,
- `fact_gold_claims`: normalized decomposed gold claims,
- `fact_generated_claims`: normalized decomposed generated claims.

If trace writing is enabled, the evaluator can also save per-claim entailment flags for debugging.

## Aggregation

After scoring all examples, the evaluator writes an aggregate JSON file containing:

- `overall`: mean `fact_claim_recall`, `fact_claim_precision`, and `fact_claim_f1` over all processed rows,
- `by_aspect`: the same means grouped by `aspect_name`.

Aggregation is a simple arithmetic mean over per-example metric values. Rows with empty or unknown summaries remain in the output and contribute zeros.

## Practical Reading of the Metrics

These metrics compare generated summaries to gold summaries through decomposed factual claims, not directly to the full source document. Therefore:

- claim recall estimates how much reference information the generated summary preserves,
- claim precision estimates how much generated information is supported by the reference,
- claim F1 balances these two directions.

Because entailment is checked against the opposite summary rather than the full source text, a generated claim that is source-faithful but absent from the gold summary may still lower precision. Similarly, a gold claim absent from the generated summary lowers recall even if the generated summary is otherwise faithful.