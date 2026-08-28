# The Cost of Decomposition: Trade-Offs in Long-Context Aspect-Based Scientific Summarization

**Pierre Achkar, Elena Senger, Tim Gollub, Martin Potthast, and Yuri Campbell**  
**EMNLP 2026 Industry Track**

This repository contains the research code for the paper. It evaluates aspect-based scientific
summarisation with several generation pipelines, including 2A2S, extraction-
to-abstraction, retrieval-augmented generation, chain-of-density, zero-shot
generation, and self-refinement.

## Repository layout

- `src/agentic/`: agentic 2A2S pipeline
- `src/e2a/`: extraction-to-abstraction pipeline
- `src/rag/`: retrieval-augmented generation pipeline
- `src/cod/`: chain-of-density baseline
- `src/zero_shot_llm/`: direct generation baseline
- `src/self_refine/`: self-refinement pipeline
- `src/data_prep/`: dataset preparation utilities
- `src/eval/`: automatic evaluation and result aggregation
- `src/language_engine.py`: shared vLLM generation wrapper

## Requirements

Use a Linux environment with a CUDA-capable GPU for generation experiments.
Install the dependencies from `requirements.txt`:

```bash
python3 -m pip install -r requirements.txt
```

PyTorch and vLLM must be installed in versions compatible with the CUDA version
on the target machine. For a GPU environment, consult the vLLM and PyTorch
installation instructions if the generic pip install selects an unsuitable
build.


## Data sources and preparation

The upstream datasets are:

- [PMC / PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/): compile the local
  PubMed dataset through
  `src/pubmed_ds`.
- [ACLSum](https://github.com/sobamchan/aclsum)
- [FacetSum](https://huggingface.co/datasets/memray/FacetSum)

Use the scripts in `src/data_prep/` to convert downloaded or compiled source
data into the JSONL format expected by the pipelines. For example, ACLSum raw
JSONL splits are converted with:

```bash
python src/data_prep/preprocess_aclsum.py \
  --input-dir /path/to/aclsum/raw_split \
  --output-dir data/aclsum \
  --splits train val test
```

FacetSum is loaded from Hugging Face by `src/data_prep/preprocess_facetsum.py`.
Run the PubMed preparation workflow under `src/pubmed_ds` before running the
summarisation pipelines.

The default data root used by `src/data_utils.py` is the repository-relative
`data/` directory. Set `ABSS_DATA_ROOT` to use another local or mounted data
directory. 

Supported dataset names are `facetsum`, `aclsum`, and `pmc`. Input records are
JSONL and must contain the fields expected by the selected pipeline, including
`source_text`, `summary`, and usually `unique_id`.

All pipeline ingestion goes through the shared `src/data_utils.py` module.
Its loaders and formatting functions normalize source text, aspect summaries,
source types, and dataset-specific fields into the structures consumed by the
generation runners. Keep the output paths produced by `data_prep` aligned with
the paths configured in `data_utils.py`.

## Model access and runtime

The default model is `Qwen/Qwen3.5-9B`; model and vLLM settings are
defined in the runner or engine configuration.

For the vLLM runners, set:

```bash
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

## Running experiments

Run commands from the repository root. `--max-samples` is useful for a smoke
test before launching a full experiment.

```bash
python src/agentic/batch_2a2s.py --dataset facetsum --split test \
  --context-size long --max-samples 10

python src/e2a/batch_e2a.py --dataset pmc --context-size long --max-samples 10

python src/rag/batch_rag.py --dataset pmc --context-size long

python src/cod/batch_cod.py --dataset aclsum --split test \
  --context-size short --max-samples 10

python src/zero_shot_llm/zero_shot_vllm.py --dataset aclsum --split test \
  --context-size short --max-samples 10

python src/self_refine/pipeline.py --dataset aclsum --split test \
  --max-samples 10 --output-dir results
```

Before a paper-scale run, verify the selected data split, model revision,
random seed, output directory, and GPU allocation. Preserve the generated
configuration and logs with the result files so runs remain reproducible.

## Evaluation

The evaluation scripts cover standard summarisation metrics, aspect-wise
aggregation, BERTScore/ROUGE analysis, G-Eval, fact scoring, token usage, and
significance testing. Typical commands are:

```bash
python src/eval/run_standard_metrics_and_aggregate_aspect_wise.py \
  --input results.jsonl --output metrics.jsonl \
  --aggregate-output metrics_aggregate.json

python src/eval/run_geval_on_rebuttal_subsample.py \
  --input results.jsonl --dataset facetsum \
  --output geval.jsonl --aggregate-output geval_aggregate.json
```

G-Eval requires OpenAI settings in the environment.

## Reproducibility notes

This is research code and several scripts retain experiment-specific defaults,
storage paths, and model choices. Treat those values as part of the experiment
configuration, review them before each run, and do not commit credentials or
large generated outputs. The LaTeX source and compiled paper are maintained
in the separate paper project.