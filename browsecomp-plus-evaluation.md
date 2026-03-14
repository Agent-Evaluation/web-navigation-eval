# BrowseComp-Plus Evaluation Process

Reference: Kim et al. (2025), *Towards a Science of Scaling Agent Systems*  
Dataset: Chen et al. (2025), *BrowseComp-Plus* — [github.com/texttron/BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus)

---

## Overview

BrowseComp-Plus evaluates agents on **100 web browsing tasks** requiring multi-website information synthesis. Tasks include comparative analysis, fact verification, and comprehensive research across multiple web sources. Evaluation uses **LLM-as-judge** (Qwen3-32B) comparing agent responses against ground truth with confidence scoring.

---

## Step 1: Setup Environment

```bash
# Clone the repo
git clone https://github.com/texttron/BrowseComp-Plus.git
cd BrowseComp-Plus

# Install uv (Python 3.10 environment manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
uv pip install --no-build-isolation flash-attn

# Java 21 (required for indexing)
conda install -c conda-forge openjdk=21
```

---

## Step 2: Download the Dataset

```bash
# Login to HuggingFace (required for obfuscated queries/answers)
huggingface-cli login

# Decrypt and download queries + ground truth answers + relevance judgements
pip install datasets
python scripts_build_index/decrypt_dataset.py \
  --output data/browsecomp_plus_decrypted.jsonl \
  --generate-tsv topics-qrels/queries.tsv
```

The corpus (non-obfuscated) can be loaded directly:

```python
from datasets import load_dataset
ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
```

---

## Step 3: Download Pre-Built Retrieval Indexes

```bash
# Downloads BM25 and Qwen3-Embedding indexes to ./indexes
bash scripts_build_index/download_indexes.sh
```

The paper uses a **local retriever tool** via function calling that returns the top 5 relevant documents with a maximum context length of 512 tokens per retrieval call, ensuring fair comparison across architectures.

---

## Step 4: Run Agent Architectures

The paper evaluates **5 canonical architectures** across **3 LLM families** (OpenAI, Google, Anthropic):

| Architecture | Description |
|---|---|
| Single-Agent (SAS) | One agent, up to 10 iterations |
| Independent MAS | 3 agents, no coordination, synthesis only |
| Centralized MAS | 1 orchestrator + 3 sub-agents, 5 orchestration rounds |
| Decentralized MAS | 3 agents, 3 debate rounds |
| Hybrid MAS | Centralized orchestration + peer communication phases |

Each agent has access to the **local retriever tool** returning top-5 documents per query.

### Output format per task

For each of the 100 queries, save a JSON file to `runs/<your_run_name>/`:

```json
{
  "query_id": "string",
  "tool_call_counts": {"retriever": 12, "...": 0},
  "status": "completed",
  "retrieved_docids": ["doc_id_1", "doc_id_2", "..."],
  "result": [
    {
      "type": "output_text",
      "output": "Final answer text from the agent"
    }
  ]
}
```

> `status` must be `"completed"` for a successful run; anything else is treated as failure.

---

## Step 5: Evaluate

```bash
python scripts_evaluation/evaluate_run.py \
  --input_dir runs/<your_run_name> \
  --tensor_parallel_size <num_gpus>
```

The judge model is **Qwen3-32B** (requires local GPU). It compares each agent output against the ground truth answer with confidence scoring and produces a binary success/failure per task.

> See [`docs/llm_as_judge.md`](https://github.com/texttron/BrowseComp-Plus/blob/main/docs/llm_as_judge.md) for full judge details.

---

## Step 6: Record Results

Collect the **% accuracy** (correct / 100 tasks) per architecture per model. This is the metric used in the paper (Table 2 and Figure 1).

Expected result patterns from the paper:
- **Decentralized coordination** excels on web navigation (+9.2% vs. +0.2% for centralized)
- BrowseComp-Plus has the highest domain complexity score (**D = 0.839**) of the four benchmarks, meaning multi-agent overhead tends to hurt performance above this threshold
- Independent agents amplify errors **17.2×**; centralized coordination contains this to **4.4×**

---

## API-Specific Guides

The BrowseComp-Plus repo provides model-specific run guides:

- [OpenAI API](https://github.com/texttron/BrowseComp-Plus/blob/main/docs/openai.md)
- [Gemini API](https://github.com/texttron/BrowseComp-Plus/blob/main/docs/gemini.md)
- [Anthropic API](https://github.com/texttron/BrowseComp-Plus/blob/main/docs/anthropic.md)
- [Custom Retriever](https://github.com/texttron/BrowseComp-Plus/blob/main/docs/custom_retriever.md)

---

## Cost Note

Evaluating all 100 queries with a frontier model (e.g., o3) can cost ~**$100–$1000 USD** depending on the model and number of architectures. Pre-computed trajectory data for expensive baselines is available:

```bash
bash scripts_build_index/download_run_files.sh
# Output: data/decrypted_run_files/
```

---

## Connection to the Paper's Framework

```
Dataset (100 tasks)
      ↓
Run 5 architectures × 3 LLM families = 15 configurations
      ↓
Save JSON outputs per task
      ↓
LLM judge (Qwen3-32B) scores each output vs. ground truth
      ↓
% accuracy per configuration
      ↓
Feed into scaling law regression (coordination metrics → performance prediction)
```

BrowseComp-Plus contributes to the paper's finding that **decentralized coordination** is optimal for dynamic web navigation, and that high-complexity tasks (D > 0.40) degrade under most multi-agent overhead.
