# GRLM Continual Learning Experiments (Books)

Sequential CL training + Tiger-style sequential evaluation for GRLM (Qwen3) on Amazon Books.

## Requirements

- NVIDIA GPU with 80GB VRAM (H100 recommended)
- 7+ GPUs to run all chains in parallel (or fewer GPUs with longer wall-clock time)
- Python 3.10+, PyTorch 2.x, CUDA 12+
- `huggingface-cli` installed (`pip install huggingface_hub`)
- ~50GB disk for models, ~10GB for data, ~100GB for checkpoints during training

## Quick Start

```bash
# 1. Clone repo
git clone git@github.com:JazyJiang/grlm_cl_exp.git
cd grlm_cl_exp

# 2. Run setup (downloads models + data + installs LlamaFactory)
#    Takes ~30min depending on network (downloads ~20GB of models + 1.1GB data)
bash setup.sh

# 3. (Optional) Verify data is correct
ls data/books_id2meta.json data/books_tid2item_id.json
ls data/cl_sft/amazon_books_cl_D0_train.json  # should exist

# 4. Run experiments (see dispatch options below)
bash dispatch_all.sh
```

## What `setup.sh` Does

1. Clones [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and installs it
2. Downloads Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B from HuggingFace → `models/`
3. Downloads experiment data from `JazySong/grlm-books-cl-data` → `data/`
4. Extracts `cl_sft.tar.gz` into `data/cl_sft/` (train + eval JSONs for all periods)
5. Creates D0 symlinks (D0 train is identical across all history caps)
6. Generates `dataset_info.json` for LlamaFactory

If any step was already completed (directory exists), it skips automatically. Safe to re-run.

## Experiment Design

**CL Protocol:** D0→D1→D2→D3 sequential fine-tuning on Amazon Books (300K users, 142K items, 5 time periods).

**Models:** Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B

**History caps:** h=2, 5, 10, 20, 30, 40, full (sliding window size for input history)

**Evaluation:** Tiger-style sequential per-target prediction. For each user's D_{t+1} items (chronologically ordered), predict one at a time with a sliding history window. Each (user, target) pair is independently evaluated. Recall@K = fraction of pairs where target appears in top-K beam search candidates.

**Grouping:** Users grouped into 5 quintiles by accumulated history length (Group 1 = longest history, most susceptible to history noise).

## Running Experiments

### Option A: Run all chains in parallel (recommended if you have 7 GPUs)

```bash
bash dispatch_all.sh
```

This launches all 7 caps for 0.6B simultaneously on GPUs 1-7. After those finish (~4-6h), the script prints commands for the 1.7B and 4B batches — copy-paste them.

**Full timeline on 7× H100:**
- 0.6B (7 caps): ~4-6h (all parallel)
- 1.7B (7 caps): ~6-10h (all parallel, after 0.6B)
- 4B (7 caps): ~10-16h (3-4 at a time, 2 GPUs each)
- **Total: ~24-32h**

### Option B: Sequential per GPU (simpler, fewer GPUs OK)

```bash
bash dispatch_sequential.sh
```

Each GPU runs 0.6B then 1.7B for one cap value. 4B chains run afterward.

### Option C: Run a single chain manually

```bash
bash run_books_cl_v2.sh <model_size> <cap> <gpu_ids>
```

Examples:
```bash
bash run_books_cl_v2.sh 06b h10 1       # 0.6B, cap=10, GPU 1
bash run_books_cl_v2.sh 17b full 3      # 1.7B, full history, GPU 3
bash run_books_cl_v2.sh 4b h20 4,5      # 4B, cap=20, GPUs 4+5
```

Parameters:
- `model_size`: `06b`, `17b`, or `4b`
- `cap`: `h2`, `h5`, `h10`, `h20`, `h30`, `h40`, or `full`
- `gpu_ids`: e.g., `1` (single GPU) or `4,5` (multi-GPU, required for 4B)

Each chain trains D0→D1→D2→D3 sequentially, with eval after each period. Previous checkpoints are auto-deleted to save disk.

### Monitoring Progress

```bash
# Watch a specific chain's log
tail -f logs/06b_h10.log

# Check which chains are done (results appear when eval finishes)
ls results/cl_results_seq/*/seq_recall_*.json

# GPU utilization
nvidia-smi
```

## Hyperparameters

| Model | D0 lr | D1+ lr | D0 epochs | D1+ epochs | GPUs | Batch (BS×GA) |
|-------|-------|--------|-----------|------------|------|---------------|
| 0.6B  | 7e-5  | 3e-5   | 5         | 3          | 1    | 4×16 = 64     |
| 1.7B  | 5e-5  | 2e-5   | 5         | 3          | 1    | 4×16 = 64     |
| 4B    | 1e-4  | 5e-5   | 5         | 3          | 2    | 4×8 = 32      |

D0 trains from the pretrained model (more epochs + higher lr). D1+ fine-tunes from the previous period's checkpoint (lower lr to reduce forgetting).

## Directory Structure

```
grlm_cl_exp/
├── setup.sh                    # One-time setup (downloads everything)
├── run_books_cl_v2.sh          # Single chain: train 4 periods + eval
├── dispatch_all.sh             # Launch all 21 chains in parallel
├── dispatch_sequential.sh      # Alternative: sequential per GPU
├── scripts/
│   └── generate_dataset_info.py
├── eval/
│   ├── s5_books_cl_eval_seq.py # Sequential eval (Tiger-style)
│   └── recompute_cl_recall.py  # Re-compute recall from saved results
├── LlamaFactory/               # Cloned by setup.sh (not in git)
├── models/                     # Downloaded by setup.sh (not in git)
│   ├── Qwen3-0.6B/
│   ├── Qwen3-1.7B/
│   └── Qwen3-4B/
├── data/                       # Downloaded by setup.sh (not in git)
│   ├── books_id2meta.json      # Item metadata (142K items, keywords+title)
│   ├── books_tid2item_id.json  # TID → item_id mapping for eval
│   └── cl_sft/                 # Train + eval JSONs per period × cap
├── checkpoints/                # Training checkpoints (auto-cleaned)
├── logs/                       # Training + eval logs
└── results/
    └── cl_results_seq/         # Final results
        ├── 06b_h10/
        │   ├── seq_recall_h10_D0.json
        │   ├── seq_results_h10_D0.jsonl
        │   └── ...
        └── ...
```

## Output Format

Each eval produces (per period):
- `seq_recall_{tag}_D{t}.json`: Recall@1/5/10/20, overall + per-group breakdown
- `seq_results_{tag}_D{t}.jsonl`: Per-(user, target) hit details for further analysis

Example `seq_recall_h10_D0.json`:
```json
{
  "overall": {"recall@1": 0.032, "recall@5": 0.089, "recall@10": 0.134, "recall@20": 0.187},
  "group_1": {"recall@1": 0.021, ...},
  "group_2": {...},
  ...
}
```

## Troubleshooting

- **`huggingface-cli: command not found`**: Run `pip install huggingface_hub`
- **CUDA OOM during eval**: Reduce eval batch size by editing `run_books_cl_v2.sh` (EVAL_BS variables around line 138)
- **Disk full during training**: Each checkpoint is 2-7GB depending on model size. The script auto-deletes the previous period's checkpoint after eval completes. Ensure ~100GB free.
- **Want to re-run eval only (without retraining)**: Use `recompute_cl_recall.py` on existing `seq_results_*.jsonl` files
- **Network issues downloading from HuggingFace**: Set `HF_ENDPOINT` or use a proxy: `export https_proxy=http://your-proxy:port`
