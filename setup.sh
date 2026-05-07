#!/bin/bash
# Setup script: download models, prepare data, install dependencies.
# Run this ONCE before starting experiments.
set -e

WORK_DIR=$(cd "$(dirname "$0")" && pwd)
MODEL_DIR=${WORK_DIR}/models
DATA_DIR=${WORK_DIR}/data
LLAMA_DIR=${WORK_DIR}/LlamaFactory

echo "=== GRLM CL Experiment Setup ==="
echo "Working directory: $WORK_DIR"

# ============================================================
# Step 1: Install LlamaFactory (if not already present)
# ============================================================
if [ ! -d "$LLAMA_DIR/src" ]; then
    echo "[Step 1] Cloning LlamaFactory..."
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git $LLAMA_DIR
    cd $LLAMA_DIR && pip install -e ".[torch,metrics]" && cd $WORK_DIR
else
    echo "[Step 1] LlamaFactory already exists, skipping."
fi

# ============================================================
# Step 2: Download Qwen3 models from HuggingFace
# ============================================================
echo "[Step 2] Downloading Qwen3 models..."
mkdir -p $MODEL_DIR

# Qwen3-0.6B
if [ ! -d "$MODEL_DIR/Qwen3-0.6B" ]; then
    echo "  Downloading Qwen3-0.6B..."
    huggingface-cli download Qwen/Qwen3-0.6B --local-dir $MODEL_DIR/Qwen3-0.6B
else
    echo "  Qwen3-0.6B already exists."
fi

# Qwen3-1.7B
if [ ! -d "$MODEL_DIR/Qwen3-1.7B" ]; then
    echo "  Downloading Qwen3-1.7B..."
    huggingface-cli download Qwen/Qwen3-1.7B --local-dir $MODEL_DIR/Qwen3-1.7B
else
    echo "  Qwen3-1.7B already exists."
fi

# Qwen3-4B (Instruct version)
if [ ! -d "$MODEL_DIR/Qwen3-4B" ]; then
    echo "  Downloading Qwen3-4B..."
    huggingface-cli download Qwen/Qwen3-4B --local-dir $MODEL_DIR/Qwen3-4B
else
    echo "  Qwen3-4B already exists."
fi

# ============================================================
# Step 3: Download/prepare data
# ============================================================
echo "[Step 3] Preparing data..."
mkdir -p $DATA_DIR/cl_sft

# Option A: Copy from shared storage (uncomment and set SRC if on same cluster)
# SRC=/workspace/jiangzhuosong/GRLM_0
# cp $SRC/in_domain/books/sum_data/books_id2meta.json $DATA_DIR/
# cp $SRC/in_domain/books/sum_data/item_id2tid/books_tid2item_id.json $DATA_DIR/
# cp $SRC/LlamaFactory/data/grlm_in_domain/amazon_books_cl_*.json $DATA_DIR/cl_sft/

# Option A: Download from Google Drive (link shared separately)
#   Download grlm_cl_data.tar.gz, books_id2meta.json, books_tid2item_id.json
#   and place them in $DATA_DIR/, then run this script.

# Auto-extract if tar.gz exists
if [ -f "$DATA_DIR/grlm_cl_data.tar.gz" ]; then
    echo "  Extracting grlm_cl_data.tar.gz..."
    tar xzf $DATA_DIR/grlm_cl_data.tar.gz -C $DATA_DIR/cl_sft
    rm -f $DATA_DIR/grlm_cl_data.tar.gz
fi

# D0 train is same for all caps (no prior history), create symlinks
if [ -f "$DATA_DIR/cl_sft/amazon_books_cl_D0_train.json" ]; then
    for cap in h2 h5 h10 h20 h30 h40; do
        ln -sf amazon_books_cl_D0_train.json $DATA_DIR/cl_sft/amazon_books_cl_D0_train_${cap}.json
    done
fi

# Check if data exists
if [ ! -f "$DATA_DIR/books_id2meta.json" ]; then
    echo ""
    echo "ERROR: Data not found at $DATA_DIR/"
    echo "Please download data from the shared Google Drive link and place:"
    echo "  $DATA_DIR/books_id2meta.json"
    echo "  $DATA_DIR/books_tid2item_id.json"
    echo "  $DATA_DIR/grlm_cl_data.tar.gz  (will be auto-extracted to cl_sft/)"
    echo ""
    exit 1
fi

# ============================================================
# Step 4: Setup dataset_info.json for LlamaFactory
# ============================================================
echo "[Step 4] Generating dataset_info.json..."
python3 ${WORK_DIR}/scripts/generate_dataset_info.py \
    --data_dir $DATA_DIR/cl_sft \
    --output $LLAMA_DIR/data/dataset_info.json

# ============================================================
# Step 5: Create symlinks and copy scripts
# ============================================================
echo "[Step 5] Setting up LlamaFactory integration..."
mkdir -p ${WORK_DIR}/results/cl_results_seq
mkdir -p ${WORK_DIR}/checkpoints

# Link data into LlamaFactory expected locations
ln -sfn $DATA_DIR/cl_sft $LLAMA_DIR/data/grlm_in_domain 2>/dev/null || true

echo ""
echo "=== Setup Complete ==="
echo "To run a single chain:  bash run_books_cl_v2.sh 06b h10 1"
echo "To run all chains:      bash dispatch_all.sh"
