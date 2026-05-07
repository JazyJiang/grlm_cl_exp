#!/bin/bash
# Run CL chain with sliding-window routing + aux loss.
# Usage: bash run_books_cl_routed.sh <model_size> <cap> <gpu_ids> <sliding_window> <aux_weight>
#   model_size: 06b, 17b, or 4b
#   cap: h2, h5, h10, h20, h30, h40, or full
#   gpu_ids: e.g., 1 or 4,5
#   sliding_window: token count (default 512 ≈ 24 items)
#   aux_weight: aux loss coefficient (default 0.1)

MODEL_SIZE=$1
CAP=$2
GPU_IDS=$3
SW=${4:-512}
AUX_W=${5:-0.1}

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR=${WORK_DIR}/LlamaFactory
EVAL_SCRIPT=${WORK_DIR}/eval/s5_books_cl_eval_seq.py
TID2ITEM=${WORK_DIR}/data/books_tid2item_id.json
ID2META=${WORK_DIR}/data/books_id2meta.json
EVAL_DIR=${WORK_DIR}/data/cl_sft
RESULT_DIR=${WORK_DIR}/results/cl_results_seq/${MODEL_SIZE}_${CAP}_sw${SW}
MODEL_DIR=${WORK_DIR}/models

mkdir -p $RESULT_DIR

# Model configs
if [ "$MODEL_SIZE" == "06b" ]; then
    MODEL_PATH=${MODEL_DIR}/Qwen3-0.6B
    LR_INIT=7e-5
    LR_FT=3e-5
    NUM_GPUS=1
    BS=4
    GA=16
elif [ "$MODEL_SIZE" == "17b" ]; then
    MODEL_PATH=${MODEL_DIR}/Qwen3-1.7B
    LR_INIT=5e-5
    LR_FT=2e-5
    NUM_GPUS=1
    BS=4
    GA=16
elif [ "$MODEL_SIZE" == "4b" ]; then
    MODEL_PATH=${MODEL_DIR}/Qwen3-4B
    LR_INIT=1e-4
    LR_FT=5e-5
    NUM_GPUS=2
    BS=4
    GA=8
else
    echo "Unknown model size: $MODEL_SIZE (use 06b, 17b, or 4b)"
    exit 1
fi

# Dataset name based on cap
if [ "$CAP" == "full" ]; then
    DATASET_SUFFIX=""
else
    DATASET_SUFFIX="_${CAP}"
fi

PREV_CKPT=""

for PERIOD in 0 1 2 3; do
    DATASET_NAME="grlm_indomain_books_cl_D${PERIOD}${DATASET_SUFFIX}"
    OUTPUT_DIR=${WORK_DIR}/checkpoints/grlm_books_cl_${MODEL_SIZE}_${CAP}_sw${SW}_D${PERIOD}

    if [ "$PERIOD" -eq 0 ]; then
        INIT_MODEL=$MODEL_PATH
        LR=$LR_INIT
        EPOCHS=5
    else
        INIT_MODEL=$PREV_CKPT
        LR=$LR_FT
        EPOCHS=3
    fi

    echo "[$(date)] === ${MODEL_SIZE} cap=${CAP} sw=${SW} aux=${AUX_W} D${PERIOD} === (from: $INIT_MODEL)"

    # Train with routing
    WANDB_DISABLED=true DISABLE_VERSION_CHECK=1 \
    PYTHONPATH=${WORK_DIR}:$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=$GPU_IDS python3 -m routing.train_with_routing \
        --model_name_or_path $INIT_MODEL \
        --do_train \
        --dataset $DATASET_NAME \
        --dataset_dir ${LLAMA_DIR}/data \
        --template qwen3_nothink \
        --finetuning_type full \
        --output_dir $OUTPUT_DIR \
        --overwrite_cache \
        --overwrite_output_dir \
        --save_strategy no \
        --per_device_train_batch_size $BS \
        --gradient_accumulation_steps $GA \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --learning_rate $LR \
        --num_train_epochs $EPOCHS \
        --plot_loss \
        --bf16 \
        --report_to none \
        --sliding_window $SW \
        --aux_loss_weight $AUX_W

    echo "[$(date)] Training D${PERIOD} DONE"

    # Eval (same as vanilla — routing is baked into the saved config)
    EVAL_FILE=${EVAL_DIR}/amazon_books_cl_D${PERIOD}_eval.json

    if [ "$CAP" == "full" ]; then
        CAP_NUM=9999
    else
        CAP_NUM=${CAP#h}
    fi

    if [ "$CAP_NUM" -le 10 ]; then
        if [ "$MODEL_SIZE" == "06b" ]; then EVAL_BS=32
        elif [ "$MODEL_SIZE" == "17b" ]; then EVAL_BS=16
        else EVAL_BS=8; fi
    elif [ "$CAP_NUM" -le 40 ]; then
        if [ "$MODEL_SIZE" == "06b" ]; then EVAL_BS=20
        elif [ "$MODEL_SIZE" == "17b" ]; then EVAL_BS=10
        else EVAL_BS=6; fi
    else
        if [ "$MODEL_SIZE" == "06b" ]; then EVAL_BS=16
        elif [ "$MODEL_SIZE" == "17b" ]; then EVAL_BS=8
        else EVAL_BS=4; fi
    fi

    echo "[$(date)] Sequential eval D${PERIOD}->D$((PERIOD+1)) (batch_size=$EVAL_BS)..."
    if [ "$CAP" == "full" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_IDS python3 $EVAL_SCRIPT \
            --model $OUTPUT_DIR --eval_file $EVAL_FILE --tid2item_id $TID2ITEM \
            --id2meta $ID2META \
            --num_gpus $NUM_GPUS --batch_size $EVAL_BS --max_users 5000 \
            --output_dir $RESULT_DIR > $RESULT_DIR/eval_D${PERIOD}.log 2>&1
    else
        CUDA_VISIBLE_DEVICES=$GPU_IDS python3 $EVAL_SCRIPT \
            --model $OUTPUT_DIR --eval_file $EVAL_FILE --tid2item_id $TID2ITEM \
            --id2meta $ID2META \
            --max_hist $CAP_NUM --num_gpus $NUM_GPUS --batch_size $EVAL_BS --max_users 5000 \
            --output_dir $RESULT_DIR > $RESULT_DIR/eval_D${PERIOD}.log 2>&1
    fi
    echo "[$(date)] Eval DONE"

    # Delete previous checkpoint
    if [ -n "$PREV_CKPT" ] && [ -d "$PREV_CKPT" ]; then
        echo "[$(date)] Deleting previous checkpoint: $PREV_CKPT"
        rm -rf $PREV_CKPT
    fi

    PREV_CKPT=$OUTPUT_DIR
    echo "[$(date)] === D${PERIOD} COMPLETE ==="
    echo ""
done

echo "[$(date)] Keeping final checkpoint: $PREV_CKPT"
echo "[$(date)] ===== ${MODEL_SIZE} cap=${CAP} sw=${SW} ALL PERIODS COMPLETE ====="
echo "Results in: $RESULT_DIR"
