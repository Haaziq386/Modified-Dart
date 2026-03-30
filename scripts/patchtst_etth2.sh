#!/bin/bash

# Script to run PatchTST on ETTh2 with parameters chosen to be comparable to other ETTh2 baselines
# Usage: bash scripts/patchtst_etth2.sh

set -o pipefail

DATASET="ETTh2"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./outputs/logs/patchtst_etth2_${TIMESTAMP}"
RESULTS_DIR="./outputs/test_results/patchtst_etth2_${TIMESTAMP}"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Dataset configuration
ROOT_PATH="./datasets/ETT-small/"
DATA_PATH="ETTh2.csv"
DATA_NAME="ETTh2"
MODEL="PatchTST"

# Model / training configuration

INPUT_LENS=(96 192 336)
ENC_IN=7
DEC_IN=7
C_OUT=7
E_LAYERS=3
N_HEADS=4
D_MODEL=16
D_FF=128
DROPOUT=0.3
FC_DROPOUT=0.3
HEAD_DROPOUT=0.0
PATCH_LEN=16
STRIDE=8
PADDING_PATCH="end"
REVIN=1
AFFINE=0
SUBTRACT_LAST=0
DECOMPOSITION=0
KERNEL_SIZE=25
INDIVIDUAL=0
EMBED_TYPE=0
TRAIN_EPOCHS=100
BATCH_SIZE=128
LEARNING_RATE=0.0001
GPU=0
ITR=1

# Prediction lengths to evaluate
PRED_LENS=(96 192 336 720)

# Logging
SUMMARY_FILE="$RESULTS_DIR/comparison_summary.txt"

echo "PatchTST ETTh2 run" > "$SUMMARY_FILE"
echo "Timestamp: $TIMESTAMP" >> "$SUMMARY_FILE"
echo "Dataset: $DATASET" >> "$SUMMARY_FILE"
echo "Model: $MODEL" >> "$SUMMARY_FILE"
echo "input_lens: ${INPUT_LENS[*]}, d_model: $D_MODEL, e_layers: $E_LAYERS, n_heads: $N_HEADS" >> "$SUMMARY_FILE"
echo "patch_len: $PATCH_LEN, stride: $STRIDE" >> "$SUMMARY_FILE"
echo "train_epochs: $TRAIN_EPOCHS, batch_size: $BATCH_SIZE" >> "$SUMMARY_FILE"
echo "----------------------------------------" >> "$SUMMARY_FILE"

echo "Input lengths: ${INPUT_LENS[*]}" >> "$SUMMARY_FILE"
echo "Prediction lengths: ${PRED_LENS[*]}" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for seq_len in "${INPUT_LENS[@]}"; do
    echo "Running input_len=$seq_len" | tee -a "$SUMMARY_FILE"
    for pred_len in "${PRED_LENS[@]}"; do
    model_id="${DATASET}_${seq_len}_${pred_len}"
    log_file="$LOG_DIR/patchtst_etth2_sl${seq_len}_pl${pred_len}.log"

    echo "Running PatchTST ETTh2, seq_len=$seq_len, pred_len=$pred_len"
    echo "  Log: $log_file"

    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --model_id "$model_id" \
        --model "$MODEL" \
        --data "$DATA_NAME" \
        --root_path "$ROOT_PATH" \
        --data_path "$DATA_PATH" \
        --features M \
        --seq_len $seq_len \
        --input_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --enc_in $ENC_IN \
        --dec_in $DEC_IN \
        --c_out $C_OUT \
        --e_layers $E_LAYERS \
        --n_heads $N_HEADS \
        --d_model $D_MODEL \
        --d_ff $D_FF \
        --dropout $DROPOUT \
        --fc_dropout $FC_DROPOUT \
        --head_dropout $HEAD_DROPOUT \
        --patch_len $PATCH_LEN \
        --stride $STRIDE \
        --padding_patch $PADDING_PATCH \
        --revin $REVIN \
        --affine $AFFINE \
        --subtract_last $SUBTRACT_LAST \
        --decomposition $DECOMPOSITION \
        --kernel_size $KERNEL_SIZE \
        --individual $INDIVIDUAL \
        --embed_type $EMBED_TYPE \
        --train_epochs $TRAIN_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --gpu $GPU \
        --itr $ITR \
        2>&1 | tee "$log_file"

    if [ $? -ne 0 ]; then
        echo "ERROR: PatchTST failed for pred_len=$pred_len" | tee -a "$SUMMARY_FILE"
        continue
    fi

    echo "PatchTST ETTh2 pred_len=$pred_len completed" | tee -a "$SUMMARY_FILE"
    echo "Log file: $log_file" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    done
    echo "Completed input_len=$seq_len" | tee -a "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
done

echo "PatchTST ETTh2 script finished. Summary: $SUMMARY_FILE"
