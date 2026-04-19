#!/bin/bash

# Script to run DLinear on Exchange with L=336 and H in {96,192,336,720}
# Uses the same downstream finetune style and key parameters as PatchTST scripts.
set -o pipefail

DATASET="Exchange"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./outputs/logs/dlinear_${DATASET}_${TIMESTAMP}"
RESULTS_DIR="./outputs/test_results/dlinear_${DATASET}_${TIMESTAMP}"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

ROOT_PATH="./datasets/exchange/"
DATA_PATH="exchange.csv"
DATA_NAME="Exchange"
MODEL="DLinear"

INPUT_LENS=(336)
LABEL_LEN=48
PRED_LENS=(96 192 336 720)
ENC_IN=8
DEC_IN=8
C_OUT=8
E_LAYERS=2
N_HEADS=8
D_MODEL=64
D_FF=128
DROPOUT=0.2
FC_DROPOUT=0.0
HEAD_DROPOUT=0.0
PATCH_LEN=16
STRIDE=8
PADDING_PATCH="end"
REVIN=1
AFFINE=0
SUBTRACT_LAST=0
DECOMPOSITION=0
KERNEL_SIZE=25
INDIVIDUAL=1
EMBED_TYPE=0
TRAIN_EPOCHS=10
BATCH_SIZE=16
LEARNING_RATE=0.0001
GPU=0
ITR=1

SUMMARY_FILE="$RESULTS_DIR/comparison_summary.txt"

echo "DLinear Exchange downstream finetune" > "$SUMMARY_FILE"
echo "Timestamp: $TIMESTAMP" >> "$SUMMARY_FILE"
echo "Dataset: $DATASET" >> "$SUMMARY_FILE"
echo "Model: $MODEL" >> "$SUMMARY_FILE"
echo "Input lengths: ${INPUT_LENS[*]}, label length: $LABEL_LEN" >> "$SUMMARY_FILE"
echo "Prediction lengths: ${PRED_LENS[*]}" >> "$SUMMARY_FILE"
echo "Individual head: $INDIVIDUAL, e_layers: $E_LAYERS, d_model: $D_MODEL" >> "$SUMMARY_FILE"
echo "Train epochs: $TRAIN_EPOCHS, batch size: $BATCH_SIZE, lr: $LEARNING_RATE" >> "$SUMMARY_FILE"
echo "----------------------------------------" >> "$SUMMARY_FILE"

echo "Starting DLinear Exchange runs..." | tee -a "$SUMMARY_FILE"

for seq_len in "${INPUT_LENS[@]}"; do
    echo "Running input_len=$seq_len" | tee -a "$SUMMARY_FILE"
    for pred_len in "${PRED_LENS[@]}"; do
        model_id="${DATASET}_L${seq_len}_H${pred_len}"
        log_file="$LOG_DIR/dlinear_${DATASET}_L${seq_len}_pl${pred_len}.log"

        echo "Running DLinear $DATASET, seq_len=$seq_len, pred_len=$pred_len" | tee -a "$SUMMARY_FILE"
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
            --label_len $LABEL_LEN \
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
            echo "ERROR: DLinear failed for pred_len=$pred_len" | tee -a "$SUMMARY_FILE"
            continue
        fi

        echo "Completed pred_len=$pred_len" | tee -a "$SUMMARY_FILE"
        echo "Log file: $log_file" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
    done
    echo "Completed input_len=$seq_len" | tee -a "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
done

echo "DLinear Exchange script finished." | tee -a "$SUMMARY_FILE"
