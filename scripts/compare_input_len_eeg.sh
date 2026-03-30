#!/bin/bash

# Compare HtulTS vs TimeDART across input_len x d_model on EEG (classification)
# EEG: 3000 native length, 2 channels, 8 classes
# Usage: bash scripts/compare_input_len_eeg.sh

set -o pipefail

DATASET="EEG"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./outputs/logs/input_len_eeg_${TIMESTAMP}"
RESULTS_DIR="./outputs/test_results/input_len_eeg_${TIMESTAMP}"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Grid axes
INPUT_LENS=(750 1500 3000)   # 3000 = native full length
D_MODELS=(128)

# patch_len=73 stride=73: 73 doesn't divide 750, avoids (il-p)%s==0 for all input_lens (750,1500,3000)
PATCH_LEN=73
STRIDE=73
GPU=2

SUMMARY_FILE="$RESULTS_DIR/comparison_summary.txt"
echo "Input Length Comparison: HtulTS vs TimeDART on EEG" > "$SUMMARY_FILE"
echo "Timestamp: $TIMESTAMP" >> "$SUMMARY_FILE"
echo "Task: Classification (8 classes, 2 channels)" >> "$SUMMARY_FILE"
echo "patch_len=$PATCH_LEN  stride=$STRIDE" >> "$SUMMARY_FILE"
echo "============================================================" >> "$SUMMARY_FILE"

echo "=========================================="
echo "Input Length Comparison: HtulTS vs TimeDART"
echo "Dataset: $DATASET (3000 native, 2 channels, 8 classes)"
echo "Input lengths:  ${INPUT_LENS[*]}"
echo "d_model values: ${D_MODELS[*]}"
echo "patch_len=$PATCH_LEN  stride=$STRIDE  GPU=$GPU"
echo "=========================================="

# ============== HtulTS ==============

run_htults_pretrain() {
    local input_len=$1
    local d_model=$2
    local ckpt_dir="./outputs/pretrain_checkpoints/htults_eeg_il${input_len}_dm${d_model}"
    local log_file="$LOG_DIR/htults_pretrain_il${input_len}_dm${d_model}.log"

    echo ">>> HtulTS PRETRAIN  input_len=$input_len  d_model=$d_model"
    python -u run.py \
        --task_name pretrain \
        --downstream_task classification \
        --root_path datasets/eeg_no_big/ \
        --model_id EEG_il${input_len}_dm${d_model} \
        --model HtulTS \
        --data EEG \
        --input_len $input_len \
        --enc_in 2 --dec_in 2 --c_out 2 \
        --num_classes 8 \
        --e_layers 2 --d_layers 1 \
        --n_heads 8 --d_model $d_model --d_ff 256 \
        --patch_len $PATCH_LEN --stride $STRIDE \
        --head_dropout 0.2 --dropout 0.2 \
        --time_steps 1000 --scheduler cosine \
        --batch_size 128 --gpu $GPU \
        --learning_rate 0.001 --train_epochs 50 \
        --lr_decay 0.99 \
        --use_noise 0 --use_norm 0 --use_forgetting 0 \
        --tfc_weight 0.05 \
        --pretrain_checkpoints "$ckpt_dir" \
        2>&1 | tee "$log_file"
    echo "    ✓ HtulTS pretrain done  input_len=$input_len  d_model=$d_model"
}

run_htults_finetune() {
    local input_len=$1
    local d_model=$2
    local ckpt_dir="./outputs/pretrain_checkpoints/htults_eeg_il${input_len}_dm${d_model}"
    local log_file="$LOG_DIR/htults_finetune_il${input_len}_dm${d_model}.log"

    echo ">>> HtulTS FINETUNE  input_len=$input_len  d_model=$d_model"
    python -u run.py \
        --task_name finetune \
        --downstream_task classification \
        --root_path datasets/eeg_no_big/ \
        --model_id EEG_il${input_len}_dm${d_model} \
        --model HtulTS \
        --data EEG \
        --input_len $input_len \
        --enc_in 2 --dec_in 2 --c_out 2 \
        --num_classes 8 \
        --e_layers 2 --d_layers 1 \
        --n_heads 8 --d_model $d_model --d_ff 256 \
        --patch_len $PATCH_LEN --stride $STRIDE \
        --head_dropout 0.2 --dropout 0.2 \
        --time_steps 1000 --scheduler cosine \
        --batch_size 128 --gpu $GPU \
        --lr_decay 0.99 --lradj decay \
        --patience 100 --learning_rate 0.001 \
        --pct_start 0.3 --train_epochs 100 \
        --use_norm 0 --use_forgetting 0 \
        --pretrain_checkpoints "$ckpt_dir" \
        2>&1 | tee "$log_file"
    echo "    ✓ HtulTS finetune done  input_len=$input_len  d_model=$d_model"
}

# ============== TimeDART ==============

run_timedart_pretrain() {
    local input_len=$1
    local d_model=$2
    local ckpt_dir="./outputs/pretrain_checkpoints/timedart_eeg_il${input_len}_dm${d_model}"
    local log_file="$LOG_DIR/timedart_pretrain_il${input_len}_dm${d_model}.log"

    echo ">>> TimeDART PRETRAIN  input_len=$input_len  d_model=$d_model"
    python -u run.py \
        --task_name pretrain \
        --downstream_task classification \
        --root_path datasets/eeg_no_big/ \
        --model_id EEG_il${input_len}_dm${d_model} \
        --model TimeDART \
        --data EEG \
        --input_len $input_len \
        --enc_in 2 --dec_in 2 --c_out 2 \
        --num_classes 8 \
        --e_layers 2 --d_layers 1 \
        --n_heads 8 --d_model $d_model --d_ff 256 \
        --patch_len $PATCH_LEN --stride $STRIDE \
        --head_dropout 0.2 --dropout 0.2 \
        --time_steps 1000 --scheduler cosine \
        --batch_size 128 --gpu $GPU \
        --learning_rate 0.001 --train_epochs 50 \
        --lr_decay 0.99 \
        --use_norm 0 --use_forgetting 0 \
        --pretrain_checkpoints "$ckpt_dir" \
        2>&1 | tee "$log_file"
    echo "    ✓ TimeDART pretrain done  input_len=$input_len  d_model=$d_model"
}

run_timedart_finetune() {
    local input_len=$1
    local d_model=$2
    local ckpt_dir="./outputs/pretrain_checkpoints/timedart_eeg_il${input_len}_dm${d_model}"
    local log_file="$LOG_DIR/timedart_finetune_il${input_len}_dm${d_model}.log"

    echo ">>> TimeDART FINETUNE  input_len=$input_len  d_model=$d_model"
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --downstream_task classification \
        --root_path datasets/eeg_no_big/ \
        --model_id EEG_il${input_len}_dm${d_model} \
        --model TimeDART \
        --data EEG \
        --input_len $input_len \
        --enc_in 2 --dec_in 2 --c_out 2 \
        --num_classes 8 \
        --e_layers 2 --d_layers 1 \
        --n_heads 8 --d_model $d_model --d_ff 256 \
        --patch_len $PATCH_LEN --stride $STRIDE \
        --head_dropout 0.2 --dropout 0.2 \
        --time_steps 1000 --scheduler cosine \
        --batch_size 128 --gpu $GPU \
        --lr_decay 0.99 --lradj decay \
        --patience 100 --learning_rate 0.001 \
        --pct_start 0.3 --train_epochs 100 \
        --pretrain_checkpoints "$ckpt_dir" \
        2>&1 | tee "$log_file"
    echo "    ✓ TimeDART finetune done  input_len=$input_len  d_model=$d_model"
}

# ============== Metric Extraction ==============

extract_metrics() {
    local log_file=$1
    local model_name=$2
    local input_len=$3
    local d_model=$4

    if [ ! -f "$log_file" ]; then
        echo "$model_name | input_len=$input_len | d_model=$d_model | Acc: FILE_NOT_FOUND | F1: N/A" >> "$SUMMARY_FILE"
        return 0
    fi

    local accuracy f1
    accuracy=$(grep -E "Test Loss:.*Acc:.*F1:" "$log_file" 2>/dev/null | tail -1 | grep -oP "Acc:\s*\K[0-9.]+" 2>/dev/null || echo "N/A")
    f1=$(grep -E "Test Loss:.*Acc:.*F1:" "$log_file" 2>/dev/null | tail -1 | grep -oP "F1:\s*\K[0-9.]+" 2>/dev/null || echo "N/A")

    [ -z "$accuracy" ] && accuracy="N/A"
    [ -z "$f1" ] && f1="N/A"

    echo "$model_name | input_len=$input_len | d_model=$d_model | Acc: $accuracy | F1: $f1" >> "$SUMMARY_FILE"
    return 0
}

# ============== Main Loop ==============

for input_len in "${INPUT_LENS[@]}"; do
    for d_model in "${D_MODELS[@]}"; do
        echo ""
        echo "=========================================="
        echo "input_len=$input_len  |  d_model=$d_model"
        echo "=========================================="
        echo ""
        echo "" >> "$SUMMARY_FILE"
        echo "--- input_len=$input_len | d_model=$d_model ---" >> "$SUMMARY_FILE"

        run_htults_pretrain   $input_len $d_model || echo "WARNING: HtulTS pretrain failed"
        run_timedart_pretrain $input_len $d_model || echo "WARNING: TimeDART pretrain failed"

        run_htults_finetune   $input_len $d_model || echo "WARNING: HtulTS finetune failed"
        run_timedart_finetune $input_len $d_model || echo "WARNING: TimeDART finetune failed"

        extract_metrics "$LOG_DIR/htults_finetune_il${input_len}_dm${d_model}.log"   "HtulTS   " $input_len $d_model
        extract_metrics "$LOG_DIR/timedart_finetune_il${input_len}_dm${d_model}.log" "TimeDART " $input_len $d_model
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results: $SUMMARY_FILE"
echo "Logs:    $LOG_DIR"
echo "=========================================="
echo ""
cat "$SUMMARY_FILE"
