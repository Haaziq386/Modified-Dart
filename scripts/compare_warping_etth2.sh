#!/bin/bash

# Script to compare HtulTS (with warping) vs SimMTM vs TimeDART on ETTh1 dataset
# Usage: bash scripts/compare_warping_etth2.sh

set -o pipefail  # Exit if any command in a pipe fails, but allow individual commands to fail

DATASET="ETTh1"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./outputs/logs/warping_comparison_${TIMESTAMP}"
RESULTS_DIR="./outputs/test_results/warping_comparison_${TIMESTAMP}"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Test different input lengths
INPUT_LENS=(96 168 240 336)

# Fixed d_model value
D_MODEL=128

# Prediction lengths
PRED_LENS=(96 192 336 720)

# GPU allocation (all models use GPU 0)
GPU=0

echo "=========================================="
echo "Warping Comparison: HtulTS (with warping) vs SimMTM vs TimeDART"
echo "Dataset: $DATASET"
echo "Timestamp: $TIMESTAMP"
echo "Testing input lengths: ${INPUT_LENS[@]}"
echo "Fixed d_model: $D_MODEL"
echo "Prediction lengths: ${PRED_LENS[@]}"
echo "GPU: $GPU"
echo "=========================================="
echo ""

# Create results summary file
SUMMARY_FILE="$RESULTS_DIR/comparison_summary.txt"
echo "Warping Comparison: HtulTS (with warping) vs SimMTM vs TimeDART on ETTh1" > "$SUMMARY_FILE"
echo "Timestamp: $TIMESTAMP" >> "$SUMMARY_FILE"
echo "Fixed d_model: $D_MODEL" >> "$SUMMARY_FILE"
echo "======================================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# ============== HtulTS Functions (with warping enabled) ==============

run_htults_pretrain() {
    local input_len=$1
    local log_file="$LOG_DIR/htults_warping_pretrain_len${input_len}.log"

    echo ">>> Running HtulTS PRETRAIN with input_len=$input_len, d_model=$D_MODEL, warping=ENABLED"
    echo "    Log: $log_file"

    python -u run.py \
        --task_name pretrain \
        --root_path ./datasets/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_len${input_len}_warp \
        --model HtulTS \
        --data ETTh1 \
        --features M \
        --input_len $input_len \
        --enc_in 7 \
        --d_model $D_MODEL \
        --patch_len 2 \
        --stride 2 \
        --learning_rate 0.0001 \
        --batch_size 16 \
        --use_noise 0 \
        --train_epochs 50 \
        --lr_decay 0.95 \
        --gpu $GPU \
        --use_forgetting 0 \
        --tfc_weight 0.05 \
        --tfc_warmup_steps 750 \
        --projection_dim 128 \
        --use_real_imag 1 \
        --use_warping 1 \
        2>&1 | tee "$log_file"

    echo "    ✓ HtulTS pretrain (with warping) completed for input_len=$input_len"
    echo ""
}

run_htults_finetune() {
    local input_len=$1
    local pred_len=$2
    local log_file="$LOG_DIR/htults_warping_finetune_len${input_len}_pred${pred_len}.log"

    echo ">>> Running HtulTS FINETUNE with input_len=$input_len, pred_len=$pred_len, d_model=$D_MODEL, warping=ENABLED"
    echo "    Log: $log_file"

    python -u run.py \
        --task_name finetune \
        --root_path ./datasets/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_len${input_len}_warp \
        --model HtulTS \
        --data ETTh1 \
        --features M \
        --input_len $input_len \
        --label_len 48 \
        --pred_len $pred_len \
        --enc_in 7 \
        --d_model $D_MODEL \
        --patch_len 2 \
        --stride 2 \
        --learning_rate 0.0001 \
        --batch_size 16 \
        --patience 5 \
        --lr_decay 0.5 \
        --lradj decay \
        --use_noise 0 \
        --gpu $GPU \
        --use_forgetting 0 \
        --use_real_imag 1 \
        --projection_dim 128 \
        --use_warping 1 \
        2>&1 | tee "$log_file"

    echo "    ✓ HtulTS finetune (with warping) completed for input_len=$input_len, pred_len=$pred_len"
    echo ""
}

# ============== SimMTM Functions ==============

run_simmtm_pretrain() {
    local input_len=$1
    local log_file="$LOG_DIR/simmtm_pretrain_len${input_len}.log"

    echo ">>> Running SimMTM PRETRAIN with input_len=$input_len, d_model=$D_MODEL"
    echo "    Log: $log_file"

    python -u run.py \
        --task_name pretrain \
        --root_path ./datasets/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_len${input_len}_dm${D_MODEL} \
        --model SimMTM \
        --data ETTh1 \
        --features M \
        --input_len $input_len \
        --seq_len $input_len \
        --enc_in 7 \
        --d_model $D_MODEL \
        --patch_len 2 \
        --stride 2 \
        --learning_rate 0.0001 \
        --batch_size 16 \
        --use_noise 0 \
        --train_epochs 50 \
        --lr_decay 0.95 \
        --gpu $GPU \
        --use_forgetting 0 \
        --tfc_weight 0.05 \
        --tfc_warmup_steps 750 \
        --projection_dim 128 \
        --use_real_imag 1 \
        2>&1 | tee "$log_file"

    echo "    ✓ SimMTM pretrain completed for input_len=$input_len"
    echo ""
}

run_simmtm_finetune() {
    local input_len=$1
    local pred_len=$2
    local log_file="$LOG_DIR/simmtm_finetune_len${input_len}_pred${pred_len}.log"

    echo ">>> Running SimMTM FINETUNE with input_len=$input_len, pred_len=$pred_len, d_model=$D_MODEL"
    echo "    Log: $log_file"

    python -u run.py \
        --task_name finetune \
        --root_path ./datasets/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_len${input_len}_dm${D_MODEL} \
        --model SimMTM \
        --data ETTh1 \
        --features M \
        --input_len $input_len \
        --seq_len $input_len \
        --label_len 48 \
        --pred_len $pred_len \
        --enc_in 7 \
        --d_model $D_MODEL \
        --patch_len 2 \
        --stride 2 \
        --learning_rate 0.0001 \
        --batch_size 16 \
        --patience 5 \
        --lr_decay 0.5 \
        --lradj decay \
        --use_noise 0 \
        --gpu $GPU \
        --use_forgetting 0 \
        --use_real_imag 1 \
        --projection_dim 128 \
        2>&1 | tee "$log_file"

    echo "    ✓ SimMTM finetune completed for input_len=$input_len, pred_len=$pred_len"
    echo ""
}

# ============== TimeDART Functions ==============

run_timedart_pretrain() {
    local input_len=$1
    local log_file="$LOG_DIR/timedart_pretrain_len${input_len}.log"

    echo ">>> Running TimeDART PRETRAIN with input_len=$input_len, d_model=$D_MODEL"
    echo "    Log: $log_file"

    python -u run.py \
        --task_name pretrain \
        --root_path ./datasets/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_len${input_len}_dm${D_MODEL} \
        --model TimeDART \
        --data ETTh1 \
        --features M \
        --input_len $input_len \
        --e_layers 2 \
        --d_layers 1 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --d_model $D_MODEL \
        --d_ff 32 \
        --patch_len 2 \
        --stride 2 \
        --head_dropout 0.1 \
        --dropout 0.2 \
        --time_steps 1000 \
        --scheduler cosine \
        --lr_decay 0.95 \
        --learning_rate 0.0001 \
        --batch_size 16 \
        --train_epochs 50 \
        --gpu $GPU \
        2>&1 | tee "$log_file"

    echo "    ✓ TimeDART pretrain completed for input_len=$input_len"
    echo ""
}

run_timedart_finetune() {
    local input_len=$1
    local pred_len=$2
    local log_file="$LOG_DIR/timedart_finetune_len${input_len}_pred${pred_len}.log"

    echo ">>> Running TimeDART FINETUNE with input_len=$input_len, pred_len=$pred_len, d_model=$D_MODEL"
    echo "    Log: $log_file"

    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_len${input_len}_dm${D_MODEL} \
        --model TimeDART \
        --data ETTh1 \
        --features M \
        --input_len $input_len \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --d_model $D_MODEL \
        --d_ff 32 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.4 \
        --head_dropout 0.1 \
        --batch_size 16 \
        --gpu $GPU \
        --lr_decay 0.5 \
        --lradj decay \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0001 \
        --pct_start 0.3 \
        2>&1 | tee "$log_file"

    echo "    ✓ TimeDART finetune completed for input_len=$input_len, pred_len=$pred_len"
    echo ""
}

# ============== Metrics Extraction ==============

extract_metrics() {
    local log_file=$1
    local model_name=$2
    local input_len=$3
    local pred_len=$4

    # Extract test MSE and MAE from log file
    if [ ! -f "$log_file" ]; then
        echo "$model_name | Input=$input_len | Pred=$pred_len | MSE: FILE_NOT_FOUND | MAE: N/A" >> "$SUMMARY_FILE"
        return 0
    fi

    mse=$(grep -E "[0-9]+->[0-9]+, mse:" "$log_file" 2>/dev/null | tail -1 | grep -oP "mse:\K[0-9.]+" 2>/dev/null || echo "N/A")
    mae=$(grep -E "[0-9]+->[0-9]+, mse:" "$log_file" 2>/dev/null | tail -1 | grep -oP "mae:\K[0-9.]+" 2>/dev/null || echo "N/A")

    if [ -z "$mse" ] || [ "$mse" = "" ]; then
        mse="N/A"
    fi
    if [ -z "$mae" ] || [ "$mae" = "" ]; then
        mae="N/A"
    fi

    echo "$model_name | Input=$input_len | Pred=$pred_len | MSE: $mse | MAE: $mae" >> "$SUMMARY_FILE"
    return 0
}

# ============== Main Execution Loop ==============

for input_len in "${INPUT_LENS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing INPUT_LEN = $input_len (d_model=$D_MODEL)"
    echo "=========================================="
    echo ""

    # Pretrain all three models
    echo "Starting pretraining phase for input_len=$input_len..."
    run_htults_pretrain $input_len || { echo "ERROR: HtulTS pretrain failed"; }
    run_simmtm_pretrain $input_len || { echo "ERROR: SimMTM pretrain failed"; }
    run_timedart_pretrain $input_len || { echo "ERROR: TimeDART pretrain failed"; }

    echo "" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"
    echo "INPUT LENGTH = $input_len (d_model=$D_MODEL)" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"

    # Finetune all three models for all prediction lengths
    for pred_len in "${PRED_LENS[@]}"; do
        echo ""
        echo "--- Testing pred_len = $pred_len ---"
        echo ""

        echo "  Running HtulTS finetune (with warping)..."
        run_htults_finetune $input_len $pred_len || echo "  WARNING: HtulTS finetune encountered an issue"

        echo "  Running SimMTM finetune..."
        run_simmtm_finetune $input_len $pred_len || echo "  WARNING: SimMTM finetune encountered an issue"

        echo "  Running TimeDART finetune..."
        run_timedart_finetune $input_len $pred_len || echo "  WARNING: TimeDART finetune encountered an issue"

        # Extract and log metrics
        echo "" >> "$SUMMARY_FILE"
        echo "Prediction Length = $pred_len:" >> "$SUMMARY_FILE"
        extract_metrics "$LOG_DIR/htults_warping_finetune_len${input_len}_pred${pred_len}.log" "HtulTS+Warp" $input_len $pred_len
        extract_metrics "$LOG_DIR/simmtm_finetune_len${input_len}_pred${pred_len}.log" "SimMTM     " $input_len $pred_len
        extract_metrics "$LOG_DIR/timedart_finetune_len${input_len}_pred${pred_len}.log" "TimeDART   " $input_len $pred_len

        echo "  Completed pred_len=$pred_len"
    done

    echo ""
    echo "Completed input_len=$input_len"
    echo ""
done

echo ""
echo "=========================================="
echo "All experiments completed successfully!"
echo "=========================================="
echo ""
echo "Results summary: $SUMMARY_FILE"
echo "Detailed logs: $LOG_DIR"
echo "Test results: $RESULTS_DIR"
echo ""
echo "View summary:"
echo "  cat $SUMMARY_FILE"
echo ""

# Display the summary
cat "$SUMMARY_FILE"
