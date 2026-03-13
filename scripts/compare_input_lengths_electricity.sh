#!/bin/bash

# Script to compare HtulTS vs TimeDART for various input lengths on electricity dataset
# Usage: bash scripts/compare_input_lengths.sh

set -o pipefail  # Exit if any command in a pipe fails, but allow individual commands to fail

DATASET="electricity"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./outputs/logs/input_len_comparison_${TIMESTAMP}"
RESULTS_DIR="./outputs/test_results/input_len_comparison_${TIMESTAMP}"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Test different input lengths (less than 336)
# INPUT_LENS=(96 168 240 336)
INPUT_LENS=(240 336)
PRED_LENS=(96 192 336 720)

# GPU allocation
# GPU_HTULTS=1
GPU_TIMEDART=2

echo "=========================================="
echo "Input Length Comparison: HtulTS vs TimeDART"
echo "Dataset: $DATASET"
echo "Timestamp: $TIMESTAMP"
echo "Testing input lengths: ${INPUT_LENS[@]}"
echo "Prediction lengths: ${PRED_LENS[@]}"
echo "=========================================="
echo ""

# Create results summary file
SUMMARY_FILE="$RESULTS_DIR/comparison_summary.txt"
echo "Input Length Comparison: HtulTS vs TimeDART on ETTh2" > "$SUMMARY_FILE"
echo "Timestamp: $TIMESTAMP" >> "$SUMMARY_FILE"
echo "======================================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Function to run HtulTS pretrain
# run_htults_pretrain() {
#     local input_len=$1
#     local log_file="$LOG_DIR/htults_pretrain_len${input_len}.log"
    
#     echo ">>> Running HtulTS PRETRAIN with input_len=$input_len"
#     echo "    Log: $log_file"
    
#     python -u run.py \
#         --task_name pretrain \
#         --root_path ./datasets/electricity/ \
#         --data_path electricity.csv \
#         --model_id Electricity_len${input_len} \
#         --model HtulTS \
#         --data Electricity \
#         --features M \
#         --input_len $input_len \
#         --enc_in 321 \
#         --d_model 128 \
#         --patch_len 8 \
#         --stride 8 \
#         --learning_rate 0.0001 \
#         --batch_size 16 \
#         --use_noise 0 \
#         --train_epochs 50 \
#         --lr_decay 0.95 \
#         --gpu $GPU_HTULTS \
#         --use_forgetting 0 \
#         --tfc_weight 0.05 \
#         --tfc_warmup_steps 750 \
#         --projection_dim 128 \
#         --use_real_imag 1 \
#         2>&1 | tee "$log_file"
    
#     echo "    ✓ HtulTS pretrain completed for input_len=$input_len"
#     echo ""
# }

# Function to run HtulTS finetune
# run_htults_finetune() {
#     local input_len=$1
#     local pred_len=$2
#     local log_file="$LOG_DIR/htults_finetune_len${input_len}_pred${pred_len}.log"
    
#     echo ">>> Running HtulTS FINETUNE with input_len=$input_len, pred_len=$pred_len"
#     echo "    Log: $log_file"
    
#     python -u run.py \
#         --task_name finetune \
#         --root_path ./datasets/electricity/ \
#         --data_path electricity.csv \
#         --model_id Electricity_len${input_len} \
#         --model HtulTS \
#         --data Electricity \
#         --features M \
#         --input_len $input_len \
#         --label_len 48 \
#         --pred_len $pred_len \
#         --enc_in 321 \
#         --d_model 128 \
#         --patch_len 8 \
#         --stride 8 \
#         --learning_rate 0.0004 \
#         --batch_size 16 \
#         --patience 3 \
#         --lr_decay 0.5 \
#         --lradj step \
#         --use_noise 0 \
#         --gpu $GPU_HTULTS \
#         --use_forgetting 0 \
#         --use_real_imag 1 \
#         --projection_dim 128 \
#         2>&1 | tee "$log_file"
    
#     echo "    ✓ HtulTS finetune completed for input_len=$input_len, pred_len=$pred_len"
#     echo ""
# }

# Function to run TimeDART pretrain
run_timedart_pretrain() {
    local input_len=$1
    local log_file="$LOG_DIR/timedart_pretrain_len${input_len}.log"
    
    echo ">>> Running TimeDART PRETRAIN with input_len=$input_len"
    echo "    Log: $log_file"
    
    python -u run.py \
        --task_name pretrain \
        --root_path ./datasets/electricity/ \
        --data_path electricity.csv \
        --model_id Electricity_len${input_len} \
        --model TimeDART \
        --data Electricity \
        --features M \
        --input_len $input_len \
        --e_layers 2 \
        --d_layers 1 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 256 \
        --patch_len 8 \
        --stride 8 \
        --head_dropout 0.1 \
        --dropout 0.2 \
        --time_steps 1000 \
        --scheduler cosine \
        --lr_decay 0.95 \
        --learning_rate 0.0001 \
        --batch_size 16 \
        --train_epochs 50 \
        --gpu $GPU_TIMEDART \
        2>&1 | tee "$log_file"
    
    echo "    ✓ TimeDART pretrain completed for input_len=$input_len"
    echo ""
}

# Function to run TimeDART finetune
run_timedart_finetune() {
    local input_len=$1
    local pred_len=$2
    local log_file="$LOG_DIR/timedart_finetune_len${input_len}_pred${pred_len}.log"
    
    echo ">>> Running TimeDART FINETUNE with input_len=$input_len, pred_len=$pred_len"
    echo "    Log: $log_file"
    
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/electricity/ \
        --data_path electricity.csv \
        --model_id Electricity_len${input_len} \
        --model TimeDART \
        --data Electricity \
        --features M \
        --input_len $input_len \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 7 \
        --n_heads 8 \
        --d_model 8 \
        --d_ff 32 \
        --patch_len 16 \
        --stride 8 \
        --dropout 0.4 \
        --head_dropout 0.1 \
        --batch_size 16 \
        --gpu $GPU_TIMEDART \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0004 \
        --pct_start 0.3 \
        2>&1 | tee "$log_file"
    
    echo "    ✓ TimeDART finetune completed for input_len=$input_len, pred_len=$pred_len"
    echo ""
}

# Function to extract metrics from log file
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
    
    mse=$(grep -iE "test.*mse" "$log_file" 2>/dev/null | tail -1 | grep -oP "mse[:\s=]*\K[0-9.]+(e-?[0-9]+)?" 2>/dev/null || echo "N/A")
    mae=$(grep -iE "test.*mae" "$log_file" 2>/dev/null | tail -1 | grep -oP "mae[:\s=]*\K[0-9.]+(e-?[0-9]+)?" 2>/dev/null || echo "N/A")
    
    if [ -z "$mse" ] || [ "$mse" = "" ]; then
        mse="N/A"
    fi
    if [ -z "$mae" ] || [ "$mae" = "" ]; then
        mae="N/A"
    fi
    
    echo "$model_name | Input=$input_len | Pred=$pred_len | MSE: $mse | MAE: $mae" >> "$SUMMARY_FILE"
    return 0
}

# Main execution loop
for input_len in "${INPUT_LENS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing INPUT_LEN = $input_len"
    echo "=========================================="
    echo ""
    
    # Pretrain both models
    echo "Starting pretraining phase for input_len=$input_len..."
    # run_htults_pretrain $input_len || { echo "ERROR: HtulTS pretrain failed"; continue; }
    run_timedart_pretrain $input_len || { echo "ERROR: TimeDART pretrain failed"; continue; }
    
    echo "" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"
    echo "INPUT LENGTH = $input_len" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"
    
    # Finetune both models for all prediction lengths
    for pred_len in "${PRED_LENS[@]}"; do
        echo ""
        echo "--- Testing pred_len = $pred_len ---"
        echo ""
        
        echo "  Running HtulTS finetune..."
        # run_htults_finetune $input_len $pred_len || echo "  WARNING: HtulTS finetune encountered an issue"
        
        echo "  Running TimeDART finetune..."
        run_timedart_finetune $input_len $pred_len || echo "  WARNING: TimeDART finetune encountered an issue"
        
        # Extract and log metrics
        echo "" >> "$SUMMARY_FILE"
        echo "Prediction Length = $pred_len:" >> "$SUMMARY_FILE"
        # extract_metrics "$LOG_DIR/htults_finetune_len${input_len}_pred${pred_len}.log" "HtulTS   " $input_len $pred_len
        extract_metrics "$LOG_DIR/timedart_finetune_len${input_len}_pred${pred_len}.log" "TimeDART " $input_len $pred_len
        
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
