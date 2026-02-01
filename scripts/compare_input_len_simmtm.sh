#!/bin/bash

# Script to compare SimMTM for various input lengths on ETTh2 dataset
# Usage: bash scripts/compare_input_lengths.sh

set -o pipefail  # Exit if any command in a pipe fails, but allow individual commands to fail

DATASET="ETTh2"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./outputs/logs/input_len_comparison_${TIMESTAMP}"
RESULTS_DIR="./outputs/test_results/input_len_comparison_${TIMESTAMP}"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Test different input lengths (less than 336)
INPUT_LENS=(96 168 240 336)
PRED_LENS=(96 192 336 720)

# GPU allocation
GPU_SIMMTM=0

echo "=========================================="
echo "Input Length Comparison: SimMTM"
echo "Dataset: $DATASET"
echo "Timestamp: $TIMESTAMP"
echo "Testing input lengths: ${INPUT_LENS[@]}"
echo "Prediction lengths: ${PRED_LENS[@]}"
echo "=========================================="
echo ""

# Create results summary file
SUMMARY_FILE="$RESULTS_DIR/comparison_summary.txt"
echo "Input Length Comparison: SimMTM on ETTh2" > "$SUMMARY_FILE"
echo "Timestamp: $TIMESTAMP" >> "$SUMMARY_FILE"
echo "======================================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Function to run SimMTM pretrain
run_simmtm_pretrain() {
    local input_len=$1
    local log_file="$LOG_DIR/simmtm_pretrain_len${input_len}.log"
    
    echo ">>> Running SimMTM PRETRAIN with input_len=$input_len"
    echo "    Log: $log_file"
    
    python -u run.py \
        --task_name pretrain \
        --root_path ./datasets/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2_len${input_len} \
        --model SimMTM \
        --data ETTh2 \
        --features M \
        --input_len $input_len \
        --seq_len $input_len \
        --enc_in 7 \
        --d_model 8 \
        --patch_len 16 \
        --stride 8 \
        --learning_rate 0.0001 \
        --batch_size 16 \
        --use_noise 0 \
        --train_epochs 50 \
        --lr_decay 0.95 \
        --gpu $GPU_SIMMTM \
        --use_forgetting 0 \
        --tfc_weight 0.05 \
        --tfc_warmup_steps 750 \
        --projection_dim 128 \
        --use_real_imag 1 \
        2>&1 | tee "$log_file"
    
    echo "    ✓ SimMTM pretrain completed for input_len=$input_len"
    echo ""
}

# Function to run SimMTM finetune
run_simmtm_finetune() {
    local input_len=$1
    local pred_len=$2
    local log_file="$LOG_DIR/simmtm_finetune_len${input_len}_pred${pred_len}.log"
    
    echo ">>> Running SimMTM FINETUNE with input_len=$input_len, pred_len=$pred_len"
    echo "    Log: $log_file"
    
    python -u run.py \
        --task_name finetune \
        --root_path ./datasets/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2_len${input_len} \
        --model SimMTM \
        --data ETTh2 \
        --features M \
        --input_len $input_len \
        --seq_len $input_len \
        --label_len 48 \
        --pred_len $pred_len \
        --enc_in 7 \
        --d_model 8 \
        --patch_len 16 \
        --stride 8 \
        --learning_rate 0.0001 \
        --batch_size 16 \
        --patience 5 \
        --lr_decay 0.5 \
        --lradj decay \
        --use_noise 0 \
        --gpu $GPU_SIMMTM \
        --use_forgetting 0 \
        --use_real_imag 1 \
        --projection_dim 128 \
        2>&1 | tee "$log_file"
    
    echo "    ✓ SimMTM finetune completed for input_len=$input_len, pred_len=$pred_len"
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
    run_simmtm_pretrain $input_len || { echo "ERROR: SimMTM pretrain failed"; continue; }
    
    echo "" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"
    echo "INPUT LENGTH = $input_len" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"
    
    # Finetune both models for all prediction lengths
    for pred_len in "${PRED_LENS[@]}"; do
        echo ""
        echo "--- Testing pred_len = $pred_len ---"
        echo ""
        
        echo "  Running SimMTM finetune..."
        run_simmtm_finetune $input_len $pred_len || echo "  WARNING: SimMTM finetune encountered an issue"
        
        # Extract and log metrics
        echo "" >> "$SUMMARY_FILE"
        echo "Prediction Length = $pred_len:" >> "$SUMMARY_FILE"
        extract_metrics "$LOG_DIR/simmtm_finetune_len${input_len}_pred${pred_len}.log" "SimMTM   " $input_len $pred_len
        
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
