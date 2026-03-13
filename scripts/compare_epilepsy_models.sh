#!/bin/bash

# Script to compare HtulTS vs SimMTM vs TimeDART on Epilepsy dataset
# Usage: bash scripts/compare_epilepsy_models.sh
# Classification task on Epilepsy dataset
# NOTE: Epilepsy has fixed 206-length sequences, so we test different patch configurations

set -o pipefail  # Exit if any command in a pipe fails, but allow individual commands to fail

DATASET="Epilepsy"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./outputs/logs/epilepsy_model_comparison_${TIMESTAMP}"
RESULTS_DIR="./outputs/test_results/epilepsy_model_comparison_${TIMESTAMP}"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Epilepsy dataset has fixed 206-length sequences (3 channels, 4 classes)
# Cannot vary input_len for classification - must match data length
INPUT_LEN=206

# Test different patch configurations (patch_len, stride)
# These affect how the model processes the fixed-length sequences
# Format: "patch_len:stride"
PATCH_CONFIGS=("6:6" "8:4" "10:5" "12:6" "16:8")

# GPU allocation - all models use GPU 0
GPU=0

echo "=========================================="
echo "Epilepsy Model Comparison: HtulTS vs SimMTM vs TimeDART"
echo "Dataset: $DATASET"
echo "Timestamp: $TIMESTAMP"
echo "Input Length: $INPUT_LEN (fixed by dataset)"
echo "Testing patch configs: ${PATCH_CONFIGS[@]}"
echo "Task: Classification"
echo "GPU: $GPU"
echo "=========================================="
echo ""

# Create results summary file
SUMMARY_FILE="$RESULTS_DIR/comparison_summary.txt"
echo "Epilepsy Model Comparison: HtulTS vs SimMTM vs TimeDART" > "$SUMMARY_FILE"
echo "Timestamp: $TIMESTAMP" >> "$SUMMARY_FILE"
echo "Task: Classification (4 classes)" >> "$SUMMARY_FILE"
echo "Input Length: $INPUT_LEN (fixed by dataset)" >> "$SUMMARY_FILE"
echo "============================================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# ============== HtulTS Functions ==============

# Function to run HtulTS pretrain
run_htults_pretrain() {
    local patch_len=$1
    local stride=$2
    local log_file="$LOG_DIR/htults_pretrain_p${patch_len}_s${stride}.log"
    
    echo ">>> Running HtulTS PRETRAIN with patch_len=$patch_len, stride=$stride"
    echo "    Log: $log_file"
    
    python -u run.py \
        --task_name pretrain \
        --downstream_task classification \
        --root_path datasets/Epilepsy/ \
        --model_id Epilepsy_p${patch_len}_s${stride} \
        --model HtulTS \
        --data Epilepsy \
        --input_len $INPUT_LEN \
        --enc_in 3 \
        --dec_in 3 \
        --c_out 3 \
        --num_classes 4 \
        --e_layers 2 \
        --d_layers 1 \
        --n_heads 16 \
        --d_model 8 \
        --d_ff 256 \
        --patch_len $patch_len \
        --stride $stride \
        --head_dropout 0.1 \
        --dropout 0.2 \
        --time_steps 1000 \
        --scheduler cosine \
        --batch_size 16 \
        --learning_rate 0.001 \
        --train_epochs 50 \
        --lr_decay 0.95 \
        --gpu $GPU \
        --use_noise 1 \
        --noise_level 0.15 \
        --use_norm 0 \
        --use_forgetting 0 \
        2>&1 | tee "$log_file"
    
    echo "    ✓ HtulTS pretrain completed for patch_len=$patch_len, stride=$stride"
    echo ""
}

# Function to run HtulTS finetune
run_htults_finetune() {
    local patch_len=$1
    local stride=$2
    local log_file="$LOG_DIR/htults_finetune_p${patch_len}_s${stride}.log"
    
    echo ">>> Running HtulTS FINETUNE with patch_len=$patch_len, stride=$stride"
    echo "    Log: $log_file"
    
    python -u run.py \
        --task_name finetune \
        --downstream_task classification \
        --root_path datasets/Epilepsy/ \
        --model_id Epilepsy_p${patch_len}_s${stride} \
        --model HtulTS \
        --data Epilepsy \
        --input_len $INPUT_LEN \
        --enc_in 3 \
        --dec_in 3 \
        --c_out 3 \
        --num_classes 4 \
        --e_layers 2 \
        --d_layers 1 \
        --n_heads 16 \
        --d_model 8 \
        --d_ff 256 \
        --patch_len $patch_len \
        --stride $stride \
        --head_dropout 0.1 \
        --dropout 0.2 \
        --time_steps 1000 \
        --scheduler cosine \
        --batch_size 16 \
        --learning_rate 0.001 \
        --patience 100 \
        --train_epochs 100 \
        --lr_decay 1.0 \
        --lradj decay \
        --gpu $GPU \
        --use_norm 0 \
        --use_forgetting 0 \
        --pct_start 0.3 \
        2>&1 | tee "$log_file"
    
    echo "    ✓ HtulTS finetune completed for patch_len=$patch_len, stride=$stride"
    echo ""
}

# ============== SimMTM Functions ==============

# Function to run SimMTM pretrain
run_simmtm_pretrain() {
    local patch_len=$1
    local stride=$2
    local log_file="$LOG_DIR/simmtm_pretrain_p${patch_len}_s${stride}.log"
    
    echo ">>> Running SimMTM PRETRAIN with patch_len=$patch_len, stride=$stride"
    echo "    Log: $log_file"
    
    python -u run.py \
        --task_name pretrain \
        --downstream_task classification \
        --root_path datasets/Epilepsy/ \
        --model_id Epilepsy_p${patch_len}_s${stride} \
        --model SimMTM \
        --data Epilepsy \
        --input_len $INPUT_LEN \
        --seq_len $INPUT_LEN \
        --enc_in 3 \
        --dec_in 3 \
        --c_out 3 \
        --num_classes 4 \
        --e_layers 2 \
        --d_layers 1 \
        --n_heads 16 \
        --d_model 8 \
        --d_ff 256 \
        --patch_len $patch_len \
        --stride $stride \
        --head_dropout 0.1 \
        --dropout 0.2 \
        --time_steps 1000 \
        --scheduler cosine \
        --batch_size 16 \
        --learning_rate 0.001 \
        --train_epochs 50 \
        --lr_decay 0.95 \
        --gpu $GPU \
        --use_noise 1 \
        --noise_level 0.15 \
        --use_norm 0 \
        --use_forgetting 0 \
        --tfc_weight 0.05 \
        --tfc_warmup_steps 750 \
        --projection_dim 128 \
        2>&1 | tee "$log_file"
    
    echo "    ✓ SimMTM pretrain completed for patch_len=$patch_len, stride=$stride"
    echo ""
}

# Function to run SimMTM finetune
run_simmtm_finetune() {
    local patch_len=$1
    local stride=$2
    local log_file="$LOG_DIR/simmtm_finetune_p${patch_len}_s${stride}.log"
    
    echo ">>> Running SimMTM FINETUNE with patch_len=$patch_len, stride=$stride"
    echo "    Log: $log_file"
    
    python -u run.py \
        --task_name finetune \
        --downstream_task classification \
        --root_path datasets/Epilepsy/ \
        --model_id Epilepsy_p${patch_len}_s${stride} \
        --model SimMTM \
        --data Epilepsy \
        --input_len $INPUT_LEN \
        --seq_len $INPUT_LEN \
        --enc_in 3 \
        --dec_in 3 \
        --c_out 3 \
        --num_classes 4 \
        --e_layers 2 \
        --d_layers 1 \
        --n_heads 16 \
        --d_model 8 \
        --d_ff 256 \
        --patch_len $patch_len \
        --stride $stride \
        --head_dropout 0.1 \
        --dropout 0.2 \
        --time_steps 1000 \
        --scheduler cosine \
        --batch_size 16 \
        --learning_rate 0.001 \
        --patience 100 \
        --train_epochs 100 \
        --lr_decay 1.0 \
        --lradj decay \
        --gpu $GPU \
        --use_norm 0 \
        --use_forgetting 0 \
        --pct_start 0.3 \
        2>&1 | tee "$log_file"
    
    echo "    ✓ SimMTM finetune completed for patch_len=$patch_len, stride=$stride"
    echo ""
}

# ============== TimeDART Functions ==============

# Function to run TimeDART pretrain
run_timedart_pretrain() {
    local patch_len=$1
    local stride=$2
    local log_file="$LOG_DIR/timedart_pretrain_p${patch_len}_s${stride}.log"
    
    echo ">>> Running TimeDART PRETRAIN with patch_len=$patch_len, stride=$stride"
    echo "    Log: $log_file"
    
    python -u run.py \
        --task_name pretrain \
        --downstream_task classification \
        --root_path datasets/Epilepsy/ \
        --model_id Epilepsy_p${patch_len}_s${stride} \
        --model TimeDART \
        --data Epilepsy \
        --input_len $INPUT_LEN \
        --enc_in 3 \
        --dec_in 3 \
        --c_out 3 \
        --num_classes 4 \
        --e_layers 2 \
        --d_layers 1 \
        --n_heads 8 \
        --d_model 8 \
        --d_ff 256 \
        --patch_len $patch_len \
        --stride $stride \
        --head_dropout 0.1 \
        --dropout 0.2 \
        --time_steps 1000 \
        --scheduler cosine \
        --batch_size 16 \
        --learning_rate 0.001 \
        --train_epochs 50 \
        --lr_decay 0.95 \
        --gpu $GPU \
        --use_norm 0 \
        --use_forgetting 0 \
        2>&1 | tee "$log_file"
    
    echo "    ✓ TimeDART pretrain completed for patch_len=$patch_len, stride=$stride"
    echo ""
}

# Function to run TimeDART finetune
run_timedart_finetune() {
    local patch_len=$1
    local stride=$2
    local log_file="$LOG_DIR/timedart_finetune_p${patch_len}_s${stride}.log"
    
    echo ">>> Running TimeDART FINETUNE with patch_len=$patch_len, stride=$stride"
    echo "    Log: $log_file"
    
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --downstream_task classification \
        --root_path datasets/Epilepsy/ \
        --model_id Epilepsy_p${patch_len}_s${stride} \
        --model TimeDART \
        --data Epilepsy \
        --input_len $INPUT_LEN \
        --enc_in 3 \
        --dec_in 3 \
        --c_out 3 \
        --num_classes 4 \
        --e_layers 2 \
        --d_layers 1 \
        --n_heads 8 \
        --d_model 8 \
        --d_ff 256 \
        --patch_len $patch_len \
        --stride $stride \
        --head_dropout 0.1 \
        --dropout 0.2 \
        --time_steps 1000 \
        --scheduler cosine \
        --batch_size 16 \
        --learning_rate 0.001 \
        --patience 100 \
        --train_epochs 100 \
        --lr_decay 1.0 \
        --lradj decay \
        --gpu $GPU \
        --pct_start 0.3 \
        2>&1 | tee "$log_file"
    
    echo "    ✓ TimeDART finetune completed for patch_len=$patch_len, stride=$stride"
    echo ""
}

# ============== Metrics Extraction ==============

# Function to extract metrics from log file
extract_metrics() {
    local log_file=$1
    local model_name=$2
    local patch_len=$3
    local stride=$4
    
    # Extract test accuracy from log file for classification task
    if [ ! -f "$log_file" ]; then
        echo "$model_name | Patch=$patch_len Stride=$stride | Accuracy: FILE_NOT_FOUND | Loss: N/A" >> "$SUMMARY_FILE"
        return 0
    fi
    
    # For classification, extract accuracy and loss
    accuracy=$(grep -iE "test.*accuracy|accuracy.*test" "$log_file" 2>/dev/null | tail -1 | grep -oP "accuracy[:\s=]*\K[0-9.]+(e-?[0-9]+)?" 2>/dev/null || echo "N/A")
    loss=$(grep -iE "test.*loss|loss.*test" "$log_file" 2>/dev/null | tail -1 | grep -oP "loss[:\s=]*\K[0-9.]+(e-?[0-9]+)?" 2>/dev/null || echo "N/A")
    
    if [ -z "$accuracy" ] || [ "$accuracy" = "" ]; then
        accuracy="N/A"
    fi
    if [ -z "$loss" ] || [ "$loss" = "" ]; then
        loss="N/A"
    fi
    
    echo "$model_name | Patch=$patch_len Stride=$stride | Accuracy: $accuracy | Loss: $loss" >> "$SUMMARY_FILE"
    return 0
}

# ============== Main Execution Loop ==============

for config in "${PATCH_CONFIGS[@]}"; do
    # Parse patch_len and stride from config (format: "patch_len:stride")
    patch_len=$(echo "$config" | cut -d':' -f1)
    stride=$(echo "$config" | cut -d':' -f2)
    
    echo ""
    echo "=========================================="
    echo "Testing PATCH_LEN=$patch_len, STRIDE=$stride"
    echo "=========================================="
    echo ""
    
    # Pretrain all three models in sequence
    echo "Starting pretraining phase for patch_len=$patch_len, stride=$stride..."
    run_htults_pretrain $patch_len $stride || { echo "ERROR: HtulTS pretrain failed"; }
    run_simmtm_pretrain $patch_len $stride || { echo "ERROR: SimMTM pretrain failed"; }
    run_timedart_pretrain $patch_len $stride || { echo "ERROR: TimeDART pretrain failed"; }
    
    echo "" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"
    echo "PATCH_LEN=$patch_len, STRIDE=$stride" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    # Finetune all three models
    echo ""
    echo "Starting finetuning phase for patch_len=$patch_len, stride=$stride..."
    echo ""
    
    echo "  Running HtulTS finetune..."
    run_htults_finetune $patch_len $stride || echo "  WARNING: HtulTS finetune encountered an issue"
    
    echo "  Running SimMTM finetune..."
    run_simmtm_finetune $patch_len $stride || echo "  WARNING: SimMTM finetune encountered an issue"
    
    echo "  Running TimeDART finetune..."
    run_timedart_finetune $patch_len $stride || echo "  WARNING: TimeDART finetune encountered an issue"
    
    # Extract and log metrics
    echo "" >> "$SUMMARY_FILE"
    echo "Finetuning Results:" >> "$SUMMARY_FILE"
    extract_metrics "$LOG_DIR/htults_finetune_p${patch_len}_s${stride}.log" "HtulTS   " $patch_len $stride
    extract_metrics "$LOG_DIR/simmtm_finetune_p${patch_len}_s${stride}.log" "SimMTM   " $patch_len $stride
    extract_metrics "$LOG_DIR/timedart_finetune_p${patch_len}_s${stride}.log" "TimeDART " $patch_len $stride
    
    echo ""
    echo "Completed patch_len=$patch_len, stride=$stride"
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
