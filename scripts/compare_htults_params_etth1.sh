#!/bin/bash

# Script to compare different HtulTS hyperparameter configurations on ETTh1
# Tests variations of tfc_weight, tfc_warmup_steps, use_real_imag, and forgetting
# Usage: bash scripts/compare_htults_params_etth1.sh

set -o pipefail

DATASET="ETTh1"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./outputs/logs/htults_params_comparison_${TIMESTAMP}"
RESULTS_DIR="./outputs/test_results/htults_params_comparison_${TIMESTAMP}"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Base parameters (same as ETTh1.sh)
INPUT_LEN=336
PRED_LENS=(96 192 336 720)
GPU=2

echo "=========================================="
echo "HtulTS Hyperparameter Comparison on ETTh1"
echo "Dataset: $DATASET"
echo "Timestamp: $TIMESTAMP"
echo "Prediction lengths: ${PRED_LENS[@]}"
echo "GPU: $GPU"
echo "=========================================="
echo ""

# Create results summary file
SUMMARY_FILE="$RESULTS_DIR/comparison_summary.txt"
echo "HtulTS Hyperparameter Comparison on ETTh1" > "$SUMMARY_FILE"
echo "Timestamp: $TIMESTAMP" >> "$SUMMARY_FILE"
echo "======================================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Define experiment configurations
# Format: "config_name|tfc_weight|tfc_warmup_steps|use_real_imag|use_forgetting|forgetting_rate|forgetting_type"

# Arrays for grid search
TFC_WEIGHTS=( 0.01 0.1)
WARMUP_STEPS=(750 1000)
USE_REAL_IMAG_VALUES=(1)

# Generate all combinations of tfc_weight, warmup_steps, and use_real_imag
CONFIGS=()
for tfc_w in "${TFC_WEIGHTS[@]}"; do
    for warmup in "${WARMUP_STEPS[@]}"; do
        for real_imag in "${USE_REAL_IMAG_VALUES[@]}"; do
            # Create descriptive name
            real_imag_label="mag_phase"
            if [ "$real_imag" = "1" ]; then
                real_imag_label="real_imag"
            fi
            config_name="w${tfc_w}_wu${warmup}_${real_imag_label}"
            CONFIGS+=("${config_name}|${tfc_w}|${warmup}|${real_imag}|0|0.0|activation")
        done
    done
done

# Add one configuration with forgetting (baseline parameters + forgetting)
CONFIGS+=("baseline_with_forgetting|0.1|1000|1|1|0.15|activation")

echo "Total configurations to test: ${#CONFIGS[@]}"
echo ""

# ============== Functions ==============

run_pretrain() {
    local config_name=$1
    local tfc_weight=$2
    local tfc_warmup_steps=$3
    local use_real_imag=$4
    local use_forgetting=$5
    local forgetting_rate=$6
    local forgetting_type=$7
    local log_file="$LOG_DIR/pretrain_${config_name}.log"

    echo ">>> Running PRETRAIN for config: $config_name"
    echo "    TFC Weight: $tfc_weight, Warmup: $tfc_warmup_steps, Real/Imag: $use_real_imag"
    echo "    Forgetting: $use_forgetting, Rate: $forgetting_rate, Type: $forgetting_type"
    echo "    Log: $log_file"

    python -u run.py \
        --task_name pretrain \
        --root_path ./datasets/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_${config_name} \
        --model HtulTS \
        --data ETTh1 \
        --features M \
        --input_len $INPUT_LEN \
        --enc_in 7 \
        --d_model 128 \
        --patch_len 16 \
        --stride 8 \
        --learning_rate 0.0001 \
        --batch_size 16 \
        --use_noise 0 \
        --train_epochs 50 \
        --lr_decay 0.95 \
        --gpu $GPU \
        --use_forgetting $use_forgetting \
        --forgetting_rate $forgetting_rate \
        --forgetting_type $forgetting_type \
        --tfc_weight $tfc_weight \
        --tfc_warmup_steps $tfc_warmup_steps \
        --projection_dim 128 \
        --use_real_imag $use_real_imag \
        2>&1 | tee "$log_file"

    echo "    ✓ Pretrain completed for $config_name"
    echo ""
}

run_finetune() {
    local config_name=$1
    local pred_len=$2
    local use_real_imag=$3
    local use_forgetting=$4
    local log_file="$LOG_DIR/finetune_${config_name}_pred${pred_len}.log"

    echo ">>> Running FINETUNE for config: $config_name, pred_len=$pred_len"
    echo "    Log: $log_file"

    python -u run.py \
        --task_name finetune \
        --root_path ./datasets/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_${config_name} \
        --model HtulTS \
        --data ETTh1 \
        --features M \
        --input_len $INPUT_LEN \
        --label_len 48 \
        --pred_len $pred_len \
        --enc_in 7 \
        --d_model 128 \
        --patch_len 16 \
        --stride 8 \
        --learning_rate 0.0001 \
        --batch_size 16 \
        --patience 5 \
        --lr_decay 0.5 \
        --lradj decay \
        --use_noise 0 \
        --gpu $GPU \
        --use_forgetting $use_forgetting \
        --use_real_imag $use_real_imag \
        --projection_dim 128 \
        2>&1 | tee "$log_file"

    echo "    ✓ Finetune completed for $config_name, pred_len=$pred_len"
    echo ""
}

extract_metrics() {
    local log_file=$1
    local config_name=$2
    local pred_len=$3

    # Extract test MSE and MAE from log file
    if [ ! -f "$log_file" ]; then
        echo "$config_name | Pred=$pred_len | MSE: FILE_NOT_FOUND | MAE: N/A" >> "$SUMMARY_FILE"
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

    printf "%-20s | Pred=%-4s | MSE: %-10s | MAE: %-10s\n" "$config_name" "$pred_len" "$mse" "$mae" >> "$SUMMARY_FILE"
    return 0
}

# ============== Main Execution Loop ==============

for config in "${CONFIGS[@]}"; do
    IFS='|' read -r config_name tfc_weight tfc_warmup_steps use_real_imag use_forgetting forgetting_rate forgetting_type <<< "$config"
    
    echo ""
    echo "=========================================="
    echo "Testing Configuration: $config_name"
    echo "=========================================="
    echo ""

    # Pretrain
    echo "Starting pretraining for $config_name..."
    run_pretrain "$config_name" "$tfc_weight" "$tfc_warmup_steps" "$use_real_imag" "$use_forgetting" "$forgetting_rate" "$forgetting_type" || {
        echo "ERROR: Pretrain failed for $config_name"
    }

    echo "" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"
    echo "Configuration: $config_name" >> "$SUMMARY_FILE"
    echo "  TFC Weight: $tfc_weight" >> "$SUMMARY_FILE"
    echo "  TFC Warmup Steps: $tfc_warmup_steps" >> "$SUMMARY_FILE"
    echo "  Use Real/Imag: $use_real_imag" >> "$SUMMARY_FILE"
    echo "  Use Forgetting: $use_forgetting" >> "$SUMMARY_FILE"
    if [ "$use_forgetting" = "1" ]; then
        echo "  Forgetting Rate: $forgetting_rate" >> "$SUMMARY_FILE"
        echo "  Forgetting Type: $forgetting_type" >> "$SUMMARY_FILE"
    fi
    echo "========================================" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"

    # Finetune for all prediction lengths
    for pred_len in "${PRED_LENS[@]}"; do
        echo ""
        echo "--- Testing pred_len = $pred_len ---"
        echo ""

        run_finetune "$config_name" "$pred_len" "$use_real_imag" "$use_forgetting" || {
            echo "WARNING: Finetune encountered an issue for $config_name, pred_len=$pred_len"
        }

        # Extract metrics
        extract_metrics "$LOG_DIR/finetune_${config_name}_pred${pred_len}.log" "$config_name" "$pred_len"
    done

    echo "" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    echo "Completed configuration: $config_name"
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
