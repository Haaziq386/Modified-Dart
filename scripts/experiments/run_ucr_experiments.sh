#!/bin/bash

# =============================================================================
# UCR Archive Comprehensive Experiments
# =============================================================================
# Runs experiments on UCR datasets with various configurations:
# - Baseline (no noise, no forgetting)
# - Noise levels (0.1, 0.15, 0.2)
# - Forgetting mechanisms (activation, weight, adaptive)
# - Multi-task (reconstruction + classification)
# - Combined configurations
#
# Usage:
#   ./run_ucr_experiments.sh                    # Run all UCR datasets
#   ./run_ucr_experiments.sh GunPoint Wafer    # Run specific datasets
#   ./run_ucr_experiments.sh --quick            # Run subset for quick test
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

UCR_PATH="datasets/UCRArchive_2018/"
OUTPUT_BASE="./outputs/experiments/ucr"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${OUTPUT_BASE}/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Model settings
MODEL="HtulTS"
PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=100
BATCH_SIZE=16
LR=0.001
D_MODEL=128
D_FF=256
N_HEADS=16
E_LAYERS=2
PATCH_LEN=8
STRIDE=8
GPU=0

# Log file
LOG_FILE="${RESULTS_DIR}/experiment_log.txt"
RESULTS_CSV="${RESULTS_DIR}/results.csv"

# Initialize CSV
echo "dataset,config,accuracy,f1,pretrain_time,finetune_time" > "$RESULTS_CSV"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

get_dataset_info() {
    local dataset_name="$1"
    local train_file="${UCR_PATH}/${dataset_name}/${dataset_name}_TRAIN.tsv"
    
    if [ ! -f "$train_file" ]; then
        echo "ERROR ERROR"
        return
    fi
    
    python3 -c "
import pandas as pd
import numpy as np
try:
    df = pd.read_csv('$train_file', sep='\t', header=None)
    seq_len = df.shape[1] - 1
    n_classes = len(np.unique(df.iloc[:, 0]))
    print(f'{seq_len} {n_classes}')
except:
    print('ERROR ERROR')
"
}

run_single_experiment() {
    local dataset_name="$1"
    local config_name="$2"
    local input_len="$3"
    local num_classes="$4"
    shift 4
    local extra_args="$@"
    
    local exp_dir="${RESULTS_DIR}/${dataset_name}/${config_name}"
    mkdir -p "$exp_dir"
    
    log_msg ">>> ${dataset_name} - ${config_name}"
    
    # Pretrain
    local pt_start=$(date +%s)
    python -u run.py \
        --task_name pretrain \
        --downstream_task classification \
        --root_path "$UCR_PATH" \
        --model_id "${dataset_name}_${config_name}" \
        --model $MODEL \
        --data UCR \
        --data_path "${dataset_name}" \
        --e_layers $E_LAYERS \
        --d_layers 1 \
        --input_len $input_len \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --num_classes $num_classes \
        --n_heads $N_HEADS \
        --d_model $D_MODEL \
        --d_ff $D_FF \
        --patch_len $PATCH_LEN \
        --stride $STRIDE \
        --head_dropout 0.1 \
        --dropout 0.2 \
        --time_steps 1000 \
        --scheduler cosine \
        --lr_decay 0.95 \
        --learning_rate $LR \
        --batch_size $BATCH_SIZE \
        --train_epochs $PRETRAIN_EPOCHS \
        --gpu $GPU \
        --use_norm 0 \
        $extra_args \
        2>&1 | tee "${exp_dir}/pretrain.log"
    local pt_end=$(date +%s)
    local pt_time=$((pt_end - pt_start))
    
    # Finetune
    local ft_start=$(date +%s)
    python -u run.py \
        --task_name finetune \
        --downstream_task classification \
        --root_path "$UCR_PATH" \
        --model_id "${dataset_name}_${config_name}" \
        --model $MODEL \
        --data UCR \
        --data_path "${dataset_name}" \
        --e_layers $E_LAYERS \
        --d_layers 1 \
        --input_len $input_len \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --num_classes $num_classes \
        --n_heads $N_HEADS \
        --d_model $D_MODEL \
        --d_ff $D_FF \
        --patch_len $PATCH_LEN \
        --stride $STRIDE \
        --head_dropout 0.1 \
        --dropout 0.2 \
        --time_steps 1000 \
        --scheduler cosine \
        --batch_size $BATCH_SIZE \
        --gpu $GPU \
        --lr_decay 1.0 \
        --lradj decay \
        --patience 100 \
        --learning_rate $LR \
        --pct_start 0.3 \
        --train_epochs $FINETUNE_EPOCHS \
        --use_norm 0 \
        $extra_args \
        2>&1 | tee "${exp_dir}/finetune.log"
    local ft_end=$(date +%s)
    local ft_time=$((ft_end - ft_start))
    
    # Extract metrics
    local acc=$(grep -oP "Accuracy[:\s]+\K[0-9.]+" "${exp_dir}/finetune.log" 2>/dev/null | tail -1 || echo "N/A")
    local f1=$(grep -oP "F1[:\s]+\K[0-9.]+" "${exp_dir}/finetune.log" 2>/dev/null | tail -1 || echo "N/A")
    
    echo "${dataset_name},${config_name},${acc},${f1},${pt_time},${ft_time}" >> "$RESULTS_CSV"
    log_msg "    Acc: $acc, F1: $f1, PT: ${pt_time}s, FT: ${ft_time}s"
}

# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

run_all_configs() {
    local dataset_name="$1"
    local input_len="$2"
    local num_classes="$3"
    
    log_msg "=============================================="
    log_msg "Dataset: $dataset_name (len=$input_len, classes=$num_classes)"
    log_msg "=============================================="
    
    # -----------------------------------------
    # 1. BASELINE (no noise, no forgetting)
    # -----------------------------------------
    run_single_experiment "$dataset_name" "baseline" "$input_len" "$num_classes" \
        --use_noise 0 \
        --use_forgetting 0
    
    # -----------------------------------------
    # 2. NOISE LEVELS
    # -----------------------------------------
    for noise in 0.1 0.15 0.2; do
        run_single_experiment "$dataset_name" "noise_${noise}" "$input_len" "$num_classes" \
            --use_noise 1 \
            --noise_level $noise \
            --use_forgetting 0
    done
    
    # -----------------------------------------
    # 3. FORGETTING MECHANISMS (with noise=0.15)
    # -----------------------------------------
    for ftype in activation weight adaptive; do
        for frate in 0.1 0.2; do
            run_single_experiment "$dataset_name" "forget_${ftype}_${frate}" "$input_len" "$num_classes" \
                --use_noise 1 \
                --noise_level 0.15 \
                --use_forgetting 1 \
                --forgetting_type $ftype \
                --forgetting_rate $frate
        done
    done
    
    # -----------------------------------------
    # 4. MULTI-TASK (reconstruction + classification)
    # -----------------------------------------
    for lambda in 0.3 0.5 0.7; do
        for schedule in constant warmup cosine; do
            run_single_experiment "$dataset_name" "mt_l${lambda}_${schedule}" "$input_len" "$num_classes" \
                --use_noise 1 \
                --noise_level 0.15 \
                --use_forgetting 0 \
                --multi_task 1 \
                --multi_task_lambda $lambda \
                --multi_task_schedule $schedule
        done
    done
    
    # -----------------------------------------
    # 5. BEST COMBINED (noise + forgetting + multi-task)
    # -----------------------------------------
    run_single_experiment "$dataset_name" "combined_best" "$input_len" "$num_classes" \
        --use_noise 1 \
        --noise_level 0.15 \
        --use_forgetting 1 \
        --forgetting_type activation \
        --forgetting_rate 0.1 \
        --multi_task 1 \
        --multi_task_lambda 0.3 \
        --multi_task_schedule warmup
}

run_quick_configs() {
    local dataset_name="$1"
    local input_len="$2"
    local num_classes="$3"
    
    log_msg "=============================================="
    log_msg "[QUICK] Dataset: $dataset_name (len=$input_len, classes=$num_classes)"
    log_msg "=============================================="
    
    # Baseline
    run_single_experiment "$dataset_name" "baseline" "$input_len" "$num_classes" \
        --use_noise 0 \
        --use_forgetting 0
    
    # Best noise
    run_single_experiment "$dataset_name" "noise_015" "$input_len" "$num_classes" \
        --use_noise 1 \
        --noise_level 0.15 \
        --use_forgetting 0
    
    # Best forgetting
    run_single_experiment "$dataset_name" "forget_act_01" "$input_len" "$num_classes" \
        --use_noise 1 \
        --noise_level 0.15 \
        --use_forgetting 1 \
        --forgetting_type activation \
        --forgetting_rate 0.1
    
    # Multi-task
    run_single_experiment "$dataset_name" "mt_l03_warmup" "$input_len" "$num_classes" \
        --use_noise 1 \
        --noise_level 0.15 \
        --multi_task 1 \
        --multi_task_lambda 0.3 \
        --multi_task_schedule warmup
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    log_msg "=============================================="
    log_msg "UCR EXPERIMENTS STARTED"
    log_msg "Results: $RESULTS_DIR"
    log_msg "=============================================="
    
    QUICK_MODE=false
    DATASETS=()
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                QUICK_MODE=true
                shift
                ;;
            --gpu)
                GPU="$2"
                shift 2
                ;;
            *)
                DATASETS+=("$1")
                shift
                ;;
        esac
    done
    
    # If no datasets specified, use all
    if [ ${#DATASETS[@]} -eq 0 ]; then
        for folder in "$UCR_PATH"/*/; do
            dataset_name=$(basename "$folder")
            DATASETS+=("$dataset_name")
        done
    fi
    
    log_msg "Datasets to process: ${#DATASETS[@]}"
    log_msg "Quick mode: $QUICK_MODE"
    
    # Process each dataset
    for dataset_name in "${DATASETS[@]}"; do
        # Get dataset info
        read input_len num_classes <<< $(get_dataset_info "$dataset_name")
        
        if [ "$input_len" == "ERROR" ]; then
            log_msg "Skipping $dataset_name (could not read metadata)"
            continue
        fi
        
        if [ "$QUICK_MODE" == true ]; then
            run_quick_configs "$dataset_name" "$input_len" "$num_classes"
        else
            run_all_configs "$dataset_name" "$input_len" "$num_classes"
        fi
    done
    
    # Generate summary
    log_msg "=============================================="
    log_msg "EXPERIMENTS COMPLETE"
    log_msg "=============================================="
    
    echo ""
    echo "=== RESULTS SUMMARY ==="
    echo ""
    column -t -s',' "$RESULTS_CSV" | head -50
    echo ""
    echo "Full results: $RESULTS_CSV"
}

main "$@"
