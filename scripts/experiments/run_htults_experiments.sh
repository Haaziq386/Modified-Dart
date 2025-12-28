#!/bin/bash

# =============================================================================
# HtulTS Comprehensive Experiment Suite
# =============================================================================
# This script runs experiments with various configurations:
# 1. Multi-Task training (schedule, lambda variations)
# 2. Forgetting mechanisms (type, rate variations)
# 3. Noise levels (for denoising pretraining)
#
# Datasets: ETTh1 (Forecasting), Epilepsy (Classification), HAR (Classification)
# =============================================================================

set -e  # Exit on error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Output directory for all experiment logs
OUTPUT_BASE="./outputs/experiments"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"
mkdir -p "${EXPERIMENT_DIR}"

# Log file
LOG_FILE="${EXPERIMENT_DIR}/experiment_log.txt"

# Common hyperparameters
MODEL="HtulTS"
TRAIN_EPOCHS=30
BATCH_SIZE=32
LEARNING_RATE=0.0001
PATIENCE=7
D_MODEL=512
D_FF=2048
N_HEADS=8
E_LAYERS=3

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log_message() {
    local message="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message" | tee -a "$LOG_FILE"
}

print_separator() {
    echo "=============================================================================" | tee -a "$LOG_FILE"
}

run_experiment() {
    local exp_name="$1"
    local exp_dir="${EXPERIMENT_DIR}/${exp_name}"
    mkdir -p "$exp_dir"
    
    log_message "Starting experiment: $exp_name"
    log_message "Output directory: $exp_dir"
    
    # Shift to get remaining arguments
    shift
    
    # Run the experiment and capture output
    python run.py "$@" 2>&1 | tee "${exp_dir}/output.log"
    
    log_message "Completed experiment: $exp_name"
    print_separator
}

# =============================================================================
# SECTION 1: ETTh1 FORECASTING EXPERIMENTS
# =============================================================================

run_etth1_experiments() {
    log_message "===== ETTh1 FORECASTING EXPERIMENTS ====="
    
    local DATA="ETTh1"
    local ROOT_PATH="./datasets/ETT-small"
    local DATA_PATH="ETTh1.csv"
    local INPUT_LEN=336
    local PRED_LEN=96
    
    # -----------------------------------------
    # 1.1 Baseline (no special features)
    # -----------------------------------------
    run_experiment "etth1_baseline" \
        --task_name finetune \
        --downstream_task forecast \
        --model $MODEL \
        --data $DATA \
        --root_path $ROOT_PATH \
        --data_path $DATA_PATH \
        --features M \
        --input_len $INPUT_LEN \
        --pred_len $PRED_LEN \
        --d_model $D_MODEL \
        --d_ff $D_FF \
        --n_heads $N_HEADS \
        --e_layers $E_LAYERS \
        --train_epochs $TRAIN_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --patience $PATIENCE \
        --use_noise 0 \
        --use_forgetting 0 \
        --model_id "etth1_baseline"
    
    # -----------------------------------------
    # 1.2 Noise Level Experiments
    # -----------------------------------------
    for NOISE_LEVEL in 0.05 0.1 0.15 0.2 0.3; do
        run_experiment "etth1_noise_${NOISE_LEVEL}" \
            --task_name finetune \
            --downstream_task forecast \
            --model $MODEL \
            --data $DATA \
            --root_path $ROOT_PATH \
            --data_path $DATA_PATH \
            --features M \
            --input_len $INPUT_LEN \
            --pred_len $PRED_LEN \
            --d_model $D_MODEL \
            --train_epochs $TRAIN_EPOCHS \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --patience $PATIENCE \
            --use_noise 1 \
            --noise_level $NOISE_LEVEL \
            --use_forgetting 0 \
            --model_id "etth1_noise_${NOISE_LEVEL}"
    done
    
    # -----------------------------------------
    # 1.3 Forgetting Mechanism Experiments
    # -----------------------------------------
    for FORGET_TYPE in activation weight adaptive; do
        for FORGET_RATE in 0.05 0.1 0.2 0.3; do
            run_experiment "etth1_forget_${FORGET_TYPE}_${FORGET_RATE}" \
                --task_name finetune \
                --downstream_task forecast \
                --model $MODEL \
                --data $DATA \
                --root_path $ROOT_PATH \
                --data_path $DATA_PATH \
                --features M \
                --input_len $INPUT_LEN \
                --pred_len $PRED_LEN \
                --d_model $D_MODEL \
                --train_epochs $TRAIN_EPOCHS \
                --batch_size $BATCH_SIZE \
                --learning_rate $LEARNING_RATE \
                --patience $PATIENCE \
                --use_noise 1 \
                --noise_level 0.15 \
                --use_forgetting 1 \
                --forgetting_type $FORGET_TYPE \
                --forgetting_rate $FORGET_RATE \
                --model_id "etth1_forget_${FORGET_TYPE}_${FORGET_RATE}"
        done
    done
    
    # -----------------------------------------
    # 1.4 Different Prediction Lengths
    # -----------------------------------------
    for PRED in 96 192 336 720; do
        run_experiment "etth1_pred_${PRED}" \
            --task_name finetune \
            --downstream_task forecast \
            --model $MODEL \
            --data $DATA \
            --root_path $ROOT_PATH \
            --data_path $DATA_PATH \
            --features M \
            --input_len $INPUT_LEN \
            --pred_len $PRED \
            --d_model $D_MODEL \
            --train_epochs $TRAIN_EPOCHS \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --patience $PATIENCE \
            --use_noise 1 \
            --noise_level 0.15 \
            --model_id "etth1_pred_${PRED}"
    done
}

# =============================================================================
# SECTION 2: EPILEPSY CLASSIFICATION EXPERIMENTS
# =============================================================================

run_epilepsy_experiments() {
    log_message "===== EPILEPSY CLASSIFICATION EXPERIMENTS ====="
    
    local DATA="Epilepsy"
    local ROOT_PATH="./datasets/Epilepsy"
    local INPUT_LEN=178  # Epilepsy sequence length
    local NUM_CLASSES=2
    
    # -----------------------------------------
    # 2.1 Baseline Classification
    # -----------------------------------------
    run_experiment "epilepsy_baseline" \
        --task_name finetune \
        --downstream_task classification \
        --model $MODEL \
        --data $DATA \
        --root_path $ROOT_PATH \
        --input_len $INPUT_LEN \
        --num_classes $NUM_CLASSES \
        --train_epochs $TRAIN_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --patience $PATIENCE \
        --use_noise 0 \
        --use_forgetting 0 \
        --model_id "epilepsy_baseline"
    
    # -----------------------------------------
    # 2.2 Noise Level Experiments for Classification
    # -----------------------------------------
    for NOISE_LEVEL in 0.05 0.1 0.15 0.2; do
        run_experiment "epilepsy_noise_${NOISE_LEVEL}" \
            --task_name finetune \
            --downstream_task classification \
            --model $MODEL \
            --data $DATA \
            --root_path $ROOT_PATH \
            --input_len $INPUT_LEN \
            --num_classes $NUM_CLASSES \
            --train_epochs $TRAIN_EPOCHS \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --patience $PATIENCE \
            --use_noise 1 \
            --noise_level $NOISE_LEVEL \
            --model_id "epilepsy_noise_${NOISE_LEVEL}"
    done
    
    # -----------------------------------------
    # 2.3 Forgetting Mechanism Experiments
    # -----------------------------------------
    for FORGET_TYPE in activation weight adaptive; do
        for FORGET_RATE in 0.1 0.2; do
            run_experiment "epilepsy_forget_${FORGET_TYPE}_${FORGET_RATE}" \
                --task_name finetune \
                --downstream_task classification \
                --model $MODEL \
                --data $DATA \
                --root_path $ROOT_PATH \
                --input_len $INPUT_LEN \
                --num_classes $NUM_CLASSES \
                --train_epochs $TRAIN_EPOCHS \
                --batch_size $BATCH_SIZE \
                --learning_rate $LEARNING_RATE \
                --patience $PATIENCE \
                --use_noise 1 \
                --noise_level 0.15 \
                --use_forgetting 1 \
                --forgetting_type $FORGET_TYPE \
                --forgetting_rate $FORGET_RATE \
                --model_id "epilepsy_forget_${FORGET_TYPE}_${FORGET_RATE}"
        done
    done
    
    # -----------------------------------------
    # 2.4 Multi-Task Experiments (if forecast data available)
    # Note: Epilepsy is classification-only, so multi-task 
    # uses reconstruction as auxiliary task
    # -----------------------------------------
    for LAMBDA in 0.2 0.3 0.5 0.7; do
        for SCHEDULE in constant warmup cosine; do
            run_experiment "epilepsy_multitask_l${LAMBDA}_${SCHEDULE}" \
                --task_name finetune \
                --downstream_task classification \
                --model $MODEL \
                --data $DATA \
                --root_path $ROOT_PATH \
                --input_len $INPUT_LEN \
                --num_classes $NUM_CLASSES \
                --train_epochs $TRAIN_EPOCHS \
                --batch_size $BATCH_SIZE \
                --learning_rate $LEARNING_RATE \
                --patience $PATIENCE \
                --use_noise 1 \
                --noise_level 0.15 \
                --multi_task 1 \
                --multi_task_lambda $LAMBDA \
                --multi_task_schedule $SCHEDULE \
                --model_id "epilepsy_multitask_l${LAMBDA}_${SCHEDULE}"
        done
    done
}

# =============================================================================
# SECTION 3: HAR CLASSIFICATION EXPERIMENTS
# =============================================================================

run_har_experiments() {
    log_message "===== HAR CLASSIFICATION EXPERIMENTS ====="
    
    local DATA="HAR"
    local ROOT_PATH="./datasets/HAR"
    local INPUT_LEN=128  # HAR sequence length
    local NUM_CLASSES=6
    
    # -----------------------------------------
    # 3.1 Baseline Classification
    # -----------------------------------------
    run_experiment "har_baseline" \
        --task_name finetune \
        --downstream_task classification \
        --model $MODEL \
        --data $DATA \
        --root_path $ROOT_PATH \
        --input_len $INPUT_LEN \
        --num_classes $NUM_CLASSES \
        --train_epochs $TRAIN_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --patience $PATIENCE \
        --use_noise 0 \
        --use_forgetting 0 \
        --model_id "har_baseline"
    
    # -----------------------------------------
    # 3.2 Best configurations from Epilepsy
    # -----------------------------------------
    for NOISE_LEVEL in 0.1 0.15 0.2; do
        run_experiment "har_noise_${NOISE_LEVEL}" \
            --task_name finetune \
            --downstream_task classification \
            --model $MODEL \
            --data $DATA \
            --root_path $ROOT_PATH \
            --input_len $INPUT_LEN \
            --num_classes $NUM_CLASSES \
            --train_epochs $TRAIN_EPOCHS \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --patience $PATIENCE \
            --use_noise 1 \
            --noise_level $NOISE_LEVEL \
            --model_id "har_noise_${NOISE_LEVEL}"
    done
    
    # -----------------------------------------
    # 3.3 Forgetting Experiments
    # -----------------------------------------
    for FORGET_TYPE in activation weight; do
        run_experiment "har_forget_${FORGET_TYPE}_0.1" \
            --task_name finetune \
            --downstream_task classification \
            --model $MODEL \
            --data $DATA \
            --root_path $ROOT_PATH \
            --input_len $INPUT_LEN \
            --num_classes $NUM_CLASSES \
            --train_epochs $TRAIN_EPOCHS \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --patience $PATIENCE \
            --use_noise 1 \
            --noise_level 0.15 \
            --use_forgetting 1 \
            --forgetting_type $FORGET_TYPE \
            --forgetting_rate 0.1 \
            --model_id "har_forget_${FORGET_TYPE}_0.1"
    done
    
    # -----------------------------------------
    # 3.4 Multi-Task Experiments
    # -----------------------------------------
    for LAMBDA in 0.3 0.5; do
        run_experiment "har_multitask_l${LAMBDA}_warmup" \
            --task_name finetune \
            --downstream_task classification \
            --model $MODEL \
            --data $DATA \
            --root_path $ROOT_PATH \
            --input_len $INPUT_LEN \
            --num_classes $NUM_CLASSES \
            --train_epochs $TRAIN_EPOCHS \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --patience $PATIENCE \
            --use_noise 1 \
            --noise_level 0.15 \
            --multi_task 1 \
            --multi_task_lambda $LAMBDA \
            --multi_task_schedule warmup \
            --model_id "har_multitask_l${LAMBDA}_warmup"
    done
}

# =============================================================================
# SECTION 4: JOINT MULTI-TASK TRAINING (ETTh1 + Classification Dataset)
# =============================================================================
# 
# NOTE ON MULTI-TASK:
# - Multi-task requires BOTH forecasting AND classification data
# - ETTh1 alone has NO class labels, so multi-task needs a paired classification dataset
# - Options:
#   1. ETTh1 (forecast) + Epilepsy (classify) - joint training
#   2. ETTh1 (forecast) + HAR (classify) - joint training
#   3. Classification datasets can use reconstruction as auxiliary forecast task
#
# =============================================================================

run_joint_multitask_experiments() {
    log_message "===== JOINT MULTI-TASK: ETTh1 + CLASSIFICATION ====="
    
    # -----------------------------------------
    # 4.1 ETTh1 Forecasting + Epilepsy Classification (Joint)
    # -----------------------------------------
    for LAMBDA in 0.3 0.5 0.7; do
        for SCHEDULE in constant warmup cosine; do
            run_experiment "joint_etth1_epilepsy_l${LAMBDA}_${SCHEDULE}" \
                --task_name finetune \
                --downstream_task multi_task \
                --model $MODEL \
                --data ETTh1 \
                --root_path ./datasets/ETT-small \
                --data_path ETTh1.csv \
                --features M \
                --input_len 336 \
                --pred_len 96 \
                --cls_data Epilepsy \
                --cls_root_path ./datasets/Epilepsy \
                --num_classes 2 \
                --d_model $D_MODEL \
                --train_epochs $TRAIN_EPOCHS \
                --batch_size $BATCH_SIZE \
                --learning_rate $LEARNING_RATE \
                --patience $PATIENCE \
                --use_noise 1 \
                --noise_level 0.15 \
                --multi_task 1 \
                --multi_task_lambda $LAMBDA \
                --multi_task_schedule $SCHEDULE \
                --model_id "joint_etth1_epilepsy_l${LAMBDA}_${SCHEDULE}"
        done
    done
    
    # -----------------------------------------
    # 4.2 ETTh1 Forecasting + HAR Classification (Joint)
    # -----------------------------------------
    for LAMBDA in 0.3 0.5; do
        run_experiment "joint_etth1_har_l${LAMBDA}_warmup" \
            --task_name finetune \
            --downstream_task multi_task \
            --model $MODEL \
            --data ETTh1 \
            --root_path ./datasets/ETT-small \
            --data_path ETTh1.csv \
            --features M \
            --input_len 336 \
            --pred_len 96 \
            --cls_data HAR \
            --cls_root_path ./datasets/HAR \
            --num_classes 6 \
            --d_model $D_MODEL \
            --train_epochs $TRAIN_EPOCHS \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --patience $PATIENCE \
            --use_noise 1 \
            --noise_level 0.15 \
            --multi_task 1 \
            --multi_task_lambda $LAMBDA \
            --multi_task_schedule warmup \
            --model_id "joint_etth1_har_l${LAMBDA}_warmup"
    done
}

# =============================================================================
# SECTION 5: COMBINED BEST CONFIGURATIONS
# =============================================================================

run_combined_experiments() {
    log_message "===== COMBINED BEST CONFIGURATIONS ====="
    
    # Test combinations of best hyperparameters
    
    # -----------------------------------------
    # 5.1 ETTh1: Noise + Forgetting (no multi-task - no class labels)
    # -----------------------------------------
    run_experiment "etth1_combined_noise_forget" \
        --task_name finetune \
        --downstream_task forecast \
        --model $MODEL \
        --data ETTh1 \
        --root_path ./datasets/ETT-small \
        --data_path ETTh1.csv \
        --features M \
        --input_len 336 \
        --pred_len 96 \
        --d_model $D_MODEL \
        --train_epochs $TRAIN_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --patience $PATIENCE \
        --use_noise 1 \
        --noise_level 0.15 \
        --use_forgetting 1 \
        --forgetting_type activation \
        --forgetting_rate 0.1 \
        --model_id "etth1_combined_noise_forget"
    
    # -----------------------------------------
    # 5.2 Epilepsy: Noise + Forgetting + Multi-Task (reconstruction + classify)
    # -----------------------------------------
    run_experiment "epilepsy_combined_all" \
        --task_name finetune \
        --downstream_task classification \
        --model $MODEL \
        --data Epilepsy \
        --root_path ./datasets/Epilepsy \
        --input_len 178 \
        --num_classes 2 \
        --train_epochs $TRAIN_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --patience $PATIENCE \
        --use_noise 1 \
        --noise_level 0.15 \
        --use_forgetting 1 \
        --forgetting_type activation \
        --forgetting_rate 0.1 \
        --multi_task 1 \
        --multi_task_lambda 0.3 \
        --multi_task_schedule warmup \
        --model_id "epilepsy_combined_all"
    
    # -----------------------------------------
    # 5.3 HAR: Full combined configuration
    # -----------------------------------------
    run_experiment "har_combined_all" \
        --task_name finetune \
        --downstream_task classification \
        --model $MODEL \
        --data HAR \
        --root_path ./datasets/HAR \
        --input_len 128 \
        --num_classes 6 \
        --train_epochs $TRAIN_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --patience $PATIENCE \
        --use_noise 1 \
        --noise_level 0.15 \
        --use_forgetting 1 \
        --forgetting_type activation \
        --forgetting_rate 0.1 \
        --multi_task 1 \
        --multi_task_lambda 0.3 \
        --multi_task_schedule warmup \
        --model_id "har_combined_all"
}

# =============================================================================
# SECTION 5: RESULTS SUMMARY
# =============================================================================

generate_summary() {
    log_message "===== GENERATING RESULTS SUMMARY ====="
    
    local SUMMARY_FILE="${EXPERIMENT_DIR}/results_summary.txt"
    
    echo "===============================================================================" > "$SUMMARY_FILE"
    echo "EXPERIMENT RESULTS SUMMARY" >> "$SUMMARY_FILE"
    echo "Timestamp: $TIMESTAMP" >> "$SUMMARY_FILE"
    echo "===============================================================================" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    # Extract results from each experiment
    for exp_dir in "${EXPERIMENT_DIR}"/*/; do
        if [[ -d "$exp_dir" ]]; then
            exp_name=$(basename "$exp_dir")
            echo "--- $exp_name ---" >> "$SUMMARY_FILE"
            
            # Extract key metrics from output log
            if [[ -f "${exp_dir}/output.log" ]]; then
                # For forecasting: extract MSE, MAE
                grep -E "mse:|MSE:|mae:|MAE:|Acc:|F1:" "${exp_dir}/output.log" | tail -5 >> "$SUMMARY_FILE" 2>/dev/null || true
            fi
            echo "" >> "$SUMMARY_FILE"
        fi
    done
    
    log_message "Summary saved to: $SUMMARY_FILE"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    print_separator
    log_message "HtulTS Experiment Suite Started"
    log_message "Experiment Directory: $EXPERIMENT_DIR"
    print_separator
    
    # Parse command line arguments
    RUN_ALL=true
    RUN_ETTH1=false
    RUN_EPILEPSY=false
    RUN_HAR=false
    RUN_JOINT=false
    RUN_COMBINED=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --etth1)
                RUN_ETTH1=true
                RUN_ALL=false
                shift
                ;;
            --epilepsy)
                RUN_EPILEPSY=true
                RUN_ALL=false
                shift
                ;;
            --har)
                RUN_HAR=true
                RUN_ALL=false
                shift
                ;;
            --joint)
                RUN_JOINT=true
                RUN_ALL=false
                shift
                ;;
            --combined)
                RUN_COMBINED=true
                RUN_ALL=false
                shift
                ;;
            --all)
                RUN_ALL=true
                shift
                ;;
            *)
                echo "Unknown option: $1"
                echo "Usage: $0 [--etth1] [--epilepsy] [--har] [--joint] [--combined] [--all]"
                exit 1
                ;;
        esac
    done
    
    # Run selected experiments
    if [[ "$RUN_ALL" == true ]] || [[ "$RUN_ETTH1" == true ]]; then
        run_etth1_experiments
    fi
    
    if [[ "$RUN_ALL" == true ]] || [[ "$RUN_EPILEPSY" == true ]]; then
        run_epilepsy_experiments
    fi
    
    if [[ "$RUN_ALL" == true ]] || [[ "$RUN_HAR" == true ]]; then
        run_har_experiments
    fi
    
    if [[ "$RUN_ALL" == true ]] || [[ "$RUN_JOINT" == true ]]; then
        run_joint_multitask_experiments
    fi
    
    if [[ "$RUN_ALL" == true ]] || [[ "$RUN_COMBINED" == true ]]; then
        run_combined_experiments
    fi
    
    # Generate summary
    generate_summary
    
    print_separator
    log_message "All experiments completed!"
    log_message "Results directory: $EXPERIMENT_DIR"
    print_separator
}

# Run main function with all arguments
main "$@"
