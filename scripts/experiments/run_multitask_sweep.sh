#!/bin/bash

# =============================================================================
# Multi-Task Parameter Exploration
# =============================================================================
# Systematic exploration of multi-task training parameters:
# - Lambda values (classification loss weight)
# - Schedule types (constant, linear, cosine, warmup)
# - Combined with noise and forgetting mechanisms
# =============================================================================

set -e

OUTPUT_DIR="./outputs/experiments/multitask_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

MODEL="HtulTS"
EPOCHS=20
BATCH=32
LR=0.0001

echo "=============================================="
echo "Multi-Task Parameter Exploration"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# =============================================================================
# 1. Lambda Exploration on Epilepsy
# =============================================================================

echo ""
echo ">>> Lambda Exploration (Epilepsy) <<<"
echo ""

for LAMBDA in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    echo "Testing lambda=$LAMBDA..."
    python run.py \
        --task_name finetune \
        --downstream_task classification \
        --model $MODEL \
        --data Epilepsy \
        --root_path ./datasets/Epilepsy \
        --input_len 178 \
        --num_classes 2 \
        --train_epochs $EPOCHS \
        --batch_size $BATCH \
        --learning_rate $LR \
        --patience 7 \
        --use_noise 1 \
        --noise_level 0.15 \
        --multi_task 1 \
        --multi_task_lambda $LAMBDA \
        --multi_task_schedule constant \
        --model_id "epilepsy_lambda_${LAMBDA}" \
        2>&1 | tee "$OUTPUT_DIR/epilepsy_lambda_${LAMBDA}.log"
done

# =============================================================================
# 2. Schedule Exploration on Epilepsy (with best lambda candidates)
# =============================================================================

echo ""
echo ">>> Schedule Exploration (Epilepsy) <<<"
echo ""

for SCHEDULE in constant linear cosine warmup; do
    for LAMBDA in 0.3 0.5; do
        echo "Testing schedule=$SCHEDULE, lambda=$LAMBDA..."
        python run.py \
            --task_name finetune \
            --downstream_task classification \
            --model $MODEL \
            --data Epilepsy \
            --root_path ./datasets/Epilepsy \
            --input_len 178 \
            --num_classes 2 \
            --train_epochs $EPOCHS \
            --batch_size $BATCH \
            --learning_rate $LR \
            --patience 7 \
            --use_noise 1 \
            --noise_level 0.15 \
            --multi_task 1 \
            --multi_task_lambda $LAMBDA \
            --multi_task_schedule $SCHEDULE \
            --model_id "epilepsy_${SCHEDULE}_${LAMBDA}" \
            2>&1 | tee "$OUTPUT_DIR/epilepsy_${SCHEDULE}_lambda_${LAMBDA}.log"
    done
done

# =============================================================================
# 3. Multi-Task + Forgetting Combinations
# =============================================================================

echo ""
echo ">>> Multi-Task + Forgetting Combinations <<<"
echo ""

for FORGET_TYPE in activation weight; do
    for FORGET_RATE in 0.1 0.2; do
        echo "Testing multi-task + forgetting: type=$FORGET_TYPE, rate=$FORGET_RATE..."
        python run.py \
            --task_name finetune \
            --downstream_task classification \
            --model $MODEL \
            --data Epilepsy \
            --root_path ./datasets/Epilepsy \
            --input_len 178 \
            --num_classes 2 \
            --train_epochs $EPOCHS \
            --batch_size $BATCH \
            --learning_rate $LR \
            --patience 7 \
            --use_noise 1 \
            --noise_level 0.15 \
            --use_forgetting 1 \
            --forgetting_type $FORGET_TYPE \
            --forgetting_rate $FORGET_RATE \
            --multi_task 1 \
            --multi_task_lambda 0.3 \
            --multi_task_schedule warmup \
            --model_id "epilepsy_mt_forget_${FORGET_TYPE}_${FORGET_RATE}" \
            2>&1 | tee "$OUTPUT_DIR/epilepsy_mt_forget_${FORGET_TYPE}_${FORGET_RATE}.log"
    done
done

# =============================================================================
# 4. HAR Multi-Task Experiments
# =============================================================================

echo ""
echo ">>> HAR Multi-Task Experiments <<<"
echo ""

for LAMBDA in 0.2 0.3 0.5; do
    for SCHEDULE in warmup constant; do
        echo "HAR: lambda=$LAMBDA, schedule=$SCHEDULE..."
        python run.py \
            --task_name finetune \
            --downstream_task classification \
            --model $MODEL \
            --data HAR \
            --root_path ./datasets/HAR \
            --input_len 128 \
            --num_classes 6 \
            --train_epochs $EPOCHS \
            --batch_size $BATCH \
            --learning_rate $LR \
            --patience 7 \
            --use_noise 1 \
            --noise_level 0.15 \
            --multi_task 1 \
            --multi_task_lambda $LAMBDA \
            --multi_task_schedule $SCHEDULE \
            --model_id "har_${SCHEDULE}_${LAMBDA}" \
            2>&1 | tee "$OUTPUT_DIR/har_${SCHEDULE}_lambda_${LAMBDA}.log"
    done
done

# =============================================================================
# RESULTS SUMMARY
# =============================================================================

echo ""
echo "=============================================="
echo "MULTI-TASK EXPERIMENTS COMPLETE"
echo "=============================================="
echo ""

SUMMARY_FILE="$OUTPUT_DIR/results_summary.csv"
echo "experiment,accuracy,f1_score" > "$SUMMARY_FILE"

for log in "$OUTPUT_DIR"/*.log; do
    name=$(basename "$log" .log)
    # Try to extract accuracy and F1 from the last test output
    acc=$(grep -oP "Acc: \K[0-9.]+" "$log" | tail -1 || echo "N/A")
    f1=$(grep -oP "F1: \K[0-9.]+" "$log" | tail -1 || echo "N/A")
    echo "$name,$acc,$f1" >> "$SUMMARY_FILE"
    echo "$name: Acc=$acc, F1=$f1"
done

echo ""
echo "CSV summary saved to: $SUMMARY_FILE"
echo ""
