#!/bin/bash

# =============================================================================
# Quick HtulTS Experiments - Focused Parameter Sweep
# =============================================================================
# Run this for a quick parameter sweep with fewer combinations
# Approximately 1-2 hours runtime depending on hardware
# =============================================================================

set -e

OUTPUT_DIR="./outputs/experiments/quick_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Quick Experiment Suite"
echo "Output: $OUTPUT_DIR"
echo "======================================"

# Common settings
MODEL="HtulTS"
EPOCHS=15
BATCH=32
LR=0.0001

# =============================================================================
# 1. ETTh1 FORECASTING - Quick Sweep
# =============================================================================

echo ""
echo ">>> ETTh1 Forecasting Experiments <<<"
echo ""

# Baseline
echo "[1/12] ETTh1 Baseline..."
python run.py \
    --task_name finetune --downstream_task forecast \
    --model $MODEL --data ETTh1 \
    --root_path ./datasets/ETT-small --data_path ETTh1.csv \
    --features M --input_len 336 --pred_len 96 \
    --train_epochs $EPOCHS --batch_size $BATCH --learning_rate $LR \
    --patience 5 --use_noise 0 --use_forgetting 0 \
    --model_id "etth1_baseline" 2>&1 | tee "$OUTPUT_DIR/etth1_baseline.log"

# Best noise level
echo "[2/12] ETTh1 Noise 0.15..."
python run.py \
    --task_name finetune --downstream_task forecast \
    --model $MODEL --data ETTh1 \
    --root_path ./datasets/ETT-small --data_path ETTh1.csv \
    --features M --input_len 336 --pred_len 96 \
    --train_epochs $EPOCHS --batch_size $BATCH --learning_rate $LR \
    --patience 5 --use_noise 1 --noise_level 0.15 \
    --model_id "etth1_noise_0.15" 2>&1 | tee "$OUTPUT_DIR/etth1_noise_015.log"

# Forgetting - activation
echo "[3/12] ETTh1 Forgetting (activation 0.1)..."
python run.py \
    --task_name finetune --downstream_task forecast \
    --model $MODEL --data ETTh1 \
    --root_path ./datasets/ETT-small --data_path ETTh1.csv \
    --features M --input_len 336 --pred_len 96 \
    --train_epochs $EPOCHS --batch_size $BATCH --learning_rate $LR \
    --patience 5 --use_noise 1 --noise_level 0.15 \
    --use_forgetting 1 --forgetting_type activation --forgetting_rate 0.1 \
    --model_id "etth1_forget_act" 2>&1 | tee "$OUTPUT_DIR/etth1_forget_activation.log"

# Forgetting - weight
echo "[4/12] ETTh1 Forgetting (weight 0.1)..."
python run.py \
    --task_name finetune --downstream_task forecast \
    --model $MODEL --data ETTh1 \
    --root_path ./datasets/ETT-small --data_path ETTh1.csv \
    --features M --input_len 336 --pred_len 96 \
    --train_epochs $EPOCHS --batch_size $BATCH --learning_rate $LR \
    --patience 5 --use_noise 1 --noise_level 0.15 \
    --use_forgetting 1 --forgetting_type weight --forgetting_rate 0.1 \
    --model_id "etth1_forget_wgt" 2>&1 | tee "$OUTPUT_DIR/etth1_forget_weight.log"

# =============================================================================
# 2. EPILEPSY CLASSIFICATION - Quick Sweep
# =============================================================================

echo ""
echo ">>> Epilepsy Classification Experiments <<<"
echo ""

# Baseline
echo "[5/12] Epilepsy Baseline..."
python run.py \
    --task_name finetune --downstream_task classification \
    --model $MODEL --data Epilepsy \
    --root_path ./datasets/Epilepsy \
    --input_len 178 --num_classes 2 \
    --train_epochs $EPOCHS --batch_size $BATCH --learning_rate $LR \
    --patience 5 --use_noise 0 --use_forgetting 0 \
    --model_id "epilepsy_baseline" 2>&1 | tee "$OUTPUT_DIR/epilepsy_baseline.log"

# With noise
echo "[6/12] Epilepsy Noise 0.15..."
python run.py \
    --task_name finetune --downstream_task classification \
    --model $MODEL --data Epilepsy \
    --root_path ./datasets/Epilepsy \
    --input_len 178 --num_classes 2 \
    --train_epochs $EPOCHS --batch_size $BATCH --learning_rate $LR \
    --patience 5 --use_noise 1 --noise_level 0.15 \
    --model_id "epilepsy_noise" 2>&1 | tee "$OUTPUT_DIR/epilepsy_noise.log"

# Forgetting
echo "[7/12] Epilepsy Forgetting..."
python run.py \
    --task_name finetune --downstream_task classification \
    --model $MODEL --data Epilepsy \
    --root_path ./datasets/Epilepsy \
    --input_len 178 --num_classes 2 \
    --train_epochs $EPOCHS --batch_size $BATCH --learning_rate $LR \
    --patience 5 --use_noise 1 --noise_level 0.15 \
    --use_forgetting 1 --forgetting_type activation --forgetting_rate 0.1 \
    --model_id "epilepsy_forget" 2>&1 | tee "$OUTPUT_DIR/epilepsy_forget.log"

# Multi-task lambda 0.3
echo "[8/12] Epilepsy Multi-Task (lambda=0.3)..."
python run.py \
    --task_name finetune --downstream_task classification \
    --model $MODEL --data Epilepsy \
    --root_path ./datasets/Epilepsy \
    --input_len 178 --num_classes 2 \
    --train_epochs $EPOCHS --batch_size $BATCH --learning_rate $LR \
    --patience 5 --use_noise 1 --noise_level 0.15 \
    --multi_task 1 --multi_task_lambda 0.3 --multi_task_schedule warmup \
    --model_id "epilepsy_mt_03" 2>&1 | tee "$OUTPUT_DIR/epilepsy_multitask_03.log"

# Multi-task lambda 0.5
echo "[9/12] Epilepsy Multi-Task (lambda=0.5)..."
python run.py \
    --task_name finetune --downstream_task classification \
    --model $MODEL --data Epilepsy \
    --root_path ./datasets/Epilepsy \
    --input_len 178 --num_classes 2 \
    --train_epochs $EPOCHS --batch_size $BATCH --learning_rate $LR \
    --patience 5 --use_noise 1 --noise_level 0.15 \
    --multi_task 1 --multi_task_lambda 0.5 --multi_task_schedule warmup \
    --model_id "epilepsy_mt_05" 2>&1 | tee "$OUTPUT_DIR/epilepsy_multitask_05.log"

# =============================================================================
# 3. HAR CLASSIFICATION - Quick Sweep
# =============================================================================

echo ""
echo ">>> HAR Classification Experiments <<<"
echo ""

# Baseline
echo "[10/12] HAR Baseline..."
python run.py \
    --task_name finetune --downstream_task classification \
    --model $MODEL --data HAR \
    --root_path ./datasets/HAR \
    --input_len 128 --num_classes 6 \
    --train_epochs $EPOCHS --batch_size $BATCH --learning_rate $LR \
    --patience 5 --use_noise 0 --use_forgetting 0 \
    --model_id "har_baseline" 2>&1 | tee "$OUTPUT_DIR/har_baseline.log"

# With noise
echo "[11/12] HAR Noise 0.15..."
python run.py \
    --task_name finetune --downstream_task classification \
    --model $MODEL --data HAR \
    --root_path ./datasets/HAR \
    --input_len 128 --num_classes 6 \
    --train_epochs $EPOCHS --batch_size $BATCH --learning_rate $LR \
    --patience 5 --use_noise 1 --noise_level 0.15 \
    --model_id "har_noise" 2>&1 | tee "$OUTPUT_DIR/har_noise.log"

# Multi-task
echo "[12/12] HAR Multi-Task..."
python run.py \
    --task_name finetune --downstream_task classification \
    --model $MODEL --data HAR \
    --root_path ./datasets/HAR \
    --input_len 128 --num_classes 6 \
    --train_epochs $EPOCHS --batch_size $BATCH --learning_rate $LR \
    --patience 5 --use_noise 1 --noise_level 0.15 \
    --multi_task 1 --multi_task_lambda 0.3 --multi_task_schedule warmup \
    --model_id "har_multitask" 2>&1 | tee "$OUTPUT_DIR/har_multitask.log"

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "======================================"
echo "QUICK EXPERIMENTS COMPLETE"
echo "======================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Summary of results:"
echo "-------------------"

# Extract key metrics
for log in "$OUTPUT_DIR"/*.log; do
    name=$(basename "$log" .log)
    echo ""
    echo "[$name]"
    # Extract final metrics
    grep -E "(mse:|MSE:|mae:|MAE:|Acc:|F1:|Test Loss)" "$log" | tail -3 || echo "  (no metrics found)"
done

echo ""
echo "======================================"
