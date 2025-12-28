#!/bin/bash

# =============================================================================
# Forgetting Mechanism Parameter Exploration
# =============================================================================
# Systematic exploration of forgetting mechanism parameters:
# - Forgetting types (activation, weight, adaptive)
# - Forgetting rates
# - Combinations with noise levels
# =============================================================================

set -e

OUTPUT_DIR="./outputs/experiments/forgetting_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

MODEL="HtulTS"
EPOCHS=20
BATCH=32
LR=0.0001

echo "=============================================="
echo "Forgetting Mechanism Exploration"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# =============================================================================
# 1. ETTh1 Forecasting - Forgetting Type Comparison
# =============================================================================

echo ""
echo ">>> ETTh1 Forgetting Type Exploration <<<"
echo ""

# Baseline (no forgetting)
echo "ETTh1 Baseline (no forgetting)..."
python run.py \
    --task_name finetune \
    --downstream_task forecast \
    --model $MODEL \
    --data ETTh1 \
    --root_path ./datasets/ETT-small \
    --data_path ETTh1.csv \
    --features M \
    --input_len 336 \
    --pred_len 96 \
    --train_epochs $EPOCHS \
    --batch_size $BATCH \
    --learning_rate $LR \
    --patience 7 \
    --use_noise 1 \
    --noise_level 0.15 \
    --use_forgetting 0 \
    --model_id "etth1_no_forget" \
    2>&1 | tee "$OUTPUT_DIR/etth1_no_forget.log"

# Test each forgetting type
for FORGET_TYPE in activation weight adaptive; do
    for FORGET_RATE in 0.05 0.1 0.15 0.2 0.25 0.3; do
        echo "ETTh1: type=$FORGET_TYPE, rate=$FORGET_RATE..."
        python run.py \
            --task_name finetune \
            --downstream_task forecast \
            --model $MODEL \
            --data ETTh1 \
            --root_path ./datasets/ETT-small \
            --data_path ETTh1.csv \
            --features M \
            --input_len 336 \
            --pred_len 96 \
            --train_epochs $EPOCHS \
            --batch_size $BATCH \
            --learning_rate $LR \
            --patience 7 \
            --use_noise 1 \
            --noise_level 0.15 \
            --use_forgetting 1 \
            --forgetting_type $FORGET_TYPE \
            --forgetting_rate $FORGET_RATE \
            --model_id "etth1_${FORGET_TYPE}_${FORGET_RATE}" \
            2>&1 | tee "$OUTPUT_DIR/etth1_${FORGET_TYPE}_${FORGET_RATE}.log"
    done
done

# =============================================================================
# 2. Epilepsy Classification - Forgetting Exploration
# =============================================================================

echo ""
echo ">>> Epilepsy Forgetting Exploration <<<"
echo ""

# Baseline
echo "Epilepsy Baseline (no forgetting)..."
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
    --use_forgetting 0 \
    --model_id "epilepsy_no_forget" \
    2>&1 | tee "$OUTPUT_DIR/epilepsy_no_forget.log"

# Test forgetting types
for FORGET_TYPE in activation weight adaptive; do
    for FORGET_RATE in 0.1 0.2 0.3; do
        echo "Epilepsy: type=$FORGET_TYPE, rate=$FORGET_RATE..."
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
            --model_id "epilepsy_${FORGET_TYPE}_${FORGET_RATE}" \
            2>&1 | tee "$OUTPUT_DIR/epilepsy_${FORGET_TYPE}_${FORGET_RATE}.log"
    done
done

# =============================================================================
# 3. Forgetting + Noise Level Interaction
# =============================================================================

echo ""
echo ">>> Forgetting + Noise Interaction (ETTh1) <<<"
echo ""

for NOISE in 0.1 0.15 0.2; do
    for FORGET_RATE in 0.1 0.2; do
        echo "ETTh1: noise=$NOISE, forget_rate=$FORGET_RATE..."
        python run.py \
            --task_name finetune \
            --downstream_task forecast \
            --model $MODEL \
            --data ETTh1 \
            --root_path ./datasets/ETT-small \
            --data_path ETTh1.csv \
            --features M \
            --input_len 336 \
            --pred_len 96 \
            --train_epochs $EPOCHS \
            --batch_size $BATCH \
            --learning_rate $LR \
            --patience 7 \
            --use_noise 1 \
            --noise_level $NOISE \
            --use_forgetting 1 \
            --forgetting_type activation \
            --forgetting_rate $FORGET_RATE \
            --model_id "etth1_noise${NOISE}_forget${FORGET_RATE}" \
            2>&1 | tee "$OUTPUT_DIR/etth1_noise${NOISE}_forget${FORGET_RATE}.log"
    done
done

# =============================================================================
# RESULTS SUMMARY
# =============================================================================

echo ""
echo "=============================================="
echo "FORGETTING EXPERIMENTS COMPLETE"
echo "=============================================="

SUMMARY_FILE="$OUTPUT_DIR/results_summary.csv"
echo "experiment,mse,mae,accuracy,f1" > "$SUMMARY_FILE"

for log in "$OUTPUT_DIR"/*.log; do
    name=$(basename "$log" .log)
    mse=$(grep -oP "mse:\K[0-9.]+" "$log" | tail -1 || echo "N/A")
    mae=$(grep -oP "mae:\K[0-9.]+" "$log" | tail -1 || echo "N/A")
    acc=$(grep -oP "Acc: \K[0-9.]+" "$log" | tail -1 || echo "N/A")
    f1=$(grep -oP "F1: \K[0-9.]+" "$log" | tail -1 || echo "N/A")
    echo "$name,$mse,$mae,$acc,$f1" >> "$SUMMARY_FILE"
done

echo ""
echo "CSV summary saved to: $SUMMARY_FILE"
echo ""
echo "Top ETTh1 results (by MSE):"
grep "etth1" "$SUMMARY_FILE" | sort -t',' -k2 -n | head -5

echo ""
echo "Top Epilepsy results (by Accuracy):"
grep "epilepsy" "$SUMMARY_FILE" | sort -t',' -k4 -rn | head -5
