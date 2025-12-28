#!/bin/bash

# =============================================================================
# Noise Level Parameter Exploration
# =============================================================================
# Systematic exploration of noise levels for denoising pretraining:
# - Noise levels from 0.05 to 0.35
# - Comparison with no-noise baseline
# - Cross-dataset evaluation
# =============================================================================

set -e

OUTPUT_DIR="./outputs/experiments/noise_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

MODEL="HtulTS"
EPOCHS=20
BATCH=32
LR=0.0001

echo "=============================================="
echo "Noise Level Exploration"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# =============================================================================
# 1. ETTh1 Forecasting - Fine-Grained Noise Sweep
# =============================================================================

echo ""
echo ">>> ETTh1 Noise Level Sweep <<<"
echo ""

# No noise baseline
echo "ETTh1: No noise baseline..."
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
    --use_noise 0 \
    --model_id "etth1_no_noise" \
    2>&1 | tee "$OUTPUT_DIR/etth1_no_noise.log"

# Fine-grained noise levels
for NOISE in 0.05 0.075 0.10 0.125 0.15 0.175 0.20 0.225 0.25 0.275 0.30 0.35; do
    echo "ETTh1: noise=$NOISE..."
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
        --model_id "etth1_noise_$NOISE" \
        2>&1 | tee "$OUTPUT_DIR/etth1_noise_$NOISE.log"
done

# =============================================================================
# 2. Epilepsy Classification - Noise Sweep
# =============================================================================

echo ""
echo ">>> Epilepsy Noise Level Sweep <<<"
echo ""

# No noise baseline
echo "Epilepsy: No noise baseline..."
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
    --use_noise 0 \
    --model_id "epilepsy_no_noise" \
    2>&1 | tee "$OUTPUT_DIR/epilepsy_no_noise.log"

for NOISE in 0.05 0.10 0.15 0.20 0.25 0.30; do
    echo "Epilepsy: noise=$NOISE..."
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
        --noise_level $NOISE \
        --model_id "epilepsy_noise_$NOISE" \
        2>&1 | tee "$OUTPUT_DIR/epilepsy_noise_$NOISE.log"
done

# =============================================================================
# 3. HAR Classification - Noise Sweep
# =============================================================================

echo ""
echo ">>> HAR Noise Level Sweep <<<"
echo ""

# No noise baseline
echo "HAR: No noise baseline..."
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
    --use_noise 0 \
    --model_id "har_no_noise" \
    2>&1 | tee "$OUTPUT_DIR/har_no_noise.log"

for NOISE in 0.05 0.10 0.15 0.20 0.25 0.30; do
    echo "HAR: noise=$NOISE..."
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
        --noise_level $NOISE \
        --model_id "har_noise_$NOISE" \
        2>&1 | tee "$OUTPUT_DIR/har_noise_$NOISE.log"
done

# =============================================================================
# 4. ETTh2 Forecasting - Noise Cross-Validation
# =============================================================================

echo ""
echo ">>> ETTh2 Noise Level Sweep <<<"
echo ""

for NOISE in 0.0 0.10 0.15 0.20 0.25; do
    if [ "$NOISE" == "0.0" ]; then
        echo "ETTh2: No noise..."
        python run.py \
            --task_name finetune \
            --downstream_task forecast \
            --model $MODEL \
            --data ETTh2 \
            --root_path ./datasets/ETT-small \
            --data_path ETTh2.csv \
            --features M \
            --input_len 336 \
            --pred_len 96 \
            --train_epochs $EPOCHS \
            --batch_size $BATCH \
            --learning_rate $LR \
            --patience 7 \
            --use_noise 0 \
            --model_id "etth2_no_noise" \
            2>&1 | tee "$OUTPUT_DIR/etth2_no_noise.log"
    else
        echo "ETTh2: noise=$NOISE..."
        python run.py \
            --task_name finetune \
            --downstream_task forecast \
            --model $MODEL \
            --data ETTh2 \
            --root_path ./datasets/ETT-small \
            --data_path ETTh2.csv \
            --features M \
            --input_len 336 \
            --pred_len 96 \
            --train_epochs $EPOCHS \
            --batch_size $BATCH \
            --learning_rate $LR \
            --patience 7 \
            --use_noise 1 \
            --noise_level $NOISE \
            --model_id "etth2_noise_$NOISE" \
            2>&1 | tee "$OUTPUT_DIR/etth2_noise_$NOISE.log"
    fi
done

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

echo ""
echo "=============================================="
echo "NOISE EXPLORATION COMPLETE"
echo "=============================================="

SUMMARY_FILE="$OUTPUT_DIR/noise_results_summary.csv"
echo "experiment,noise_level,dataset,task,mse,mae,accuracy,f1" > "$SUMMARY_FILE"

for log in "$OUTPUT_DIR"/*.log; do
    name=$(basename "$log" .log)
    
    # Extract noise level from name
    if [[ "$name" == *"no_noise"* ]]; then
        noise=0.0
    else
        noise=$(echo "$name" | grep -oP "noise_\K[0-9.]+" || echo "N/A")
    fi
    
    # Extract dataset
    if [[ "$name" == etth1* ]]; then
        dataset="ETTh1"
        task="forecast"
    elif [[ "$name" == etth2* ]]; then
        dataset="ETTh2"
        task="forecast"
    elif [[ "$name" == epilepsy* ]]; then
        dataset="Epilepsy"
        task="classification"
    elif [[ "$name" == har* ]]; then
        dataset="HAR"
        task="classification"
    else
        dataset="unknown"
        task="unknown"
    fi
    
    mse=$(grep -oP "mse:\K[0-9.]+" "$log" | tail -1 || echo "N/A")
    mae=$(grep -oP "mae:\K[0-9.]+" "$log" | tail -1 || echo "N/A")
    acc=$(grep -oP "Acc: \K[0-9.]+" "$log" | tail -1 || echo "N/A")
    f1=$(grep -oP "F1: \K[0-9.]+" "$log" | tail -1 || echo "N/A")
    
    echo "$name,$noise,$dataset,$task,$mse,$mae,$acc,$f1" >> "$SUMMARY_FILE"
done

echo ""
echo "CSV summary saved to: $SUMMARY_FILE"
echo ""
echo "=== ETTh1 Results (sorted by MSE) ==="
grep ",ETTh1," "$SUMMARY_FILE" | sort -t',' -k5 -n | head -10

echo ""
echo "=== ETTh2 Results (sorted by MSE) ==="
grep ",ETTh2," "$SUMMARY_FILE" | sort -t',' -k5 -n | head -10

echo ""
echo "=== Epilepsy Results (sorted by Accuracy) ==="
grep ",Epilepsy," "$SUMMARY_FILE" | sort -t',' -k7 -rn | head -10

echo ""
echo "=== HAR Results (sorted by Accuracy) ==="
grep ",HAR," "$SUMMARY_FILE" | sort -t',' -k7 -rn | head -10

echo ""
echo "ANALYSIS: Look for the optimal noise level per dataset/task"
