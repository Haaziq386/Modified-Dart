#!/bin/bash

# =============================================================================
# UCR Quick Sweep - Test Key Configurations on Selected Datasets
# =============================================================================
# Runs a focused set of experiments on popular UCR datasets.
# Good for validating the implementation before full sweep.
#
# Selected datasets:
#   - GunPoint (short, easy)
#   - Wafer (medium)
#   - ECG200 (medical)
#   - FordA (long, challenging)
#   - TwoLeadECG (medical)
#
# Total: 5 datasets × 6 configs = 30 experiments (~2-3 hours)
# =============================================================================

set -e

UCR_PATH="datasets/UCRArchive_2018/"
OUTPUT_DIR="./outputs/experiments/ucr_quick_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Popular/representative UCR datasets
DATASETS=("GunPoint" "Wafer" "ECG200" "FordA" "TwoLeadECG")

MODEL="HtulTS"
PT_EPOCHS=30
FT_EPOCHS=50
BATCH=16
LR=0.001
GPU=0

RESULTS_CSV="$OUTPUT_DIR/results.csv"
echo "dataset,input_len,num_classes,config,accuracy,f1" > "$RESULTS_CSV"

echo "=============================================="
echo "UCR Quick Sweep"
echo "Datasets: ${DATASETS[*]}"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

get_info() {
    local ds="$1"
    python3 -c "
import pandas as pd
import numpy as np
df = pd.read_csv('${UCR_PATH}/${ds}/${ds}_TRAIN.tsv', sep='\t', header=None)
print(df.shape[1] - 1, len(np.unique(df.iloc[:, 0])))
"
}

run_exp() {
    local ds="$1"
    local cfg="$2"
    local inlen="$3"
    local nclass="$4"
    shift 4
    local args="$@"
    
    echo ""
    echo ">>> $ds - $cfg"
    
    # Pretrain
    python -u run.py \
        --task_name pretrain \
        --downstream_task classification \
        --root_path "$UCR_PATH" \
        --model_id "${ds}_${cfg}" \
        --model $MODEL \
        --data UCR \
        --data_path "$ds" \
        --e_layers 2 \
        --d_layers 1 \
        --input_len $inlen \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --num_classes $nclass \
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
        --learning_rate $LR \
        --batch_size $BATCH \
        --train_epochs $PT_EPOCHS \
        --gpu $GPU \
        --use_norm 0 \
        $args \
        2>&1 | tee "$OUTPUT_DIR/${ds}_${cfg}_pt.log"
    
    # Finetune
    python -u run.py \
        --task_name finetune \
        --downstream_task classification \
        --root_path "$UCR_PATH" \
        --model_id "${ds}_${cfg}" \
        --model $MODEL \
        --data UCR \
        --data_path "$ds" \
        --e_layers 2 \
        --d_layers 1 \
        --input_len $inlen \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --num_classes $nclass \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 256 \
        --patch_len 8 \
        --stride 8 \
        --head_dropout 0.1 \
        --dropout 0.2 \
        --time_steps 1000 \
        --scheduler cosine \
        --batch_size $BATCH \
        --gpu $GPU \
        --lr_decay 1.0 \
        --lradj decay \
        --patience 50 \
        --learning_rate $LR \
        --pct_start 0.3 \
        --train_epochs $FT_EPOCHS \
        --use_norm 0 \
        $args \
        2>&1 | tee "$OUTPUT_DIR/${ds}_${cfg}_ft.log"
    
    # Extract results
    acc=$(grep -oP "Accuracy[:\s]+\K[0-9.]+" "$OUTPUT_DIR/${ds}_${cfg}_ft.log" 2>/dev/null | tail -1 || echo "N/A")
    f1=$(grep -oP "F1[:\s]+\K[0-9.]+" "$OUTPUT_DIR/${ds}_${cfg}_ft.log" 2>/dev/null | tail -1 || echo "N/A")
    echo "$ds,$inlen,$nclass,$cfg,$acc,$f1" >> "$RESULTS_CSV"
    echo "    => Acc: $acc, F1: $f1"
}

# =============================================================================
# RUN EXPERIMENTS
# =============================================================================

for ds in "${DATASETS[@]}"; do
    # Check if dataset exists
    if [ ! -d "${UCR_PATH}/${ds}" ]; then
        echo "Dataset $ds not found, skipping..."
        continue
    fi
    
    # Get dataset info
    read inlen nclass <<< $(get_info "$ds")
    echo ""
    echo "=============================================="
    echo "Dataset: $ds (len=$inlen, classes=$nclass)"
    echo "=============================================="
    
    # 1. Baseline
    run_exp "$ds" "baseline" "$inlen" "$nclass" \
        --use_noise 0 \
        --use_forgetting 0
    
    # 2. Noise only
    run_exp "$ds" "noise_015" "$inlen" "$nclass" \
        --use_noise 1 \
        --noise_level 0.15 \
        --use_forgetting 0
    
    # 3. Noise + Forgetting (activation)
    run_exp "$ds" "forget_act" "$inlen" "$nclass" \
        --use_noise 1 \
        --noise_level 0.15 \
        --use_forgetting 1 \
        --forgetting_type activation \
        --forgetting_rate 0.1
    
    # 4. Noise + Forgetting (weight)
    run_exp "$ds" "forget_wt" "$inlen" "$nclass" \
        --use_noise 1 \
        --noise_level 0.15 \
        --use_forgetting 1 \
        --forgetting_type weight \
        --forgetting_rate 0.1
    
    # 5. Multi-task (λ=0.3, warmup)
    run_exp "$ds" "mt_l03" "$inlen" "$nclass" \
        --use_noise 1 \
        --noise_level 0.15 \
        --use_forgetting 0 \
        --multi_task 1 \
        --multi_task_lambda 0.3 \
        --multi_task_schedule warmup
    
    # 6. Combined best
    run_exp "$ds" "combined" "$inlen" "$nclass" \
        --use_noise 1 \
        --noise_level 0.15 \
        --use_forgetting 1 \
        --forgetting_type activation \
        --forgetting_rate 0.1 \
        --multi_task 1 \
        --multi_task_lambda 0.3 \
        --multi_task_schedule warmup
done

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "=============================================="
echo "EXPERIMENTS COMPLETE"
echo "=============================================="
echo ""
echo "Results:"
column -t -s',' "$RESULTS_CSV"
echo ""
echo "Saved to: $RESULTS_CSV"

# Find best config per dataset
echo ""
echo "=== Best Config Per Dataset ==="
for ds in "${DATASETS[@]}"; do
    best=$(grep "^$ds," "$RESULTS_CSV" | sort -t',' -k5 -rn | head -1)
    if [ -n "$best" ]; then
        echo "$best" | awk -F',' '{print $1": "$4" (Acc: "$5")"}'
    fi
done
