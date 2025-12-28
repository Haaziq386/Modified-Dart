#!/bin/bash

# =============================================================================
# UCR Multi-Task Parameter Sweep
# =============================================================================
# Focused exploration of multi-task parameters on UCR datasets.
# Tests various λ values and schedules.
#
# Usage:
#   ./run_ucr_multitask.sh GunPoint ECG200    # Specific datasets
#   ./run_ucr_multitask.sh                     # All UCR datasets
# =============================================================================

set -e

UCR_PATH="datasets/UCRArchive_2018/"
OUTPUT_DIR="./outputs/experiments/ucr_multitask_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

MODEL="HtulTS"
PT_EPOCHS=50
FT_EPOCHS=100
BATCH=16
LR=0.001
GPU=0

RESULTS_CSV="$OUTPUT_DIR/multitask_results.csv"
echo "dataset,lambda,schedule,noise,forgetting,accuracy,f1" > "$RESULTS_CSV"

echo "=============================================="
echo "UCR Multi-Task Parameter Sweep"
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

run_mt_exp() {
    local ds="$1"
    local inlen="$2"
    local nclass="$3"
    local lam="$4"
    local sched="$5"
    local noise="$6"
    local forget="$7"
    
    local cfg="mt_l${lam}_${sched}"
    if [ "$forget" == "1" ]; then
        cfg="${cfg}_forget"
    fi
    
    echo ">>> $ds: λ=$lam, schedule=$sched, noise=$noise, forget=$forget"
    
    local extra_args="--use_noise $noise --noise_level 0.15"
    extra_args="$extra_args --multi_task 1 --multi_task_lambda $lam --multi_task_schedule $sched"
    
    if [ "$forget" == "1" ]; then
        extra_args="$extra_args --use_forgetting 1 --forgetting_type activation --forgetting_rate 0.1"
    else
        extra_args="$extra_args --use_forgetting 0"
    fi
    
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
        $extra_args \
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
        --patience 100 \
        --learning_rate $LR \
        --pct_start 0.3 \
        --train_epochs $FT_EPOCHS \
        --use_norm 0 \
        $extra_args \
        2>&1 | tee "$OUTPUT_DIR/${ds}_${cfg}_ft.log"
    
    # Extract results
    acc=$(grep -oP "Accuracy[:\s]+\K[0-9.]+" "$OUTPUT_DIR/${ds}_${cfg}_ft.log" 2>/dev/null | tail -1 || echo "N/A")
    f1=$(grep -oP "F1[:\s]+\K[0-9.]+" "$OUTPUT_DIR/${ds}_${cfg}_ft.log" 2>/dev/null | tail -1 || echo "N/A")
    echo "$ds,$lam,$sched,$noise,$forget,$acc,$f1" >> "$RESULTS_CSV"
    echo "    => Acc: $acc, F1: $f1"
}

# =============================================================================
# MAIN
# =============================================================================

# Get datasets from args or use all
DATASETS=()
if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    for folder in "$UCR_PATH"/*/; do
        DATASETS+=($(basename "$folder"))
    done
fi

echo "Datasets: ${#DATASETS[@]}"

for ds in "${DATASETS[@]}"; do
    if [ ! -d "${UCR_PATH}/${ds}" ]; then
        echo "Dataset $ds not found, skipping..."
        continue
    fi
    
    read inlen nclass <<< $(get_info "$ds")
    
    echo ""
    echo "=============================================="
    echo "Dataset: $ds (len=$inlen, classes=$nclass)"
    echo "=============================================="
    
    # -----------------------------------------
    # 1. Lambda exploration (with warmup schedule)
    # -----------------------------------------
    for lam in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
        run_mt_exp "$ds" "$inlen" "$nclass" "$lam" "warmup" "1" "0"
    done
    
    # -----------------------------------------
    # 2. Schedule exploration (with λ=0.3)
    # -----------------------------------------
    for sched in constant linear cosine warmup; do
        run_mt_exp "$ds" "$inlen" "$nclass" "0.3" "$sched" "1" "0"
    done
    
    # -----------------------------------------
    # 3. Multi-task + Forgetting (best λ values)
    # -----------------------------------------
    for lam in 0.3 0.5; do
        run_mt_exp "$ds" "$inlen" "$nclass" "$lam" "warmup" "1" "1"
    done
done

# =============================================================================
# ANALYSIS
# =============================================================================

echo ""
echo "=============================================="
echo "MULTI-TASK SWEEP COMPLETE"
echo "=============================================="

echo ""
echo "All results:"
column -t -s',' "$RESULTS_CSV"

echo ""
echo "=== Best λ per Dataset ==="
for ds in "${DATASETS[@]}"; do
    best=$(grep "^$ds," "$RESULTS_CSV" | sort -t',' -k6 -rn | head -1)
    if [ -n "$best" ]; then
        echo "$best" | awk -F',' '{print $1": λ="$2", sched="$3" (Acc: "$6")"}'
    fi
done

echo ""
echo "=== Average Accuracy by Lambda ==="
for lam in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    avg=$(grep ",$lam," "$RESULTS_CSV" | awk -F',' '{if($6!="N/A") {sum+=$6; count++}} END {if(count>0) print sum/count; else print "N/A"}')
    echo "λ=$lam: $avg"
done

echo ""
echo "Full results: $RESULTS_CSV"
