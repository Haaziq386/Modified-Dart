#!/bin/bash

# Script to compare HtulTS vs SimMTM vs TimeDART for various d_model values on Weather dataset
# Usage: bash scripts/compare_dmodel_weather.sh

set -o pipefail  # Exit if any command in a pipe fails, but allow individual commands to fail

DATASET="Weather"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./outputs/logs/dmodel_comparison_weather_${TIMESTAMP}"
RESULTS_DIR="./outputs/test_results/dmodel_comparison_weather_${TIMESTAMP}"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Test different input lengths
INPUT_LENS=(96 192 336 512)

# Test different d_model values
D_MODELS=(32 64 128)

# Prediction lengths
PRED_LENS=(96 192 336 720)

# Weather specific settings
ENC_IN=21
GPU=1

echo "=========================================="
echo "d_model Comparison: HtulTS vs SimMTM vs TimeDART"
echo "Dataset: $DATASET"
echo "Timestamp: $TIMESTAMP"
echo "Testing input lengths: ${INPUT_LENS[@]}"
echo "Testing d_model values: ${D_MODELS[@]}"
echo "Prediction lengths: ${PRED_LENS[@]}"
echo "GPU: $GPU"
echo "=========================================="
echo ""

SUMMARY_FILE="$RESULTS_DIR/comparison_summary.txt"
echo "d_model Comparison: HtulTS vs SimMTM vs TimeDART on Weather" > "$SUMMARY_FILE"
echo "Timestamp: $TIMESTAMP" >> "$SUMMARY_FILE"
echo "======================================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# ============== HtulTS Functions ==============

run_htults_pretrain() {
    local input_len=$1
    local d_model=$2
    local d_ff=$d_model
    local log_file="$LOG_DIR/htults_pretrain_len${input_len}_dm${d_model}.log"

    echo ">>> Running HtulTS PRETRAIN with input_len=$input_len, d_model=$d_model"
    echo "    Log: $log_file"

    python -u run.py \
        --task_name pretrain \
        --root_path ./datasets/weather/ \
        --data_path weather.csv \
        --model_id Weather_len${input_len}_dm${d_model} \
        --model HtulTS \
        --data Weather \
        --features M \
        --input_len $input_len \
        --e_layers 2 \
        --d_layers 1 \
        --enc_in $ENC_IN \
        --dec_in $ENC_IN \
        --c_out $ENC_IN \
        --n_heads 8 \
        --d_model $d_model \
        --d_ff $d_ff \
        --patch_len 2 \
        --stride 2 \
        --head_dropout 0.1 \
        --dropout 0.2 \
        --time_steps 1000 \
        --scheduler cosine \
        --lr_decay 0.95 \
        --learning_rate 0.001 \
        --batch_size 16 \
        --train_epochs 50 \
        --use_noise 0 \
        --use_forgetting 1 \
        --forgetting_type activation \
        --forgetting_rate 0.1 \
        --tfc_weight 0.05 \
        --tfc_warmup_steps 750 \
        --projection_dim 128 \
        --use_real_imag 1 \
        --use_warping 1 \
        --use_cpc 1 \
        --cpc_lambda 0.1 \
        --cpc_freq_mask_ratio 0.2 \
        --cpc_time_mask_ratio 0.2 \
        --cpc_use_learned_mask 1 \
        --cpc_loss_type l2 \
        --gpu $GPU \
        2>&1 | tee "$log_file"

    echo "    ✓ HtulTS pretrain completed for input_len=$input_len, d_model=$d_model"
    echo ""
}

run_htults_finetune() {
    local input_len=$1
    local pred_len=$2
    local d_model=$3
    local d_ff=$d_model
    local log_file="$LOG_DIR/htults_finetune_len${input_len}_pred${pred_len}_dm${d_model}.log"

    echo ">>> Running HtulTS FINETUNE with input_len=$input_len, pred_len=$pred_len, d_model=$d_model"
    echo "    Log: $log_file"

    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/weather/ \
        --data_path weather.csv \
        --model_id Weather_len${input_len}_dm${d_model} \
        --model HtulTS \
        --data Weather \
        --features M \
        --input_len $input_len \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in $ENC_IN \
        --dec_in $ENC_IN \
        --c_out $ENC_IN \
        --n_heads 8 \
        --d_model $d_model \
        --d_ff $d_ff \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 16 \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0004 \
        --pct_start 0.3 \
        --train_epochs 10 \
        --use_noise 0 \
        --use_forgetting 1 \
        --forgetting_type activation \
        --forgetting_rate 0.1 \
        --projection_dim 128 \
        --use_real_imag 1 \
        --use_warping 1 \
        --gpu $GPU \
        2>&1 | tee "$log_file"

    echo "    ✓ HtulTS finetune completed for input_len=$input_len, pred_len=$pred_len, d_model=$d_model"
    echo ""
}

# ============== SimMTM Functions ==============

run_simmtm_pretrain() {
    local input_len=$1
    local d_model=$2
    local d_ff=$d_model
    local log_file="$LOG_DIR/simmtm_pretrain_len${input_len}_dm${d_model}.log"

    echo ">>> Running SimMTM PRETRAIN with input_len=$input_len, d_model=$d_model"
    echo "    Log: $log_file"

    python -u run.py \
        --task_name pretrain \
        --root_path ./datasets/weather/ \
        --data_path weather.csv \
        --model_id Weather_len${input_len}_dm${d_model} \
        --model SimMTM \
        --data Weather \
        --features M \
        --input_len $input_len \
        --seq_len $input_len \
        --e_layers 2 \
        --d_layers 1 \
        --enc_in $ENC_IN \
        --dec_in $ENC_IN \
        --c_out $ENC_IN \
        --n_heads 8 \
        --d_model $d_model \
        --d_ff $d_ff \
        --patch_len 2 \
        --stride 2 \
        --head_dropout 0.1 \
        --dropout 0.2 \
        --time_steps 1000 \
        --scheduler cosine \
        --lr_decay 0.95 \
        --learning_rate 0.001 \
        --batch_size 16 \
        --train_epochs 50 \
        --use_noise 0 \
        --use_forgetting 1 \
        --forgetting_type activation \
        --forgetting_rate 0.1 \
        --tfc_weight 0.05 \
        --tfc_warmup_steps 750 \
        --projection_dim 128 \
        --use_real_imag 1 \
        --gpu $GPU \
        2>&1 | tee "$log_file"

    echo "    ✓ SimMTM pretrain completed for input_len=$input_len, d_model=$d_model"
    echo ""
}

run_simmtm_finetune() {
    local input_len=$1
    local pred_len=$2
    local d_model=$3
    local d_ff=$d_model
    local log_file="$LOG_DIR/simmtm_finetune_len${input_len}_pred${pred_len}_dm${d_model}.log"

    echo ">>> Running SimMTM FINETUNE with input_len=$input_len, pred_len=$pred_len, d_model=$d_model"
    echo "    Log: $log_file"

    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/weather/ \
        --data_path weather.csv \
        --model_id Weather_len${input_len}_dm${d_model} \
        --model SimMTM \
        --data Weather \
        --features M \
        --input_len $input_len \
        --seq_len $input_len \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in $ENC_IN \
        --dec_in $ENC_IN \
        --c_out $ENC_IN \
        --n_heads 8 \
        --d_model $d_model \
        --d_ff $d_ff \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 16 \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0004 \
        --pct_start 0.3 \
        --train_epochs 10 \
        --use_noise 0 \
        --use_forgetting 1 \
        --forgetting_type activation \
        --forgetting_rate 0.1 \
        --projection_dim 128 \
        --use_real_imag 1 \
        --gpu $GPU \
        2>&1 | tee "$log_file"

    echo "    ✓ SimMTM finetune completed for input_len=$input_len, pred_len=$pred_len, d_model=$d_model"
    echo ""
}

# ============== TimeDART Functions ==============

run_timedart_pretrain() {
    local input_len=$1
    local d_model=$2
    local d_ff=$d_model
    local log_file="$LOG_DIR/timedart_pretrain_len${input_len}_dm${d_model}.log"

    echo ">>> Running TimeDART PRETRAIN with input_len=$input_len, d_model=$d_model"
    echo "    Log: $log_file"

    python -u run.py \
        --task_name pretrain \
        --root_path ./datasets/weather/ \
        --data_path weather.csv \
        --model_id Weather_len${input_len}_dm${d_model} \
        --model TimeDART \
        --data Weather \
        --features M \
        --input_len $input_len \
        --e_layers 2 \
        --d_layers 1 \
        --enc_in $ENC_IN \
        --dec_in $ENC_IN \
        --c_out $ENC_IN \
        --n_heads 8 \
        --d_model $d_model \
        --d_ff $d_ff \
        --patch_len 2 \
        --stride 2 \
        --head_dropout 0.1 \
        --dropout 0.2 \
        --time_steps 1000 \
        --scheduler cosine \
        --lr_decay 0.95 \
        --learning_rate 0.001 \
        --batch_size 16 \
        --train_epochs 50 \
        --gpu $GPU \
        2>&1 | tee "$log_file"

    echo "    ✓ TimeDART pretrain completed for input_len=$input_len, d_model=$d_model"
    echo ""
}

run_timedart_finetune() {
    local input_len=$1
    local pred_len=$2
    local d_model=$3
    local d_ff=$d_model
    local log_file="$LOG_DIR/timedart_finetune_len${input_len}_pred${pred_len}_dm${d_model}.log"

    echo ">>> Running TimeDART FINETUNE with input_len=$input_len, pred_len=$pred_len, d_model=$d_model"
    echo "    Log: $log_file"

    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/weather/ \
        --data_path weather.csv \
        --model_id Weather_len${input_len}_dm${d_model} \
        --model TimeDART \
        --data Weather \
        --features M \
        --input_len $input_len \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in $ENC_IN \
        --dec_in $ENC_IN \
        --c_out $ENC_IN \
        --n_heads 8 \
        --d_model $d_model \
        --d_ff $d_ff \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 16 \
        --gpu $GPU \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0004 \
        --pct_start 0.3 \
        --train_epochs 10 \
        2>&1 | tee "$log_file"

    echo "    ✓ TimeDART finetune completed for input_len=$input_len, pred_len=$pred_len, d_model=$d_model"
    echo ""
}

extract_metrics() {
    local log_file=$1
    local model_name=$2
    local input_len=$3
    local pred_len=$4
    local d_model=$5

    if [ ! -f "$log_file" ]; then
        echo "$model_name | Input=$input_len | Pred=$pred_len | d_model=$d_model | MSE: FILE_NOT_FOUND | MAE: N/A" >> "$SUMMARY_FILE"
        return 0
    fi

    mse=$(grep -E "[0-9]+->[0-9]+, mse:" "$log_file" 2>/dev/null | tail -1 | grep -oP "mse:\K[0-9.]+" 2>/dev/null || echo "N/A")
    mae=$(grep -E "[0-9]+->[0-9]+, mse:" "$log_file" 2>/dev/null | tail -1 | grep -oP "mae:\K[0-9.]+" 2>/dev/null || echo "N/A")

    [ -z "$mse" ] && mse="N/A"
    [ -z "$mae" ] && mae="N/A"

    echo "$model_name | Input=$input_len | Pred=$pred_len | d_model=$d_model | MSE: $mse | MAE: $mae" >> "$SUMMARY_FILE"
    return 0
}

for input_len in "${INPUT_LENS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing INPUT_LEN = $input_len"
    echo "=========================================="
    echo ""

    for d_model in "${D_MODELS[@]}"; do
        echo ""
        echo "--- Testing d_model = $d_model ---"
        echo ""

        echo "Starting pretraining phase for input_len=$input_len, d_model=$d_model..."
        run_htults_pretrain $input_len $d_model || { echo "ERROR: HtulTS pretrain failed"; }
        # run_simmtm_pretrain $input_len $d_model || { echo "ERROR: SimMTM pretrain failed"; }
        # run_timedart_pretrain $input_len $d_model || { echo "ERROR: TimeDART pretrain failed"; }

        echo "" >> "$SUMMARY_FILE"
        echo "========================================" >> "$SUMMARY_FILE"
        echo "INPUT LENGTH = $input_len | d_model = $d_model" >> "$SUMMARY_FILE"
        echo "========================================" >> "$SUMMARY_FILE"

        for pred_len in "${PRED_LENS[@]}"; do
            echo ""
            echo "--- Testing pred_len = $pred_len ---"
            echo ""

            echo "  Running HtulTS finetune..."
            run_htults_finetune $input_len $pred_len $d_model || echo "  WARNING: HtulTS finetune encountered an issue"

            # echo "  Running SimMTM finetune..."
            # run_simmtm_finetune $input_len $pred_len $d_model || echo "  WARNING: SimMTM finetune encountered an issue"

            # echo "  Running TimeDART finetune..."
            # run_timedart_finetune $input_len $pred_len $d_model || echo "  WARNING: TimeDART finetune encountered an issue"

            echo "" >> "$SUMMARY_FILE"
            echo "Prediction Length = $pred_len:" >> "$SUMMARY_FILE"
            extract_metrics "$LOG_DIR/htults_finetune_len${input_len}_pred${pred_len}_dm${d_model}.log" "HtulTS   " $input_len $pred_len $d_model
            # extract_metrics "$LOG_DIR/simmtm_finetune_len${input_len}_pred${pred_len}_dm${d_model}.log" "SimMTM   " $input_len $pred_len $d_model
            # extract_metrics "$LOG_DIR/timedart_finetune_len${input_len}_pred${pred_len}_dm${d_model}.log" "TimeDART " $input_len $pred_len $d_model

            echo "  Completed pred_len=$pred_len"
        done

        echo ""
        echo "Completed input_len=$input_len, d_model=$d_model"
        echo ""
    done
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

cat "$SUMMARY_FILE"
 