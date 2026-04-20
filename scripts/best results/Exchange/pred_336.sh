#!/bin/bash
set -euo pipefail

# Best HTULTS config from latest summary: outputs/test_results/htults_grid_Exchange_20260419_064546/summary.txt
# Dataset=Exchange, pred_len=336

DATASET="Exchange"
ROOT_PATH="./datasets/exchange/"
DATA_PATH="exchange.csv"
INPUT_LEN="336"
PRED_LEN="336"
GPU="${GPU:-0}"
TARGET_CONDA_ENV="${TARGET_CONDA_ENV:-timedart}"

if [ "${CONDA_DEFAULT_ENV:-}" != "$TARGET_CONDA_ENV" ]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$TARGET_CONDA_ENV"
  else
    echo "ERROR: conda not found. Activate env '$TARGET_CONDA_ENV' first."
    exit 1
  fi
fi

run_stamp=$(date +"%Y%m%d_%H%M%S")
run_tag="best_latest_${DATASET}_pl${PRED_LEN}_${run_stamp}"
model_id="HTULTS_${run_tag}"

log_dir="./outputs/logs/best_results/${DATASET}"
ckpt_dir="./outputs/pretrain_checkpoints/${DATASET}"
src_ckpt="${ckpt_dir}/ckpt_best.pth"
transfer_ckpt="ckpt_${run_tag}.pth"
dst_ckpt="${ckpt_dir}/${transfer_ckpt}"
pretrain_log="${log_dir}/pretrain_${run_tag}.log"
finetune_log="${log_dir}/finetune_${run_tag}.log"

mkdir -p "${log_dir}" "${ckpt_dir}"

echo "[1/2] Pretraining ${model_id} (pred_len=${PRED_LEN})"
python -u run.py \
  --task_name pretrain \
  --root_path "${ROOT_PATH}" \
  --data_path "${DATA_PATH}" \
  --model_id "${model_id}" \
  --model HtulTS \
  --data "${DATASET}" \
  --features M \
  --input_len "${INPUT_LEN}" \
  --seq_len "${INPUT_LEN}" \
  --enc_in 7 \
  --d_model "64" \
  --patch_len "16" \
  --stride "8" \
  --learning_rate "0.0001" \
  --batch_size "16" \
  --use_noise 0 \
  --train_epochs "50" \
  --lr_decay 0.95 \
  --gpu "${GPU}" \
  --use_warping "1" \
  --use_cpc "1" \
  --cpc_lambda "0.2" \
  --cpc_freq_mask_ratio "0.9" \
  --cpc_time_mask_ratio "1" \
  --cpc_use_learned_mask "1" \
  --cpc_loss_type "l2" \
  --use_forgetting "1" \
  --forgetting_type "weight" \
  --forgetting_rate "0.05" \
  --tfc_weight "0.01" \
  --tfc_warmup_steps "750" \
  --projection_dim "64" \
  --use_real_imag "0" \
  2>&1 | tee "${pretrain_log}"

if [ ! -f "${src_ckpt}" ]; then
  echo "ERROR: expected pretrain checkpoint not found at ${src_ckpt}"
  exit 1
fi

cp "${src_ckpt}" "${dst_ckpt}"

echo "[2/2] Finetuning ${model_id} (pred_len=${PRED_LEN})"
python -u run.py \
  --task_name finetune \
  --is_training 1 \
  --root_path "${ROOT_PATH}" \
  --data_path "${DATA_PATH}" \
  --model_id "${model_id}" \
  --model HtulTS \
  --data "${DATASET}" \
  --features M \
  --input_len "${INPUT_LEN}" \
  --seq_len "${INPUT_LEN}" \
  --label_len 48 \
  --pred_len "${PRED_LEN}" \
  --enc_in 7 \
  --d_model "64" \
  --patch_len "16" \
  --stride "8" \
  --learning_rate "0.0001" \
  --batch_size "16" \
  --train_epochs "10" \
  --patience 5 \
  --lr_decay 0.5 \
  --lradj decay \
  --use_noise 0 \
  --gpu "${GPU}" \
  --transfer_checkpoints "${transfer_ckpt}" \
  --use_warping "1" \
  --use_cpc "1" \
  --cpc_lambda "0.2" \
  --cpc_freq_mask_ratio "0.9" \
  --cpc_time_mask_ratio "1" \
  --cpc_use_learned_mask "1" \
  --cpc_loss_type "l2" \
  --use_forgetting "1" \
  --forgetting_type "weight" \
  --forgetting_rate "0.05" \
  --tfc_weight "0.01" \
  --tfc_warmup_steps "750" \
  --projection_dim "64" \
  --use_real_imag "0" \
  2>&1 | tee "${finetune_log}"

echo "Completed. Logs: ${pretrain_log}, ${finetune_log}"
