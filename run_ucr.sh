#!/bin/bash

# Path to the UCR Archive
UCR_PATH="datasets/UCRArchive_2018/"

# Loop through every folder in the UCR Archive
for folder in "$UCR_PATH"/*/; do
    
    # Extract Dataset Name (e.g., "GunPoint" from "datasets/.../GunPoint/")
    dataset_name=$(basename "$folder")
    
    echo "========================================================"
    echo ">>>> Processing Dataset: $dataset_name"
    echo "========================================================"

    # ---------------------------------------------------------
    # AUTOMATICALLY DETECT SEQ_LEN AND NUM_CLASSES
    # ---------------------------------------------------------
    # We use a tiny python command to peek at the TRAIN file headers
    # Assumes TSV format: Column 0 is Label, Columns 1..N are Features
    
    train_file="${folder}/${dataset_name}_TRAIN.tsv"
    
    if [ ! -f "$train_file" ]; then
        echo "Skipping $dataset_name (TRAIN file not found)"
        continue
    fi

    # Python one-liner to get metadata
    read input_len num_classes <<< $(python3 -c "
import pandas as pd
import numpy as np
try:
    df = pd.read_csv('$train_file', sep='\t', header=None)
    # Length is columns - 1 (label col)
    seq_len = df.shape[1] - 1
    # Count unique labels in first column
    n_classes = len(np.unique(df.iloc[:, 0]))
    print(f'{seq_len} {n_classes}')
except Exception as e:
    print('ERROR ERROR')
")

    if [ "$input_len" == "ERROR" ]; then
        echo "Error reading metadata for $dataset_name. Skipping."
        continue
    fi

    echo "Detected -> Input Len: $input_len, Num Classes: $num_classes"

    # ---------------------------------------------------------
    # 1. PRETRAINING
    # ---------------------------------------------------------
    echo "Running Pretraining..."
    python -u run.py \
        --task_name pretrain \
        --downstream_task classification \
        --root_path datasets/UCRArchive_2018/ \
        --model_id "${dataset_name}" \
        --model HtulTS \
        --data UCR \
        --data_path "${dataset_name}" \
        --e_layers 2 \
        --d_layers 1 \
        --input_len $input_len \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --num_classes $num_classes \
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
        --learning_rate 0.001 \
        --batch_size 16 \
        --train_epochs 50 \
        --gpu 1 \
        --use_norm 0 \
        --use_noise 1 \
        --noise_level 0.15 \
        --use_forgetting 0 \

    # ---------------------------------------------------------
    # 2. FINETUNING
    # ---------------------------------------------------------
    echo "Running Finetuning..."
    python -u run.py \
        --task_name finetune \
        --downstream_task classification \
        --root_path datasets/UCRArchive_2018/ \
        --model_id "${dataset_name}" \
        --model HtulTS \
        --data UCR \
        --data_path "${dataset_name}" \
        --e_layers 2 \
        --d_layers 1 \
        --input_len $input_len \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --num_classes $num_classes \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 256 \
        --patch_len 8 \
        --stride 8 \
        --head_dropout 0.1 \
        --dropout 0.2 \
        --time_steps 1000 \
        --scheduler cosine \
        --batch_size 16 \
        --gpu 1 \
        --lr_decay 1.0 \
        --lradj decay \
        --patience 100 \
        --learning_rate 0.001 \
        --pct_start 0.3 \
        --train_epochs 100 \
        --use_norm 0 \
        --use_forgetting 0 \

done