for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --root_path ./datasets/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1 \
        --model HtulTS \
        --data ETTh1 \
        --features M \
        --input_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --enc_in 7 \
        --learning_rate 0.0001 \
        --batch_size 16 \
        --train_epochs 50 \
        --patience 3 \
        --pct_start 0.3 \
        --lr_decay 0.5 \
        --lradj step \
        --patience 3 \
        --learning_rate 0.0001 \
        --pct_start 0.3 \
        --use_noise 0 \
        --use_forgetting 0 \
        --forgetting_type activation \
        --forgetting_rate 0.1 \
        --use_real_image 1 \
        --projection_dim 128
done