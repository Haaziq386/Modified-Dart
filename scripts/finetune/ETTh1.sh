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
        --d_model 512 \
        --patch_len 16 \
        --stride 8 \
        --learning_rate 0.0001 \
        --batch_size 16 \
        --patience 5 \
        --lradj step \
        --lr_decay 0.5 \
        --use_noise 0 \
        --gpu 2 \
        --use_forgetting 1 \
        --forgetting_type adaptive \
        --forgetting_rate 0.05 \
        --use_real_imag 1 \
        --projection_dim 128
done