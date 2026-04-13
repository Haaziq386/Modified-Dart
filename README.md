# HtulTS (Current Model)

HtulTS is a dual-branch time-series model used in this repo for:
- self-supervised pretraining
- forecasting fine-tuning
- classification fine-tuning

This README reflects the current implementation in `models/HtulTS.py`.

## Architecture Summary

### 1) Time branch (patch-based)
- Input is reshaped from `[B, L, C]` to `[B*C, L]`.
- `PatchEmbedding` extracts overlapping 1D patches with configurable `patch_len` and `stride`.
- Patch tokens are projected and flattened.
- `patch_mixer` aggregates all patches into one global time embedding `h_time`.

### 2) Frequency branch (FFT-based)
- `FrequencyEncoder` applies Hann-windowed `rFFT`.
- Two feature modes:
    - `use_real_imag=1`: real + imaginary channels
    - `use_real_imag=0`: log-amplitude + sin(phase) + cos(phase)
- Optional `AdaptiveFrequencyWarping` (`use_warping=1`) provides learnable monotonic re-indexing of frequency bins.
- Frequency features are normalized per sample, flattened, then projected to `h_freq`.

### 3) Fusion and optional forgetting
- `LearnableFusion` combines `h_time` and `h_freq`.
- Optional `ForgettingMechanisms` (mostly useful during finetune):
    - `activation`
    - `weight`
    - `adaptive`

### 4) Task heads
- `task_name=pretrain`: decoder reconstructs full input sequence.
- `task_name=finetune` + `downstream_task=forecast`: residual forecasting head predicts `pred_len`.
- `task_name=finetune` + `downstream_task=classification`: CNN classifier over fused features.

## Pretraining Objectives

Pretraining returns `(reconstruction, aux_loss)` where:
- reconstruction loss is MSE between input and reconstruction (computed in trainer)
- auxiliary loss is:
    - TF-C loss (`TFC_Loss`) always enabled in pretrain path
    - optional CPC-TF loss (`CPCTFLoss`) when `use_cpc=1`

Total pretraining loss used in `exp/exp_htults.py`:

`total_loss = recon_loss + aux_loss`

When CPC-TF is enabled, `aux_loss` internally is:

`loss_tfc + cpc_lambda * (loss_time_to_freq + loss_freq_to_time)`

TF-C supports warmup through `tfc_warmup_steps`.

## Main Runtime Arguments

### Core
- `--task_name`: `pretrain` or `finetune`
- `--downstream_task`: `forecast` or `classification`
- `--model HtulTS`
- `--input_len`, `--pred_len`, `--enc_in`, `--d_model`

### Time branch
- `--patch_len` (default 16)
- `--stride` (default 8)

### Frequency branch / TF-C
- `--use_real_imag` (0/1)
- `--use_warping` (0/1)
- `--projection_dim` (default 128)
- `--tfc_weight` (default 0.05)
- `--tfc_warmup_steps` (default 0)

### CPC-TF (optional)
- `--use_cpc` (0/1)
- `--cpc_lambda`
- `--cpc_freq_mask_ratio`
- `--cpc_time_mask_ratio`
- `--cpc_use_learned_mask` (0/1)
- `--cpc_loss_type` (`l2` or `smooth_l1`)
- `--cpc_pos_emb_dim`
- `--cpc_hidden_dim`

### Regularization / transfer
- `--use_noise` (pretrain)
- `--noise_level`
- `--use_forgetting` (finetune)
- `--forgetting_type`
- `--forgetting_rate`
- `--use_norm`

## Example Commands

### Pretrain (forecast setting)
```bash
python run.py \
    --task_name pretrain \
    --downstream_task forecast \
    --is_training 1 \
    --model_id HtulTS_ETTh1 \
    --model HtulTS \
    --data ETTh1 \
    --features M \
    --input_len 336 \
    --pred_len 96 \
    --enc_in 7 \
    --d_model 512 \
    --patch_len 16 \
    --stride 8 \
    --tfc_weight 0.05 \
    --tfc_warmup_steps 500 \
    --use_real_imag 1 \
    --use_warping 0 \
    --use_cpc 1 \
    --cpc_lambda 0.1
```

### Finetune (forecast)
```bash
python run.py \
    --task_name finetune \
    --downstream_task forecast \
    --is_training 1 \
    --model_id HtulTS_ETTh1 \
    --model HtulTS \
    --data ETTh1 \
    --features M \
    --input_len 336 \
    --pred_len 96 \
    --enc_in 7 \
    --d_model 512 \
    --patch_len 16 \
    --stride 8 \
    --use_forgetting 1 \
    --forgetting_type activation \
    --forgetting_rate 0.1
```

### Finetune (classification)
```bash
python run.py \
    --task_name finetune \
    --downstream_task classification \
    --is_training 1 \
    --model_id HtulTS_UCR \
    --model HtulTS \
    --data ECG \
    --input_len 512 \
    --enc_in 1 \
    --num_classes 2
```

## Checkpoints and Logging

- Pretrain checkpoints: `./outputs/pretrain_checkpoints/<dataset>/`
- Finetune checkpoints: `./outputs/checkpoints/<setting>/`
- Test outputs: `./outputs/test_results/`
- TensorBoard logs: `./outputs/logs/`

Checkpoint loading is shape-safe in `exp/exp_htults.py`:
- mismatched keys are skipped
- loading uses `strict=False`

This is expected when transferring from pretrain to finetune because pretrain-only modules (for example decoder/CPC modules) are not needed in downstream heads.

## Related Files

- `models/HtulTS.py`: full model implementation
- `exp/exp_htults.py`: training/validation/testing loops
- `run.py`: CLI arguments and experiment entrypoint
