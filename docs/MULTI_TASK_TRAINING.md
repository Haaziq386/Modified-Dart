# Multi-Task Joint Training for TimeDART

This document describes the multi-task joint training feature that enables simultaneous training on forecasting and classification objectives.

## Overview

Multi-task learning trains a shared encoder on both forecasting and classification objectives, addressing the **loss mismatch problem** where pre-training objectives (reconstruction) don't align well with downstream classification tasks.

### Key Benefits

1. **Reduced Loss Mismatch**: The encoder learns features useful for both tasks simultaneously
2. **Mitigated Catastrophic Forgetting**: Continuous training on both objectives prevents the model from "forgetting" forecasting capabilities when fine-tuning for classification
3. **Improved Representation Learning**: Shared representations benefit from diverse learning signals

## Quick Start

### Using HtulTS with Multi-Task Training

```bash
python run.py \
    --task_name finetune \
    --model HtulTS \
    --data ETTh1 \
    --root_path ./datasets/ETT-small \
    --data_path ETTh1.csv \
    --input_len 336 \
    --pred_len 96 \
    --train_epochs 20 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --multi_task 1 \
    --multi_task_lambda 0.3 \
    --multi_task_schedule warmup \
    --num_classes 6
```

### Using MultiTaskTimeDART

```bash
python run.py \
    --task_name finetune \
    --model MultiTaskTimeDART \
    --data ETTh1 \
    --root_path ./datasets/ETT-small \
    --data_path ETTh1.csv \
    --input_len 336 \
    --pred_len 96 \
    --d_model 512 \
    --e_layers 3 \
    --train_epochs 20 \
    --batch_size 32 \
    --multi_task 1 \
    --multi_task_lambda 0.25 \
    --multi_task_schedule warmup
```

## Configuration Parameters

### Core Multi-Task Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--multi_task` | 0 | Enable multi-task training (1=True, 0=False) |
| `--multi_task_lambda` | 0.5 | Weight for classification loss (0-1). Higher = more emphasis on classification |
| `--multi_task_schedule` | constant | Lambda schedule: `constant`, `linear`, `cosine`, `warmup` |
| `--use_uncertainty_weighting` | 0 | Enable automatic loss balancing (1=True) |
| `--gradient_clip` | 1.0 | Gradient clipping for stability |

### Lambda Schedules

1. **constant**: Fixed lambda throughout training
2. **linear**: Gradually increase classification weight from 0 to lambda
3. **cosine**: Smooth cosine ramp-up
4. **warmup**: Linear warmup for first 20% of training, then constant

## Preserving Forecasting Performance

To minimize impact on forecasting MSE while improving classification:

### Recommended Settings

```bash
--multi_task_lambda 0.2    # Low classification weight
--multi_task_schedule warmup   # Gradual introduction
--gradient_clip 1.0        # Stability
```

### Strategy: Conservative Multi-Task Learning

1. **Start with low lambda (0.2-0.3)**: Prioritizes forecasting
2. **Use warmup schedule**: Lets forecasting stabilize before adding classification gradient
3. **Monitor both metrics**: Ensure MSE stays within 5% of baseline

### Example: Minimal MSE Impact

```bash
python run.py \
    --task_name finetune \
    --model HtulTS \
    --data ETTh1 \
    --multi_task 1 \
    --multi_task_lambda 0.2 \
    --multi_task_schedule warmup \
    --train_epochs 30 \
    --patience 7
```

## Understanding the Loss Function

The total loss is computed as:

$$\mathcal{L}_{total} = (1 - \lambda) \cdot \mathcal{L}_{MSE} + \lambda \cdot \mathcal{L}_{CE}$$

Where:
- $\mathcal{L}_{MSE}$: Mean Squared Error for forecasting
- $\mathcal{L}_{CE}$: Cross-Entropy for classification
- $\lambda$: Multi-task lambda weight

### Uncertainty Weighting (Optional)

When `--use_uncertainty_weighting 1` is enabled, the loss becomes:

$$\mathcal{L}_{total} = \frac{1}{2\sigma_1^2} \mathcal{L}_{MSE} + \frac{1}{2\sigma_2^2} \mathcal{L}_{CE} + \log\sigma_1 + \log\sigma_2$$

Where $\sigma_1, \sigma_2$ are learned task-specific uncertainties.

## Architecture

### HtulTS Multi-Task Model

```
Input → Shared Linear Backbone (512d) → [Forecasting Head, Classification Head]
                                              ↓                    ↓
                                        [pred_len output]   [num_classes output]
```

### MultiTaskTimeDART

```
Input → Patch Embedding → Causal Transformer Encoder → [Forecast Head, Cls Head]
              ↓                     ↓
        Diffusion Noise     Denoising Decoder (pretrain)
```

## Training Tips

### 1. Dataset Preparation

Ensure you have both forecasting and classification datasets. The system will:
- Load forecasting data from `--data` / `--root_path`
- Attempt to load classification data from the same paths

### 2. Class Balance

If classification classes are imbalanced, consider:
- Using weighted cross-entropy (modify the experiment class)
- Increasing `multi_task_lambda` slightly

### 3. Learning Rate

Multi-task training often benefits from:
- Slightly lower learning rates (0.0001 or lower)
- AdamW optimizer with weight decay

### 4. Early Stopping

The implementation uses combined loss for early stopping. For stricter forecasting preservation:
- Modify `early_stopping` to use only forecast loss
- Set lower patience values

## Experimental Results Tracking

Results are saved to:
- `./outputs/test_results/{data}/score.txt`: Test metrics
- `./outputs/logs/`: TensorBoard logs

Monitor both:
- **Forecast MSE**: Should stay within 5% of baseline
- **Classification Accuracy**: Should approach or exceed separate classification models

## Troubleshooting

### High Forecasting MSE

1. Reduce `multi_task_lambda` (try 0.1-0.2)
2. Use `warmup` schedule
3. Increase gradient clipping

### Low Classification Accuracy

1. Increase `multi_task_lambda` (try 0.5-0.7)
2. Use `constant` or `cosine` schedule
3. Ensure classification data is properly loaded

### Training Instability

1. Enable gradient clipping (`--gradient_clip 1.0`)
2. Use uncertainty weighting (`--use_uncertainty_weighting 1`)
3. Reduce learning rate

## Files Added

- `models/MultiTaskTimeDART.py`: Transformer-based multi-task model
- `exp/exp_multitask.py`: Multi-task experiment class
- `models/HtulTS.py`: Extended with `MultiTaskModel` and `MultiTaskLightweightModel`
- `exp/exp_htults.py`: Extended with `multi_task_train()` method
- `scripts/multitask/`: Example training scripts

## References

- Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics.
- Caruana, R. (1997). Multitask learning. Machine learning, 28(1), 41-75.
