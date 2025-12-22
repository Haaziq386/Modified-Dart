# HtulTS - Lightweight Time Series Model with Forgetting Mechanisms

## Overview

HtulTS is a lightweight time series model designed for both pretraining and downstream tasks (forecasting and classification). It features a shared linear backbone architecture with task-specific heads and optional forgetting mechanisms for improved transfer learning capabilities.This is a experimentation and in progress repository aiming to make a good Time Series Forecaster.

## Key Features

- **Shared Linear Backbone**: Efficient MLP-based backbone for feature extraction across all tasks
- **Task-Specific Heads**: Separate heads optimized for:
  - **Pretraining**: Denoising autoencoder for representation learning
  - **Forecasting**: Regression head for time series prediction
  - **Classification**: Convolutional layers for time series classification
- **Forgetting Mechanisms**: Multiple strategies to reduce catastrophic forgetting during fine-tuning
- **Noise Injection**: Optional Gaussian noise during pretraining for robust feature learning

## Architecture Components

### 1. ForgettingMechanisms Module

Implements three types of forgetting strategies to prevent catastrophic forgetting:

```python
class ForgettingMechanisms(nn.Module):
    """Learnable forgetting gates and adaptive mechanisms"""
```

**Forgetting Types:**
- `activation`: Learnable forgetting gates applied to activation patterns
- `weight`: Weight-level importance scoring with decay
- `adaptive`: Adaptive forgetting based on gradient magnitude

### 2. LightweightModel

Core model architecture with shared backbone and task-specific heads:

```python
class LightweightModel(nn.Module):
    """
    Shared linear backbone + task-specific heads
    """
```

**Shared Backbone:**
- 2-layer MLP with ReLU activations
- Hidden dimension: 512
- Dropout: 0.2 and 0.1

**Task-Specific Heads:**

| Task | Head Type | Components |
|------|-----------|------------|
| Pretraining | Linear Decoder | Single linear layer mapping to input length |
| Forecasting | Regression Head | 2-layer MLP with dropout for prediction length |
| Classification | CNN Classifier | 3-layer Conv1d + adaptive pooling + classifier |

### 3. Model (Forecasting)

Wrapper for pretraining and forecasting tasks:

```python
class Model(nn.Module):
    """TimeDART forecasting model"""
```

**Methods:**
- `pretrain()`: Denoising autoencoder training
- `forecast()`: Time series forecasting
- Automatic normalization/denormalization

### 4. ClsModel (Classification)

Specialized model for classification tasks:

```python
class ClsModel(nn.Module):
    """TimeDART classification model"""
```

## Configuration Parameters

### Model Parameters

```python
args.input_len          # Input sequence length (default: 336)
args.pred_len           # Prediction length (for forecasting)
args.enc_in             # Input channels/features
args.task_name          # "pretrain" or "finetune"
args.downstream_task    # "forecast" or "classification"
args.use_norm           # Apply normalization (default: True)
```

### Noise Parameters

```python
args.use_noise          # Enable noise injection during pretraining
args.noise_level        # Noise scaling factor (default: 0.1)
```

### Forgetting Parameters

```python
args.use_forgetting     # Enable forgetting mechanisms
args.forgetting_type    # "activation", "weight", or "adaptive"
args.forgetting_rate    # Forgetting rate (default: 0.1)
```

## Training and Evaluation

```bash
sh scripts/pretrain/ETTh2.sh && sh scripts/finetune/ETTh2.sh
```


## Related Files

- [HtulTS.py](models/HtulTS.py) - Model architecture
- [exp_htults.py](exp/exp_htults.py) - Training pipeline
- [Main README](README.md) - Project overview

## Notes

- Checkpoints are saved to `./outputs/pretrain_checkpoints/` during pretraining
- Fine-tuning checkpoints go to `./outputs/checkpoints/`
- Test results and metrics saved to `./outputs/test_results/`
- TensorBoard logs available in `./outputs/logs/`
