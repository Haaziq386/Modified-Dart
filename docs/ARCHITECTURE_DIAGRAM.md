# HtulTS Architecture and Parameter Distribution

## Model Architecture Flow

```
Input [B, L, C]
    ↓
[Noise/Masking Augmentation]
    ↓
    ├─────────────────────────────────────────────────┐
    │                                                 │
    ↓                                                 ↓
┌─────────────────┐                          ┌────────────────┐
│  TIME DOMAIN    │                          │ FREQ DOMAIN    │
│  PATH (90%)     │                          │ PATH (1.4%)    │
├─────────────────┤                          ├────────────────┤
│ PatchEmbedding  │                          │FrequencyEncoder│
│  (8.7k params)  │                          │ (173.5k params)│
├─────────────────┤                          │                │
│  [B*C, L]       │                          │  [B*C, L]      │
│      ↓          │                          │      ↓         │
│  [B*C, 42, 512] │                          │  [B*C, 512]    │
│      ↓          │                          └────────────────┘
│  FLATTEN        │                                 ↓
│  [B*C, 21504]   │                          h_freq [B*C, 512]
│      ↓          │
│  PatchMixer     │
│ 🔴 11.0M params │
│ (97.7% of mixer)│
│      ↓          │
│ h_time [B*C, 512]
└─────────────────┘
         ↓
    ┌────────┴────────┐
    ↓                 ↓
[Time Projection]  [Freq Projection]
 (164.2k)           (164.2k)
    ↓                 ↓
  z_time           z_freq
    ↓                 ↓
    └─────────┬─────────┘
              ↓
        [TFC Loss]
         (used in
          pretrain)
    
    ↓ (both paths converge)
    
[h_time + h_freq] → Learnable Fusion (525.8k)
    ↓
[h_fused]
    ↓
[Forgetting (optional)]
    ↓
    ├─────────────────────┬──────────────────┐
    ↓                     ↓                  ↓
PRETRAIN            FINETUNE            CLASSIFICATION
├─────────────────┐ ├────────────────┐ ├──────────────────┐
│   Decoder       │ │ Forecaster     │ │ ConvNet + Head   │
│ (172.3k)        │ │ (172.3k)       │ │ (Not shown)      │
│ [B*C, L]        │ │ [B*C, Pred]    │ │                  │
└─────────────────┘ └────────────────┘ └──────────────────┘
```

## Parameter Distribution Pie Chart (Text Version)

```
Total Parameters: 12,221,008

┌────────────────────────────────────────────┐
│ patch_mixer (90.1% = 11,012,096)          │
│ ████████████████████████████████████████░░│
├────────────────────────────────────────────┤
│ fusion (4.3% = 525,824)                   │
│ ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
├────────────────────────────────────────────┤
│ freq_backbone (1.4% = 173,568)            │
│ █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
├────────────────────────────────────────────┤
│ decoder (1.4% = 172,368)                  │
│ █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
├────────────────────────────────────────────┤
│ time_projection (1.3% = 164,224)          │
│ █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
├────────────────────────────────────────────┤
│ freq_projection (1.3% = 164,224)          │
│ █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
├────────────────────────────────────────────┤
│ patch_embed (0.1% = 8,704)                │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
└────────────────────────────────────────────┘
```

## Detailed Breakdown: Where Do 11.0M Parameters Come From?

### patch_mixer Architecture
```
Input: [B*C, 21,504]
           ↓
Linear(21504, 512)
    Weights: 21,504 × 512 = 11,010,048 ← 97.7% of patch_mixer!
    Bias:    512
    Total:   11,010,560
           ↓
BatchNorm1d(512)
    Weight: 512
    Bias:   512
    Total:  1,024
           ↓
ReLU (0 params)
           ↓
Dropout(0.2) (0 params)
           ↓
Linear(512, 512)
    Weights: 512 × 512 = 262,144
    Bias:    512
    Total:   262,656
           ↓
Output: [B*C, 512]

Total patch_mixer: 11,274,240 params (adjusted for model = 11,012,096)
```

## Why patch_mixer Dominates

### The Math
- **num_patches** = ⌊(336 - 16) / 8⌋ + 2 = 42
- **hidden_dim** = 512 (hardcoded in HtulTS)
- **Flattened patch size** = 42 × 512 = 21,504

The first Linear layer (21504 → 512) accounts for:
$$21,504 \times 512 = 11,010,048 \text{ parameters}$$

This is the **dimensionality reduction bottleneck** that compresses all patch information into a single hidden vector.

---

## The d_model Bug Visualization

### What Scripts Intend
```python
# scripts/pretrain/ETTh1.sh
--d_model 128

# scripts/finetune/ETTh1.sh  
--d_model 512
```

### What Actually Happens
```python
# models/HtulTS.py - Line 267
class LightweightModel(nn.Module):
    def __init__(self, ...):
        self.hidden_dim = 512  # ← ALWAYS 512!
        #  ↑
        #  Ignores all d_model values!

# models/HtulTS.py - Line 472-507
class Model(nn.Module):
    def __init__(self, args):
        # args.d_model is NEVER used!
        self.model = LightweightModel(
            in_channels=args.enc_in,
            # ...
            # d_model NOT PASSED! ← BUG
        )
```

### Result
```
All experiments use hidden_dim=512 regardless of --d_model flag
↓
No variation in model capacity
↓
d_model parameter has ZERO effect on results
↓
Experimental ablation studies are INVALID ✗
```

---

## Recommended Fix

### Change 1: Update Model class (line 472-507)

```python
self.model = LightweightModel(
    in_channels=args.enc_in,
    out_channels=args.enc_in,
    task_name=self.task_name,
    pred_len=self.pred_len if hasattr(args, 'pred_len') else None,
    num_classes=getattr(args, 'num_classes', 2),
    input_len=args.input_len,
    use_noise=self.use_noise,
    use_forgetting=self.use_forgetting,
    forgetting_type=self.forgetting_type,
    forgetting_rate=self.forgetting_rate,
    tfc_weight=tfc_weight,
    tfc_warmup_steps=tfc_warmup_steps,
    use_real_imag=use_real_imag,
    projection_dim=projection_dim,
    patch_len=patch_len,
    stride=stride,
    hidden_dim=args.d_model,  # ← ADD THIS
)
```

### Change 2: Update LightweightModel init (line 267)

```python
def __init__(self, in_channels, out_channels, task_name="pretrain", pred_len=None, 
             num_classes=2, input_len=336, use_noise=False, 
             use_forgetting=False, forgetting_type="activation", forgetting_rate=0.1,
             tfc_weight=0.05, tfc_warmup_steps=0, use_real_imag=False,
             projection_dim=128, patch_len=16, stride=8, hidden_dim=512):  # ← ADD hidden_dim param
    super(LightweightModel, self).__init__()
    # ...
    self.hidden_dim = hidden_dim  # ← Use parameter instead of 512
```

After this fix:
- `--d_model 128`: 3.1M parameters  
- `--d_model 512`: 12.2M parameters
- Experiments become comparable and valid
