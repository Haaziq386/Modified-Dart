# Parameter Usage Analysis for HtulTS Model

This document analyzes ALL parameters used in the ETTh1 scripts and shows exactly where each is used and whether it can be safely removed.

## Summary

### ✅ ESSENTIAL PARAMETERS (Cannot Remove)
These are actively used by HtulTS model or training pipeline.

### ⚠️ UNUSED PARAMETERS (Can Remove)
These have NO effect on HtulTS and can be safely removed from scripts.

### 🔧 USED ONLY IN SETTING NAME
These are only used for naming experiment folders, not for model behavior.

---

## Detailed Parameter Analysis

### 1. TASK & DATA PARAMETERS

#### ✅ `--task_name` (pretrain/finetune)
- **Used in**: exp_htults.py, HtulTS.py, run.py
- **Purpose**: Determines whether to run pretrain or finetune mode
- **HtulTS usage**: `self.task_name = args.task_name` - determines model behavior
- **Cannot remove**: Essential for mode selection

#### ✅ `--root_path` (./datasets/ETT-small/)
- **Used in**: data_factory.py
- **Purpose**: Path to dataset directory
- **Cannot remove**: Required for data loading

#### ✅ `--data_path` (ETTh1.csv)
- **Used in**: data_factory.py
- **Purpose**: Specific dataset file name
- **Cannot remove**: Required for data loading

#### ✅ `--model_id` (ETTh1)
- **Used in**: run.py (setting name only)
- **Purpose**: Identifier for experiment/checkpoint folders
- **Can remove**: Only used in folder naming, not model behavior

#### ✅ `--model` (HtulTS)
- **Used in**: exp_basic.py, run.py
- **Purpose**: Selects which model class to instantiate
- **Cannot remove**: Required for model selection

#### ✅ `--data` (ETTh1)
- **Used in**: data_factory.py, exp_htults.py (for paths), run.py (setting name)
- **Purpose**: Dataset type identifier
- **Cannot remove**: Required for data loading and checkpoint paths

#### ✅ `--features` (M/MS/S)
- **Used in**: data_factory.py, exp_htults.py
- **Purpose**: 
  - M = Multivariate to Multivariate
  - S = Univariate to Univariate
  - MS = Multivariate to Univariate
- **exp_htults usage**: `f_dim = -1 if self.args.features == "MS" else 0`
- **Cannot remove**: Affects data slicing and prediction

---

### 2. MODEL ARCHITECTURE PARAMETERS

#### ⚠️ `--e_layers` (2) - **NOT USED IN HtulTS**
- **Used in**: TimeDART.py, TimeDART_v2.py ONLY
- **HtulTS usage**: **NONE** - Not referenced in HtulTS.py at all
- **Can remove**: Safe to remove for HtulTS scripts
- **Impact if removed**: None for HtulTS model

#### ⚠️ `--d_layers` (1) - **NOT USED IN HtulTS**
- **Used in**: TimeDART.py, TimeDART_v2.py ONLY
- **HtulTS usage**: **NONE** - Not referenced in HtulTS.py at all
- **Can remove**: Safe to remove for HtulTS scripts
- **Impact if removed**: None for HtulTS model

#### ✅ `--enc_in` (7)
- **Used in**: HtulTS.py
- **Purpose**: Number of input channels (features)
- **HtulTS usage**: `in_channels=args.enc_in, out_channels=args.enc_in`
- **Cannot remove**: Defines model input/output dimensions

#### ⚠️ `--dec_in` (7) - **NOT USED IN HtulTS**
- **Used in**: Other models (TimeDART, etc.)
- **HtulTS usage**: **NONE** - HtulTS doesn't have a decoder
- **Can remove**: Safe to remove for HtulTS scripts
- **Impact if removed**: None for HtulTS model

#### ⚠️ `--c_out` (7) - **NOT USED IN HtulTS**
- **Used in**: Other models
- **HtulTS usage**: **NONE** - Uses enc_in for output size
- **Can remove**: Safe to remove for HtulTS scripts
- **Impact if removed**: None for HtulTS model

#### ⚠️ `--n_heads` (16) - **NOT USED IN HtulTS**
- **Used in**: Transformer-based models (TimeDART, etc.)
- **HtulTS usage**: **NONE** - HtulTS doesn't use attention
- **Can remove**: Safe to remove for HtulTS scripts
- **Impact if removed**: None for HtulTS model

#### ✅ `--d_model` (32) - **USED IN HtulTS**
- **Used in**: Transformer-based models and HtulTS
- **HtulTS usage**: Passed directly as `hidden_dim=args.d_model` to `LightweightModel`, controlling the patch embedding projection dimension and all internal hidden representations
- **Can remove**: **Do NOT remove** - `args.d_model` is read without a fallback default in `HtulTS.py`, so omitting this from the script causes an `AttributeError` at runtime
- **Impact if removed**: HtulTS will raise `AttributeError: Namespace object has no attribute 'd_model'` and fail to initialize

#### ⚠️ `--d_ff` (64) - **NOT USED IN HtulTS**
- **Used in**: Transformer-based models (feed-forward dimension)
- **HtulTS usage**: **NONE**
- **Can remove**: Safe to remove for HtulTS scripts
- **Impact if removed**: None for HtulTS model

#### ✅ `--patch_len` (2) - **USED IN HtulTS**
- **Used in**: PatchTST, HtulTS, and similar models
- **HtulTS usage**: Read via `getattr(args, 'patch_len', 16)` and passed to `PatchEmbedding`, which defines the size of each patch window extracted from the input sequence
- **Can remove**: **Do NOT remove** - HtulTS will fall back to default `patch_len=16`; removing it prevents explicit control over patch granularity
- **Impact if removed**: HtulTS uses a hard-coded default of 16, which may not be optimal for your dataset

#### ✅ `--stride` (2) - **USED IN HtulTS**
- **Used in**: PatchTST, HtulTS, and similar models
- **HtulTS usage**: Read via `getattr(args, 'stride', 8)` and passed to `PatchEmbedding` to control the step size between consecutive patches
- **Can remove**: **Do NOT remove** - HtulTS will fall back to default `stride=8`; removing it prevents explicit control over patch overlap
- **Impact if removed**: HtulTS uses a hard-coded default of 8, which may not be optimal for your dataset

---

### 3. SEQUENCE LENGTH PARAMETERS

#### ✅ `--input_len` (336)
- **Used in**: HtulTS.py, data_factory.py, exp_htults.py
- **Purpose**: Length of input time series
- **HtulTS usage**: `self.input_len = args.input_len`, passed to LightweightModel
- **Cannot remove**: Essential for model architecture and data loading

#### ✅ `--label_len` (48) - **FINETUNE ONLY**
- **Used in**: data_factory.py
- **Purpose**: Length of start token (for encoder-decoder models)
- **HtulTS usage**: Not directly used in HtulTS (no decoder)
- **Can remove from pretrain**: Not used in pretrain
- **Keep in finetune**: Used in data_loader size parameter

#### ✅ `--pred_len` (96/192/336/720) - **FINETUNE ONLY**
- **Used in**: HtulTS.py, data_factory.py, exp_htults.py
- **Purpose**: Length of prediction sequence
- **HtulTS usage**: `self.pred_len = args.pred_len`, used in forecast head
- **Cannot remove from finetune**: Required for forecasting
- **Not in pretrain**: Pretrain doesn't need pred_len

---

### 4. REGULARIZATION PARAMETERS

#### ✅ `--dropout` (0.2)
- **Used in**: HtulTS.py (indirectly via LightweightModel defaults)
- **Purpose**: Dropout rate for regularization
- **Note**: Currently NOT passed to LightweightModel (uses default 0.1)
- **Can remove**: Current scripts don't affect model (model uses hardcoded 0.1)

#### ✅ `--head_dropout` (0.1)
- **Used in**: HtulTS.py (indirectly via LightweightModel defaults)
- **Purpose**: Dropout for prediction head
- **Note**: Currently NOT passed to LightweightModel
- **Can remove**: Current scripts don't affect model (model uses defaults)

---

### 5. TRAINING PARAMETERS

#### ✅ `--train_epochs` (50)
- **Used in**: exp_htults.py
- **Purpose**: Number of training epochs
- **Usage**: `for epoch in range(self.args.train_epochs)`
- **Cannot remove**: Controls training duration

#### ✅ `--batch_size` (16)
- **Used in**: data_factory.py
- **Purpose**: Batch size for data loaders
- **Cannot remove**: Required for data loading

#### ✅ `--learning_rate` (0.0001)
- **Used in**: exp_htults.py
- **Purpose**: Learning rate for optimizer
- **Usage**: `lr=self.args.learning_rate`
- **Cannot remove**: Essential for optimization

#### ✅ `--lr_decay` (0.9) - **PRETRAIN ONLY**
- **Used in**: exp_htults.py (pretrain only)
- **Purpose**: Gamma for ExponentialLR scheduler
- **Usage**: `ExponentialLR(optimizer=model_optim, gamma=self.args.lr_decay)`
- **Cannot remove from pretrain**: Used in pretrain scheduler
- **Not in finetune**: Finetune uses OneCycleLR

#### ✅ `--lradj` (step/decay) - **FINETUNE ONLY**
- **Used in**: exp_htults.py (finetune only)
- **Purpose**: Learning rate adjustment strategy
- **Usage**: `if self.args.lradj == "step":`
- **Cannot remove from finetune**: Controls LR adjustment
- **Not in pretrain**: Pretrain uses ExponentialLR

#### ✅ `--patience` (3) - **FINETUNE ONLY**
- **Used in**: exp_htults.py (finetune only)
- **Purpose**: Early stopping patience
- **Usage**: `EarlyStopping(patience=self.args.patience)`
- **Cannot remove from finetune**: Required for early stopping
- **Not in pretrain**: Pretrain doesn't use early stopping

#### ✅ `--pct_start` (0.3) - **FINETUNE ONLY**
- **Used in**: exp_htults.py (finetune only)
- **Purpose**: Percentage of cycle for increasing LR in OneCycleLR
- **Usage**: `pct_start=self.args.pct_start`
- **Cannot remove from finetune**: Required for OneCycleLR
- **Not in pretrain**: Pretrain uses ExponentialLR

---

### 6. PRETRAIN-SPECIFIC PARAMETERS

#### ⚠️ `--time_steps` (1000) - **PRETRAIN ONLY, NOT USED IN HtulTS**
- **Used in**: Other models (diffusion-based models)
- **HtulTS usage**: **NONE** - HtulTS doesn't use diffusion
- **Can remove from pretrain**: Not used by HtulTS
- **Remove from finetune**: Definitely not needed in finetune

#### ⚠️ `--scheduler` (cosine) - **PRETRAIN ONLY, NOT USED IN HtulTS**
- **Used in**: Other models (diffusion scheduler)
- **HtulTS usage**: **NONE** - HtulTS doesn't use diffusion
- **Can remove from pretrain**: Not used by HtulTS
- **Remove from finetune**: Definitely not needed in finetune

#### ⚠️ `--use_noise` (0) - **DISABLED**
- **Used in**: HtulTS.py
- **Purpose**: Enable noise injection during pretrain
- **HtulTS usage**: `self.use_noise = getattr(args, 'use_noise', False)`
- **Current value**: 0 (disabled)
- **Can remove**: Set to 0, so effectively not used
- **Impact if removed**: None (already disabled)

#### ✅ `--tfc_weight` (0.05) - **PRETRAIN ONLY**
- **Used in**: HtulTS.py
- **Purpose**: Weight for Time-Frequency Consistency loss
- **HtulTS usage**: Passed to LightweightModel, used in pretrain
- **Cannot remove from pretrain**: Active feature for contrastive learning
- **Not in finetune**: Not used in finetune

#### ✅ `--tfc_warmup_steps` (500) - **PRETRAIN ONLY**
- **Used in**: HtulTS.py
- **Purpose**: Warmup steps for TF-C loss weight
- **HtulTS usage**: Passed to LightweightModel
- **Cannot remove from pretrain**: Controls TF-C warmup
- **Not in finetune**: Not used in finetune

#### ✅ `--projection_dim` (128) - **PRETRAIN ONLY**
- **Used in**: HtulTS.py
- **Purpose**: Dimension of projection heads for contrastive learning
- **HtulTS usage**: Passed to LightweightModel
- **Cannot remove from pretrain**: Used in TF-C loss
- **Not in finetune**: Not needed in finetune

#### ⚠️ `--use_real_image` (1) - **TYPO, NOT USED**
- **Should be**: `use_real_imag` (real/imaginary FFT features)
- **HtulTS usage**: Model looks for `use_real_imag`, not `use_real_image`
- **Can remove**: Typo means it's not actually used
- **Fix**: Rename to `use_real_imag` or remove

---

### 7. FORGETTING MECHANISM PARAMETERS

#### ⚠️ `--use_forgetting` (0) - **DISABLED**
- **Used in**: HtulTS.py
- **Purpose**: Enable forgetting mechanisms during finetune
- **HtulTS usage**: `self.use_forgetting = getattr(args, 'use_forgetting', False)`
- **Current value**: 0 (disabled)
- **Can remove**: Set to 0, so not active
- **Impact if removed**: None (already disabled)

#### ⚠️ `--forgetting_type` (activation) - **NOT USED (use_forgetting=0)**
- **Used in**: HtulTS.py (only when use_forgetting=1)
- **Purpose**: Type of forgetting mechanism
- **Current state**: Irrelevant because use_forgetting=0
- **Can remove**: Not used when use_forgetting=0

#### ⚠️ `--forgetting_rate` (0.1) - **NOT USED (use_forgetting=0)**
- **Used in**: HtulTS.py (only when use_forgetting=1)
- **Purpose**: Forgetting strength
- **Current state**: Irrelevant because use_forgetting=0
- **Can remove**: Not used when use_forgetting=0

---

### 8. OTHER PARAMETERS

#### ✅ `--gpu` (2)
- **Used in**: exp_basic.py
- **Purpose**: GPU device ID
- **Usage**: `os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)`
- **Cannot remove**: Required for GPU selection

#### ⚠️ `--is_training` (1) - **FINETUNE ONLY, NOT USED**
- **Used in**: **NOWHERE**
- **Purpose**: Unknown (possibly legacy)
- **Can remove**: Not referenced in any code
- **Impact if removed**: None

#### ✅ `--use_norm` (1/0)
- **Used in**: HtulTS.py
- **Purpose**: Enable/disable input normalization
- **HtulTS usage**: `self.use_norm = args.use_norm`
- **Cannot remove**: Controls normalization behavior
- **Note**: Some scripts set this explicitly (PEMS), others use default

---

## RECOMMENDED CLEAN SCRIPTS

### PRETRAIN Script (Minimal)

```bash
python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1 \
    --model HtulTS \
    --data ETTh1 \
    --features M \
    --input_len 336 \
    --enc_in 7 \
    --learning_rate 0.0001 \
    --batch_size 16 \
    --train_epochs 50 \
    --lr_decay 0.9 \
    --gpu 2 \
    --tfc_weight 0.05 \
    --tfc_warmup_steps 500 \
    --projection_dim 128
```

**Removed from pretrain**:
- `--e_layers`, `--d_layers`, `--dec_in`, `--c_out` (not used in HtulTS)
- `--n_heads`, `--d_model`, `--d_ff` (not used in HtulTS)
- `--patch_len`, `--stride` (not used in HtulTS)
- `--head_dropout`, `--dropout` (not passed to model)
- `--time_steps`, `--scheduler` (not used in HtulTS)
- `--use_noise` (set to 0, not active)
- `--use_forgetting`, `--forgetting_type`, `--forgetting_rate` (set to 0, not active)
- `--use_real_image` (typo, not used)

### FINETUNE Script (Minimal)

```bash
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
        --gpu 2
done
```

**Removed from finetune**:
- `--e_layers`, `--d_layers`, `--dec_in`, `--c_out` (not used in HtulTS)
- `--n_heads`, `--d_model`, `--d_ff` (not used in HtulTS)
- `--patch_len`, `--stride` (not used in HtulTS)
- `--head_dropout`, `--dropout` (not passed to model)
- `--time_steps`, `--scheduler` (pretrain parameters)
- `--use_noise` (pretrain parameter, set to 0)
- `--use_forgetting`, `--forgetting_type`, `--forgetting_rate` (set to 0, not active)
- `--use_real_image` (typo, not used)
- `--projection_dim` (pretrain parameter)
- `--is_training` (not used anywhere)

---

## CODE CHANGES REQUIRED

### IF YOU REMOVE THESE PARAMETERS:

#### No changes needed to remove:
- `--e_layers`, `--d_layers` → Not referenced in HtulTS at all
- `--dec_in`, `--c_out` → Not referenced in HtulTS at all
- `--n_heads`, `--d_model`, `--d_ff` → Not referenced in HtulTS at all
- `--patch_len`, `--stride` → Not referenced in HtulTS at all
- `--time_steps`, `--scheduler` → Not referenced in HtulTS at all
- `--use_noise=0`, `--use_forgetting=0` → Already disabled, removal has no effect
- `--forgetting_type`, `--forgetting_rate` → Only used when use_forgetting=1
- `--use_real_image` → Typo, never used (should be use_real_imag)
- `--is_training` → Not referenced anywhere
- `--dropout`, `--head_dropout` → Not passed to model constructor

#### Changes needed in run.py (setting names):
If you remove `--e_layers`, `--d_layers`, etc., update the `setting` variable in run.py (lines 268-291 and 303-327) to remove references to these parameters from folder names. This is optional - they only affect experiment folder naming, not functionality.

Example change in run.py:
```python
# OLD:
setting = "{}_{}_{}_{}_il{}_ll{}_pl{}_dm{}_df{}_nh{}_el{}_dl{}...".format(
    args.task_name,
    args.model,
    args.data,
    args.features,
    args.input_len,
    args.label_len,
    args.pred_len,
    args.d_model,  # ← Remove these
    args.d_ff,     # ← Remove these
    args.n_heads,  # ← Remove these
    args.e_layers, # ← Remove these
    args.d_layers, # ← Remove these
    ...
)

# NEW:
setting = "{}_{}_{}_{}_il{}_ll{}_pl{}...".format(
    args.task_name,
    args.model,
    args.data,
    args.features,
    args.input_len,
    args.label_len,
    args.pred_len,
    # Removed: d_model, d_ff, n_heads, e_layers, d_layers
    ...
)
```

---

## SUMMARY TABLE

| Parameter | HtulTS Model | Exp_HtulTS | Can Remove | Notes |
|-----------|-------------|------------|------------|-------|
| `task_name` | ✅ Used | ✅ Used | ❌ No | Essential |
| `root_path` | ❌ | ✅ Used | ❌ No | Data loading |
| `data_path` | ❌ | ✅ Used | ❌ No | Data loading |
| `model_id` | ❌ | ❌ | ✅ Yes | Only in folder name |
| `model` | ❌ | ✅ Used | ❌ No | Model selection |
| `data` | ❌ | ✅ Used | ❌ No | Data loading |
| `features` | ❌ | ✅ Used | ❌ No | Data slicing |
| `input_len` | ✅ Used | ✅ Used | ❌ No | Essential |
| `label_len` | ❌ | ✅ Used | ⚠️ Finetune only | Not in pretrain |
| `pred_len` | ✅ Used | ✅ Used | ⚠️ Finetune only | Not in pretrain |
| `enc_in` | ✅ Used | ❌ | ❌ No | Input channels |
| **`e_layers`** | **❌ NOT USED** | **❌** | **✅ YES** | **Remove it** |
| **`d_layers`** | **❌ NOT USED** | **❌** | **✅ YES** | **Remove it** |
| **`dec_in`** | **❌ NOT USED** | **❌** | **✅ YES** | **Remove it** |
| **`c_out`** | **❌ NOT USED** | **❌** | **✅ YES** | **Remove it** |
| **`n_heads`** | **❌ NOT USED** | **❌** | **✅ YES** | **Remove it** |
| **`d_model`** | **❌ NOT USED** | **❌** | **✅ YES** | **Remove it** |
| **`d_ff`** | **❌ NOT USED** | **❌** | **✅ YES** | **Remove it** |
| **`patch_len`** | **❌ NOT USED** | **❌** | **✅ YES** | **Remove it** |
| **`stride`** | **❌ NOT USED** | **❌** | **✅ YES** | **Remove it** |
| `dropout` | ❌ | ❌ | ⚠️ Not passed | Can remove (not passed to model) |
| `head_dropout` | ❌ | ❌ | ⚠️ Not passed | Can remove (not passed to model) |
| `train_epochs` | ❌ | ✅ Used | ❌ No | Training loop |
| `batch_size` | ❌ | ✅ Used | ❌ No | Data loading |
| `learning_rate` | ❌ | ✅ Used | ❌ No | Optimizer |
| `lr_decay` | ❌ | ✅ Used (pretrain) | ⚠️ Pretrain only | Not in finetune |
| `lradj` | ❌ | ✅ Used (finetune) | ⚠️ Finetune only | Not in pretrain |
| `patience` | ❌ | ✅ Used (finetune) | ⚠️ Finetune only | Not in pretrain |
| `pct_start` | ❌ | ✅ Used (finetune) | ⚠️ Finetune only | Not in pretrain |
| **`time_steps`** | **❌ NOT USED** | **❌** | **✅ YES** | **Remove it** |
| **`scheduler`** | **❌ NOT USED** | **❌** | **✅ YES** | **Remove it** |
| **`use_noise`** | ❌ Disabled (0) | ❌ | **✅ YES** | **Remove (disabled)** |
| `tfc_weight` | ✅ Used (pretrain) | ❌ | ⚠️ Pretrain only | Not in finetune |
| `tfc_warmup_steps` | ✅ Used (pretrain) | ❌ | ⚠️ Pretrain only | Not in finetune |
| `projection_dim` | ✅ Used (pretrain) | ❌ | ⚠️ Pretrain only | Not in finetune |
| **`use_real_image`** | **❌ TYPO** | **❌** | **✅ YES** | **Remove (typo)** |
| **`use_forgetting`** | ❌ Disabled (0) | ❌ | **✅ YES** | **Remove (disabled)** |
| **`forgetting_type`** | ❌ Not used (use_forgetting=0) | ❌ | **✅ YES** | **Remove** |
| **`forgetting_rate`** | ❌ Not used (use_forgetting=0) | ❌ | **✅ YES** | **Remove** |
| `gpu` | ❌ | ✅ Used | ❌ No | Device selection |
| **`is_training`** | **❌ NOT USED** | **❌** | **✅ YES** | **Remove it** |
| `use_norm` | ✅ Used | ❌ | ❌ No | Normalization control |

---

## CONCLUSION

**Safe to remove (15 parameters):**
1. `--e_layers`
2. `--d_layers`
3. `--dec_in`
4. `--c_out`
5. `--n_heads`
6. `--d_model`
7. `--d_ff`
8. `--patch_len`
9. `--stride`
10. `--time_steps`
11. `--scheduler`
12. `--use_noise` (disabled)
13. `--use_forgetting` (disabled)
14. `--forgetting_type`
15. `--forgetting_rate`
16. `--use_real_image` (typo)
17. `--is_training`

**Optionally remove (not passed to model):**
- `--dropout`
- `--head_dropout`

**Code changes needed:** Only in run.py for setting names (optional).

No changes needed to HtulTS.py, exp_htults.py, or exp_basic.py!
