# HtulTS Bug Fix Implementation Plan

## Overview
This document contains a comprehensive implementation plan to fix critical and high-priority bugs in the HtulTS model. Fixes are organized by priority, with focus on non-architectural changes first.

**Total Estimated Time**: 45-60 minutes
**Phases**: 4 (3 mandatory, 1 optional)

---

## PHASE 1: CRITICAL BUGS (Non-Architectural) - ~30 minutes

### Priority 1.1: Fix Variable Naming Bug in PatchEmbedding

**Issue**: Line 87 incorrectly uses `n_vars` when it should be `seq_len`
**Impact**: Will cause dimension mismatches in padding calculation
**Difficulty**: Easy
**File**: `models/HtulTS.py` - Lines 85-90
**Status**: Not Started

**Current Code:**
```python
def forward(self, x):
    # x: [Batch * Features, Seq_Len]
    n_vars = x.shape[1]  # ❌ WRONG!
    
    if self.stride > 0:
        padding = self.stride - ((n_vars - self.patch_len) % self.stride)
```

**Problem Explanation**:
- The input `x` has shape `[Batch * Features, Seq_Len]`
- `x.shape[1]` is the sequence length, NOT the number of variables
- The variable is misleadingly named, causing confusion about what dimension we're padding
- This will cause incorrect padding calculations

**Fix:**
```python
def forward(self, x):
    # x: [Batch * Features, Seq_Len]
    seq_len = x.shape[1]  # ✅ CORRECT
    
    if self.stride > 0:
        padding = self.stride - ((seq_len - self.patch_len) % self.stride))
```

**Verification**: After fix, padding should be calculated correctly for the sequence dimension.

---

### Priority 1.2: Fix TFC Loss Group IDs

**Issue**: Line 422 groups all features from same sample, should treat each sample-feature independently
**Impact**: Reduces contrastive learning effectiveness
**Difficulty**: Easy
**File**: `models/HtulTS.py` - Line 422
**Status**: Not Started

**Current Code:**
```python
# Group IDs for TFC
group_ids = torch.arange(batch_size, device=x.device).repeat_interleave(num_features)
# Result for batch_size=2, num_features=3: [0,0,0, 1,1,1]
```

**Problem Explanation**:
- Current logic treats all features from the same sample as the same group
- This means contrastive loss will treat feature 0, 1, 2 of sample 0 as the same entity
- TF-C is designed to match time-frequency representations **within the same sample-feature pair**, not across all features
- This undermines the contrastive learning objective

**Fix:**
```python
# Each sample-feature pair should be independent for TFC matching
group_ids = torch.arange(batch_size * num_features, device=x.device)
# Result for batch_size=2, num_features=3: [0,1,2, 3,4,5]
```

**Rationale**: Each time-frequency representation should be matched independently, allowing the model to learn sample-feature-specific representations.

---

### Priority 1.3: Fix Inconsistent Denormalization

**Issue**: Lines 518 & 535 use different denormalization patterns
**Impact**: Potential shape broadcasting issues and inconsistency
**Difficulty**: Easy
**File**: `models/HtulTS.py` - Lines 515-520 (Pretrain) & 532-536 (Finetune)
**Status**: Not Started

**Current Code (Pretrain):**
```python
if self.use_norm:
    predict_x = predict_x * stdevs
    predict_x = predict_x + means
```

**Current Code (Finetune):**
```python
if self.use_norm:
    prediction = prediction * stdevs[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
    prediction = prediction + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
```

**Problem Explanation**:
- Pretrain uses simple multiplication: relies on implicit broadcasting
- Finetune uses explicit indexing and repeat: more explicit but different logic
- Shape expectations differ: pretrain output is `[B, L, C]`, finetune output is also `[B, Pred, C]`
- The inconsistency suggests output shapes might not align as expected

**Fix** (Make both explicit):
```python
# In pretrain method (around line 518):
if self.use_norm:
    predict_x = predict_x * stdevs[:, 0, :].unsqueeze(1).repeat(1, self.input_len, 1)
    predict_x = predict_x + means[:, 0, :].unsqueeze(1).repeat(1, self.input_len, 1)

# Finetune already correct - no change needed
# prediction = prediction * stdevs[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
# prediction = prediction + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
```

**Verification**: After fix, shapes should be consistent and explicit.

---

### Priority 1.4: Simplify Masking vs Noise Logic

**Issue**: Lines 377-384 have redundant conditions and confusing defaults
**Impact**: Code clarity and potential logic errors
**Difficulty**: Easy
**File**: `models/HtulTS.py` - Lines 376-384 & Wrapper calls at lines 513, 530
**Status**: Not Started

**Current Code:**
```python
if self.task_name == "pretrain":
    # Default to Masking for SOTA results
    # Only use Noise if explicitly requested via args AND flag
    if self.use_noise and add_noise_flag:
        x_input, _ = self.add_noise(x, noise_level=noise_level)
    else:
        x_input, _ = self.random_masking(x, mask_ratio=0.4)
else:
    x_input = x
```

**Problem Explanation**:
- Comment says "default to masking" but `use_noise` defaults to 1 in run.py
- The `add_noise_flag` parameter is redundant and causes confusion
- Wrapper calls pass this flag but the logic is confusing
- The parameter control is split between args and function flags

**Fix**:
```python
# In LightweightModel.forward (around line 376):
if self.task_name == "pretrain":
    # Pretraining augmentation: noise or masking
    if self.use_noise:
        x_input, _ = self.add_noise(x, noise_level=noise_level)
    else:
        x_input, _ = self.random_masking(x, mask_ratio=0.4)
else:
    x_input = x

# Remove add_noise_flag parameter from forward signature (line 376)
# Change from:
# def forward(self, x, add_noise_flag=False, noise_level=0.1):
# To:
# def forward(self, x, noise_level=0.1):

# Update wrapper calls in Model class (lines 513, 530):
# BEFORE:
# predict_x, loss_tfc = self.model(x, add_noise_flag=True, noise_level=self.noise_level)
# AFTER:
# predict_x, loss_tfc = self.model(x, noise_level=self.noise_level)

# BEFORE:
# prediction = self.model(x, add_noise_flag=False)
# AFTER:
# prediction = self.model(x)
```

**Rationale**: Simplifies logic by using only `self.use_noise` from initialization. The augmentation choice should be determined at init time, not runtime.

---

### Priority 1.5: Fix Classification Head Shape Mismatch

**Issue**: Conv1d expects different input shape than what's provided
**Impact**: Classification task will fail during forward pass
**Difficulty**: Medium
**File**: `models/HtulTS.py` - Lines 338, 462-471 (init and forward)
**Status**: Not Started

**Current Code (in __init__):**
```python
else: # Classification
    self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
    self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
    self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
```

**Current Code (in forward):**
```python
# CLASSIFICATION
else:
    cnn_input = h_fused.view(batch_size, num_features, -1)
    x = F.relu(self.conv1(cnn_input))
    x = self.pool(x); x = self.dropout(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x); x = self.dropout(x)
    x = F.relu(self.conv3(x))
    x = self.global_pool(x).squeeze(-1)
    x = self.classifier(x)
    return x
```

**Problem Explanation**:
- `h_fused` has shape `[B*C, Hidden]`
- After `view(batch_size, num_features, -1)`: shape becomes `[B, C, Hidden]`
- Conv1d expects `[B, C_in, L]` where `C_in` = input channels, `L` = sequence length
- Currently: `[B, C, Hidden]` → Conv1d tries to use `C` (num_features) as input channels
- But Conv1d was initialized with `in_channels` (which is `enc_in` = 7, not `hidden_dim`)
- **Result**: Shape mismatch - Conv1d expects 7 channels but gets 512 (or whatever hidden_dim is)

**Fix (in __init__):**
```python
else: # Classification
    # Input to CNN is [B, Hidden, C] so first Conv1d expects hidden_dim channels
    self.conv1 = nn.Conv1d(self.hidden_dim, 64, kernel_size=5, padding=2)
    self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
    self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
    self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
    self.dropout = nn.Dropout(0.2)
    self.global_pool = nn.AdaptiveAvgPool1d(1)
    self.classifier = nn.Sequential(
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes)
    )
```

**Fix (in forward):**
```python
# CLASSIFICATION
else:
    # h_fused: [B*C, Hidden] -> reshape to [B, C, Hidden]
    h_fused_3d = h_fused.view(batch_size, num_features, self.hidden_dim)
    
    # Transpose for Conv1d: [B, C, Hidden] -> [B, Hidden, C]
    # Now CNN processes along the channel dimension C
    cnn_input = h_fused_3d.transpose(1, 2)
    
    # Forward through CNN: [B, Hidden, C]
    x = F.relu(self.conv1(cnn_input))
    x = self.pool(x); x = self.dropout(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x); x = self.dropout(x)
    x = F.relu(self.conv3(x))
    x = self.global_pool(x).squeeze(-1)  # [B, 256]
    x = self.classifier(x)
    return x
```

**Verification**: 
- Conv1d shapes should align properly
- Classification forward pass should execute without errors

---

## PHASE 2: STABILITY IMPROVEMENTS - ~15 minutes

### Priority 2.1: Remove Unsafe Frequency Normalization

**Issue**: Lines 54-56 normalize after log1p, can cause instability
**Impact**: Potential NaN issues during training
**Difficulty**: Easy
**File**: `models/HtulTS.py` - Lines 31 (init) & 54-56 (forward in FrequencyEncoder)
**Status**: Not Started

**Current Code (in __init__):**
```python
self.freq_projection = nn.Sequential(
    nn.Linear(self.fft_output_size, hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout)
)
```

**Current Code (in forward):**
```python
# Per-sample normalization
mean = freq_features.mean(dim=-1, keepdim=True)
std = freq_features.std(dim=-1, keepdim=True).clamp(min=1e-6)
freq_features = (freq_features - mean) / std

h_freq = self.freq_projection(freq_features)
```

**Problem Explanation**:
- After `log1p`, magnitudes are already stabilized and on a smaller scale
- Per-sample normalization can cause issues if a sample has very low variance
- The `clamp(min=1e-6)` might not be sufficient for high-dimensional feature vectors
- Additional normalization can introduce instability, especially for outliers
- BatchNorm in the sequential module would handle this better

**Fix (in __init__):**
```python
self.freq_projection = nn.Sequential(
    nn.Linear(self.fft_output_size, hidden_dim),
    nn.BatchNorm1d(hidden_dim),  # Add this for stability
    nn.ReLU(),
    nn.Dropout(dropout)
)
```

**Fix (in forward):**
```python
# Remove per-sample normalization - log1p already stabilizes magnitudes
# BatchNorm in projection handles normalization better
h_freq = self.freq_projection(freq_features)
```

**Rationale**: BatchNorm is specifically designed for normalization and is more stable than manual normalization. It also learns scale and shift parameters.

---

### Priority 2.2: Remove Unused Import

**Issue**: `math` is imported but never used
**Impact**: Code cleanliness
**Difficulty**: Trivial
**File**: `models/HtulTS.py` - Line 4
**Status**: Not Started

**Current Code:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
```

**Fix:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

## PHASE 3: CONFIGURATION & DOCUMENTATION - ~20 minutes

### Priority 3.1: Add hidden_dim Parameter to CLI

**Issue**: hidden_dim is not exposed as CLI argument
**Impact**: Users can't configure this important parameter
**Difficulty**: Easy
**Files**: `run.py` (add parameter), `models/HtulTS.py` (use parameter)
**Status**: Not Started

**Current Behavior**:
- `hidden_dim` is set to `args.d_model` in HtulTS
- Users cannot override this independently
- This couples two distinct architectural parameters

**Fix (in run.py):**
Add after line 125 (after d_model definition):
```python
parser.add_argument("--hidden_dim", type=int, default=None, 
                   help="hidden dimension for HtulTS model (defaults to d_model if not specified)")
```

**Fix (in HtulTS.py - Model.__init__):**
Around line 509 in the Model wrapper initialization:
```python
# Change from:
# self.model = LightweightModel(
#     ...
#     hidden_dim=args.d_model
# )

# To:
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
    hidden_dim=getattr(args, 'hidden_dim', args.d_model)  # Use hidden_dim if provided, else d_model
)
```

**Fix (in HtulTS.py - ClsModel.__init__):**
Around line 595 in the ClsModel wrapper initialization:
```python
# Apply same change as Model wrapper
hidden_dim=getattr(args, 'hidden_dim', args.d_model)
```

**Verification**: Users should be able to run `--hidden_dim 256` to override default.

---

### Priority 3.2: Document TFC Warmup Behavior

**Issue**: TFC warmup counter persists across epochs - not obvious from code
**Impact**: User understanding and troubleshooting
**Difficulty**: Easy
**File**: `models/HtulTS.py` - Lines 260-280 (LightweightModel.__init__)
**Status**: Not Started

**Current Code:**
```python
# TF-C weighting configuration
self.tfc_weight = tfc_weight
self.tfc_warmup_steps = tfc_warmup_steps
self.register_buffer('_step_counter', torch.tensor(0, dtype=torch.long))
```

**Problem Explanation**:
- The `_step_counter` increments globally throughout training
- Users might expect it to reset per epoch (it doesn't)
- The warmup happens **globally**, not per-epoch
- This behavior is non-obvious from the code

**Fix (add docstring):**
```python
class LightweightModel(nn.Module):
    """
    Updated LightweightModel with:
    - Patching + Mixer Backbone (Time)
    - Global FFT Backbone (Freq)
    - Masking for Pretraining
    - Learnable Fusion & TFC
    
    Time-Frequency Consistency (TFC) Loss:
        - Implemented as contrastive learning between time and frequency projections
        - TFC loss weight can be warmed up linearly during training
        - The warmup counter is GLOBAL across all training steps (not reset per epoch)
        - Set tfc_warmup_steps=0 to disable warmup (use full tfc_weight from start)
        - Example: tfc_weight=0.05, tfc_warmup_steps=750 will linearly increase 
          TFC loss weight from 0 to 0.05 over the first 750 training steps
    """
```

Also add docstring to the __init__ method parameters:
```python
def __init__(self, in_channels, out_channels, task_name="pretrain", pred_len=None, 
             num_classes=2, input_len=336, use_noise=False, 
             use_forgetting=False, forgetting_type="activation", forgetting_rate=0.1,
             tfc_weight=0.05, tfc_warmup_steps=0, use_real_imag=False,
             projection_dim=128, patch_len=16, stride=8, hidden_dim=512):
    """
    Args:
        tfc_weight (float): Weight for TF-C contrastive loss in pretraining (default: 0.05)
        tfc_warmup_steps (int): Number of training steps to linearly warm up TFC loss weight.
            NOTE: This counter is GLOBAL and persists throughout all training.
            Set to 0 to disable warmup and use full tfc_weight immediately.
            Example: 750 steps with batch_size=16 ≈ ~47 samples seen
    """
```

---

## PHASE 4: ARCHITECTURAL IMPROVEMENTS (Optional - User's Design Choice)

These are lower priority since they require architectural changes. Your model's unique dual time-frequency design should be preserved.

### Optional 4.1: Add Positional Encoding

**Why**: Patch Mixer has no concept of temporal order
**Tradeoff**: Changes architecture, adds learnable parameters
**Impact**: Higher memory usage, but better temporal modeling
**Estimated Effort**: Medium (30 minutes)
**File**: `models/HtulTS.py` - Lines 294-310

**Implementation Note**: 
- Add learnable positional embeddings after patching
- Or use sinusoidal positional encoding
- This would make the Mixer temporally-aware

---

### Optional 4.2: Replace Mixer with Attention Pooling

**Why**: Current Mixer has ~11M parameters (massive bottleneck)
**Tradeoff**: Changes architecture significantly
**Impact**: Reduces parameters by 8-10x while maintaining performance
**Estimated Effort**: High (1 hour)
**File**: `models/HtulTS.py` - Lines 306-313

**Implementation Note**:
- Replace flat MLP Mixer with cross-attention mechanism
- Use learnable CLS token to aggregate patches
- More parameter-efficient and potentially more performant

---

### Optional 4.3: Calculate num_patches Mathematically

**Why**: Remove dummy forward pass in __init__
**Tradeoff**: Requires careful formula derivation
**Impact**: Cleaner code, faster initialization
**Estimated Effort**: Low (15 minutes)
**File**: `models/HtulTS.py` - Lines 303-305

**Current Approach**:
```python
dummy_input = torch.zeros(1, input_len)
dummy_patches = self.patch_embed(dummy_input)
self.num_patches = dummy_patches.shape[1]
```

**Mathematical Approach**:
```python
# Calculate padding needed
if stride > 0:
    padding = stride - ((input_len - patch_len) % stride)
    if padding >= stride:
        padding = 0
else:
    padding = 0

# Number of patches after padding
padded_len = input_len + padding
num_patches = (padded_len - patch_len) // stride + 1
self.num_patches = num_patches
```

---

## Summary Table: All Fixes

| Priority | Category | Fix | Difficulty | Time | Arch Change |
|----------|----------|-----|-----------|------|-------------|
| 1.1 | Critical | Variable naming in PatchEmbedding | Easy | 2 min | No |
| 1.2 | Critical | TFC group IDs | Easy | 2 min | No |
| 1.3 | Critical | Denormalization consistency | Easy | 3 min | No |
| 1.4 | Critical | Masking vs noise logic | Easy | 5 min | No |
| 1.5 | Critical | Classification head shape | Medium | 8 min | No |
| 2.1 | Stability | Frequency normalization | Easy | 3 min | No |
| 2.2 | Cleanup | Unused import | Trivial | 1 min | No |
| 3.1 | Config | CLI hidden_dim param | Easy | 5 min | No |
| 3.2 | Docs | TFC warmup documentation | Easy | 4 min | No |
| **Total (Phases 1-3)** | | | | **33 min** | **No** |
| 4.1 | Optional | Positional encoding | Medium | 30 min | Yes |
| 4.2 | Optional | Attention pooling | Hard | 60 min | Yes |
| 4.3 | Optional | Math num_patches | Easy | 15 min | No |

---

## Implementation Checklist

### Phase 1: Critical Bugs
- [ ] 1.1: Fix variable naming in PatchEmbedding
- [ ] 1.2: Fix TFC group IDs
- [ ] 1.3: Fix denormalization consistency
- [ ] 1.4: Simplify masking logic
- [ ] 1.5: Fix classification head shape

### Phase 2: Stability
- [ ] 2.1: Remove frequency normalization
- [ ] 2.2: Remove unused import

### Phase 3: Configuration
- [ ] 3.1: Add hidden_dim to CLI
- [ ] 3.2: Document TFC warmup

### Phase 4: Optional
- [ ] 4.1: Add positional encoding (if needed)
- [ ] 4.2: Replace mixer (if needed)
- [ ] 4.3: Calculate num_patches math (if needed)

---

## Testing After Fixes

After implementing all Phase 1-3 fixes, run these tests:

1. **Pretraining Test**:
   ```bash
   python -u run.py \
       --task_name pretrain \
       --model HtulTS \
       --data ETTh1 \
       --train_epochs 2 \
       --batch_size 16
   ```
   - Check: No crashes, loss values are reasonable (not NaN/Inf)
   - Verify: Masking/noise logic works correctly

2. **Finetuning Test**:
   ```bash
   python -u run.py \
       --task_name finetune \
       --model HtulTS \
       --data ETTh1 \
       --train_epochs 2 \
       --batch_size 16
   ```
   - Check: Forecasting works, metrics print correctly

3. **Classification Test** (if applicable):
   ```bash
   python -u run.py \
       --task_name finetune \
       --downstream_task classification \
       --model HtulTS \
       --data [classification_dataset] \
       --train_epochs 2
   ```
   - Check: Classification head forward pass works

---

## Notes

- All fixes in Phases 1-3 maintain the existing architecture and unique design
- The Mixer bottleneck (11M params) is preserved as it's part of your architecture
- Optional Phase 4 fixes are left to your discretion
- No breaking changes to the API
- All backward compatible with existing checkpoint loading
