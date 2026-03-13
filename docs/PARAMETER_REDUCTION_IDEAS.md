# Parameter Reduction Strategies for HtulTS

## ✅ Fixed: d_model Issue

The d_model parameter now **properly controls model size**:

| d_model | Parameters | vs d=512 | Memory Savings |
|---------|-----------|----------|----------------|
| 64      | 272,688   | 45x smaller | ~98% |
| 128     | 843,856   | 14.5x smaller | ~93% |
| 256     | 3,161,680 | 3.9x smaller | ~74% |
| **512** | **12,221,008** | **baseline** | **0%** |

---

## Current Problem: patch_mixer Dominates (90% of params)

The patch_mixer's first linear layer accounts for most parameters:
- **Formula**: `num_patches × hidden_dim × hidden_dim`
- **With d_model=512**: `42 × 512 × 512 = 11,010,048 params`
- This is **unavoidable** with the current flatten-then-project architecture

---

## 🎯 Strategy 1: Reduce d_model (Easiest, Immediate)

**Impact**: Quadratic reduction in parameters

```bash
# Pretrain script - change from 128 to 64 or 96
--d_model 64   # 272,688 params (98% reduction)
--d_model 96   # 524,400 params (96% reduction)
--d_model 128  # 843,856 params (93% reduction)
```

**Pros**:
- ✅ No code changes needed (already fixed)
- ✅ Dramatic parameter reduction
- ✅ Faster training and inference

**Cons**:
- ⚠️ May reduce model capacity/accuracy
- ⚠️ Need to retune hyperparameters

**Recommendation**: Start with `d_model=128` or `d_model=256` for good balance.

---

## 🎯 Strategy 2: Replace patch_mixer with Efficient Alternatives

### Option 2A: Use Attention Pooling Instead of Flatten

**Current approach** (inefficient):
```python
# Flatten all patches: 42 × 512 = 21,504 dimensional vector
x_patches_flat = x_patches.view(x_patches.size(0), -1)
# Project 21,504 → 512 (11M params!)
h_time = self.patch_mixer(x_patches_flat)
```

**Better approach** (attention pooling):
```python
# Use attention to aggregate patches (no flattening)
self.patch_aggregator = nn.MultiheadAttention(
    embed_dim=hidden_dim,
    num_heads=8,
    batch_first=True
)

# Create a learnable query token
self.agg_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

# In forward:
query = self.agg_query.expand(batch_size, -1, -1)
h_time, _ = self.patch_aggregator(query, x_patches, x_patches)
h_time = h_time.squeeze(1)  # [B*C, hidden_dim]
```

**Parameter savings**:
- Attention: ~1.0M params (vs 11M)
- **Reduction: 90%**

### Option 2B: Use Depthwise Separable Convolution

```python
self.patch_aggregator = nn.Sequential(
    # Depthwise conv across patches
    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, 
              padding=1, groups=hidden_dim),
    nn.BatchNorm1d(hidden_dim),
    nn.ReLU(),
    # Pointwise conv
    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten()
)
```

**Parameter savings**:
- Depthwise + Pointwise: ~262K params
- **Reduction: 97.5%**

### Option 2C: Use Gated Recurrent Aggregation

```python
self.patch_aggregator = nn.GRU(
    input_size=hidden_dim,
    hidden_size=hidden_dim,
    num_layers=2,
    batch_first=True
)

# In forward:
_, h_time = self.patch_aggregator(x_patches)
h_time = h_time[-1]  # Take last layer hidden state
```

**Parameter savings**:
- GRU: ~3.1M params
- **Reduction: 72%**

---

## 🎯 Strategy 3: Reduce Number of Patches

Current: `num_patches = ⌊(336 - 16) / 8⌋ + 2 = 42`

### Option 3A: Increase Stride
```bash
--patch_len 16
--stride 16    # Instead of 8
# Result: 22 patches instead of 42
# Parameter reduction: ~50%
```

### Option 3B: Increase Patch Length
```bash
--patch_len 32  # Instead of 16
--stride 16
# Result: 20 patches instead of 42
# Parameter reduction: ~52%
```

**Trade-off**: Fewer patches = less fine-grained temporal resolution

---

## 🎯 Strategy 4: Factorize the Mixer Layer

Instead of one huge `21,504 → 512` projection, use factorization:

```python
# Current (expensive):
nn.Linear(num_patches * hidden_dim, hidden_dim)  # 11M params

# Factorized (cheaper):
nn.Sequential(
    nn.Linear(num_patches * hidden_dim, bottleneck_dim),  # Compress
    nn.ReLU(),
    nn.Linear(bottleneck_dim, hidden_dim)  # Expand
)

# With bottleneck_dim=1024:
# Params = (21,504 × 1,024) + (1,024 × 512) = 22.5M → still large
# Need bottleneck_dim=512 for meaningful reduction:
# Params = (21,504 × 512) + (512 × 512) = 11.3M (minimal gain)
```

**Verdict**: Not effective for this architecture.

---

## 🎯 Strategy 5: Share Weights Across Components

### Share Projection Heads
```python
# Current: Separate projections for time and freq
self.time_projection = ProjectionHead(hidden_dim, hidden_dim//2, proj_dim)
self.freq_projection = ProjectionHead(hidden_dim, hidden_dim//2, proj_dim)

# Shared: Single projection for both
self.shared_projection = ProjectionHead(hidden_dim, hidden_dim//2, proj_dim)

# In forward:
z_time = self.shared_projection(h_time)
z_freq = self.shared_projection(h_freq)
```

**Savings**: ~164K params (1.3%)

---

## 🎯 Strategy 6: Quantization & Pruning (Post-training)

### Weight Quantization
- Convert FP32 → INT8 or FP16
- **Memory reduction**: 4x (FP32→INT8) or 2x (FP32→FP16)
- Minimal accuracy loss with proper calibration

### Structured Pruning
- Remove entire neurons/channels with low importance
- **Parameter reduction**: 30-50% typical
- Requires retraining/fine-tuning

---

## 🎯 Strategy 7: Use LoRA-style Low-Rank Adaptation

For fine-tuning, freeze most weights and add low-rank adaptations:

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.base.weight.requires_grad = False  # Freeze
        
        # Low-rank adaptation
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))
```

**Trainable params**: `rank × (in_features + out_features)`
- Original: 11M params
- LoRA (rank=16): ~344K trainable params
- **Reduction: 97%** (for fine-tuning)

---

## 📊 Recommended Implementation Strategy

### Phase 1: Quick Wins (No Architecture Changes)
1. ✅ **Use smaller d_model**: Try 128 or 256 instead of 512
2. ✅ **Increase stride**: Change from 8 to 12 or 16
3. ✅ **Share projection heads**: Merge time & freq projections

**Expected reduction**: ~85% with d_model=128 + stride=16

### Phase 2: Architecture Improvements
4. 🔄 **Replace patch_mixer with attention pooling** (Option 2A)
5. 🔄 **Or use depthwise separable conv** (Option 2B)

**Expected reduction**: ~90-95% total (vs original d=512)

### Phase 3: Advanced Optimization
6. 🔬 **Quantization**: FP32 → FP16 or INT8
7. 🔬 **LoRA for fine-tuning**: Freeze pretrained, adapt efficiently

---

## 💡 Practical Code Changes for Quick Wins

### Change 1: Update Pretrain Script
```bash
# scripts/pretrain/ETTh1.sh
--d_model 128    # Instead of 128 (already set, keep it!)
--stride 12      # Instead of 8 (reduce patches from 42 to 28)
```

**Result**: 
- Params: ~500K (vs 12.2M)
- **Reduction: 96%**

### Change 2: Share Projections in HtulTS.py

```python
# Around line 318, replace:
self.time_projection = ProjectionHead(self.hidden_dim, self.hidden_dim // 2, projection_dim)
self.freq_projection = ProjectionHead(self.hidden_dim, self.hidden_dim // 2, projection_dim)

# With:
self.shared_projection = ProjectionHead(self.hidden_dim, self.hidden_dim // 2, projection_dim)

# Then in forward (around line 425):
z_time = self.shared_projection(h_time)
z_freq = self.shared_projection(h_freq)
```

### Change 3: Replace patch_mixer (More Involved)

Add this new class after LearnableFusion:

```python
class EfficientPatchAggregator(nn.Module):
    """Replaces inefficient flatten-and-project with attention pooling."""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.2):
        super().__init__()
        self.aggregation_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x_patches):
        # x_patches: [B*C, num_patches, hidden_dim]
        batch_size = x_patches.size(0)
        query = self.aggregation_token.expand(batch_size, -1, -1)
        h_time, _ = self.attention(query, x_patches, x_patches)
        h_time = h_time.squeeze(1)  # [B*C, hidden_dim]
        h_time = self.norm(h_time)
        return h_time
```

Then replace patch_mixer initialization (around line 305):

```python
# OLD:
self.patch_mixer = nn.Sequential(
    nn.Linear(self.num_patches * self.hidden_dim, self.hidden_dim),
    nn.BatchNorm1d(self.hidden_dim),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(self.hidden_dim, self.hidden_dim)
)

# NEW:
self.patch_mixer = EfficientPatchAggregator(
    hidden_dim=self.hidden_dim,
    num_heads=8,
    dropout=0.2
)
```

And update forward (around line 410):

```python
# OLD:
x_patches_flat = x_patches.view(x_patches.size(0), -1)
h_time = self.patch_mixer(x_patches_flat)

# NEW:
h_time = self.patch_mixer(x_patches)  # No flattening needed!
```

**Params with this change**:
- d_model=128: ~400K (vs 843K)
- d_model=512: ~2.2M (vs 12.2M)
- **Reduction: ~82%**

---

## Summary Table

| Strategy | Code Changes | Param Reduction | Accuracy Impact |
|----------|-------------|-----------------|-----------------|
| Use d_model=128 | ✅ None (fixed) | 93% | Low-Medium |
| Use d_model=64 | ✅ None (fixed) | 98% | Medium |
| Increase stride to 16 | ✅ 1 line in script | 50% | Low |
| Share projections | ✅ 3 lines | 1.3% | Minimal |
| Attention pooling | 🔄 ~30 lines | 82% | Low-Medium |
| Depthwise conv | 🔄 ~20 lines | 97.5% | Medium |
| LoRA fine-tuning | 🔄 ~50 lines | 97%* | Low |

*For fine-tuning only

---

## Recommended Action Plan

**For immediate ~95% reduction with minimal risk**:
1. Keep d_model=128 (already in pretrain script)
2. Change stride from 8 to 12 in scripts
3. Share projection heads (3-line change)

**For ~98% reduction with architecture improvement**:
4. Implement attention pooling to replace patch_mixer
5. This requires retraining pretrained models

**Conservative recommendation**: Start with d_model=128 + stride=12, which gives you ~500K params (96% reduction) without any architecture changes.
