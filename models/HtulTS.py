"""
HtulTS v2 - Improved Architecture (d_model=128 compatible)
===========================================================
Changes marked with:
- # NEW: New addition
- # FIX: Bug fix or improvement  
- # REMOVED: Removed component (commented out or deleted)
- # MODIFIED: Modified from original

Compatible with TimeDART benchmark settings (d_model=128, depth from args)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    Crucial for SOTA on ETTh1 to handle distribution shift.
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode:str):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps)
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + 1e-10)
            x = x * self.stdev + self.mean
            return x


# REMOVED: SpectrallyGatedMixerLayer - Replaced by HierarchicalMixerBlock
# class SpectrallyGatedMixerLayer(nn.Module):
#     """Old mixer layer - replaced by HierarchicalMixerBlock"""
#     pass


# NEW: Hierarchical Multi-Scale Mixer Block
class HierarchicalMixerBlock(nn.Module):
    """
    NEW: Multi-scale mixer block with local, medium, and global receptive fields.
    Replaces SpectrallyGatedMixerLayer for better multi-scale modeling.
    """
    def __init__(self, num_patches, hidden_dim, dropout=0.1):
        super(HierarchicalMixerBlock, self).__init__()
        
        # NEW: Local mixing (adjacent patches)
        self.local_mix = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        )
        
        # NEW: Medium-range mixing (dilated convolution)
        self.medium_mix = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2, groups=hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        )
        
        # Global mixing (similar to original but improved)
        self.global_mix = nn.Sequential(
            nn.Linear(num_patches, num_patches),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Channel mixing with expansion factor
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # NEW: Learnable scale combination weights
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Frequency gating (kept from original)
        self.freq_gate = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, h_freq=None):
        """
        Args:
            x: [B*C, P, H] - Patch sequence
            h_freq: [B*C, H] - Optional frequency context for gating
        """
        residual = x
        x = self.norm1(x)
        
        # NEW: Multi-scale token mixing
        x_t = x.transpose(1, 2)  # [B*C, H, P] for conv1d
        
        local_out = self.local_mix(x_t).transpose(1, 2)   # [B*C, P, H]
        medium_out = self.medium_mix(x_t).transpose(1, 2)
        global_out = self.global_mix(x.transpose(1, 2)).transpose(1, 2)
        
        # NEW: Weighted combination of scales
        w = F.softmax(self.scale_weights, dim=0)
        x = w[0] * local_out + w[1] * medium_out + w[2] * global_out
        
        # Frequency gating (kept from original)
        if h_freq is not None:
            gate = 1 + 0.1 * torch.tanh(self.freq_gate(h_freq)).unsqueeze(1)
            x = x * gate
        
        x = residual + self.dropout(x)
        
        # Channel mixing
        x = x + self.channel_mix(x)
        
        return x


# NEW: Deep Mixer Stack with Skip Connections
class DeepMixerStack(nn.Module):
    """
    NEW: Deeper mixer stack with skip connections every 2 layers.
    Allows increasing depth without gradient issues.
    """
    def __init__(self, num_patches, hidden_dim, depth=4, dropout=0.1):
        super(DeepMixerStack, self).__init__()
        self.depth = depth
        
        self.layers = nn.ModuleList([
            HierarchicalMixerBlock(num_patches, hidden_dim, dropout)
            for _ in range(depth)
        ])
        
        # NEW: Skip connections every 2 layers for gradient flow
        self.skip_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) if i % 2 == 1 else nn.Identity()
            for i in range(depth)
        ])
        
    def forward(self, x, h_freq=None):
        """
        Args:
            x: [B*C, P, H] - Patch sequence
            h_freq: [B*C, H] - Optional frequency context
        """
        skip = None
        for i, (layer, skip_proj) in enumerate(zip(self.layers, self.skip_projs)):
            if i % 2 == 0:
                skip = x
            
            x = layer(x, h_freq)
            
            if i % 2 == 1 and skip is not None:
                x = x + skip_proj(skip)
        
        return x


class FrequencyEncoder(nn.Module):
    """
    Frequency-domain encoder using FFT with improved stability.
    Takes time-domain input and produces frequency embeddings.
    """
    def __init__(self, input_len, hidden_dim=512, dropout=0.2, use_real_imag=False):
        super(FrequencyEncoder, self).__init__()
        self.input_len = input_len
        self.hidden_dim = hidden_dim
        self.use_real_imag = use_real_imag
        
        # FFT output size: L//2 + 1 complex values
        self.num_freq_bins = input_len // 2 + 1
        
        # Feature size depends on representation mode
        if use_real_imag:
            self.fft_output_size = 2 * self.num_freq_bins
        else:
            self.fft_output_size = 3 * self.num_freq_bins
        
        # Register Hann window as buffer
        self.register_buffer('hann_window', torch.hann_window(input_len))
        
        self.freq_projection = nn.Sequential(
            nn.Linear(self.fft_output_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: [N, L] time-domain signal
        x_windowed = x * self.hann_window
        spectrum = torch.fft.rfft(x_windowed, dim=-1)
        
        if self.use_real_imag:
            real_part = spectrum.real
            imag_part = spectrum.imag
            freq_features = torch.cat([real_part, imag_part], dim=-1)
        else:
            amplitude = torch.log1p(torch.abs(spectrum))
            phase = torch.angle(spectrum)
            sin_phase = torch.sin(phase)
            cos_phase = torch.cos(phase)
            freq_features = torch.cat([amplitude, sin_phase, cos_phase], dim=-1)
        
        # Per-sample normalization
        mean = freq_features.mean(dim=-1, keepdim=True)
        std = freq_features.std(dim=-1, keepdim=True).clamp(min=1e-6)
        freq_features = (freq_features - mean) / std
        
        h_freq = self.freq_projection(freq_features)
        return h_freq


class PatchEmbedding(nn.Module):
    """
    Patchifies the time series.
    Input: [Batch, Length] -> Output: [Batch, Num_Patches, Patch_Dim]
    """
    def __init__(self, seq_len, patch_len, stride, d_model, dropout=0.1):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        self.num_patches = int((seq_len - patch_len) / stride) + 2
        
        self.value_embedding = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        
        if self.stride > 0:
            padding = self.stride - ((n_vars - self.patch_len) % self.stride)
            if padding < self.stride:
                 x = F.pad(x, (0, padding), "replicate")
        
        x_patched = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        x_emb = self.value_embedding(x_patched)
        
        return self.dropout(x_emb)


class TFC_Loss(nn.Module):
    """
    Time-Frequency Consistency Loss using NT-Xent with group support.
    """
    def __init__(self, temperature=0.2):
        super(TFC_Loss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z_time, z_freq, group_ids=None):
        batch_size = z_time.size(0)
        device = z_time.device
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        z_time = F.normalize(z_time, dim=-1)
        z_freq = F.normalize(z_freq, dim=-1)
        
        sim_tf = torch.matmul(z_time, z_freq.T) / self.temperature
        
        if group_ids is not None:
            same_group_mask = group_ids.unsqueeze(0) == group_ids.unsqueeze(1)
            diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
            mask_out = same_group_mask & ~diag_mask
            
            valid_negatives = ~same_group_mask | diag_mask
            has_valid_negatives = (valid_negatives & ~diag_mask).any(dim=1)
            
            if not has_valid_negatives.all():
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            sim_tf = sim_tf.masked_fill(mask_out, float('-inf'))
        
        labels = torch.arange(batch_size, device=device)
        loss_tf = F.cross_entropy(sim_tf, labels)
        loss_ft = F.cross_entropy(sim_tf.T, labels)
        
        loss = (loss_tf + loss_ft) / 2.0
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss


# NEW: Multi-Scale TFC Loss
class MultiScaleTFCLoss(nn.Module):
    """
    NEW: Hierarchical Time-Frequency Consistency Loss.
    Aligns representations at both global and patch levels.
    """
    def __init__(self, hidden_dim, num_patches, temperature=0.2):
        super(MultiScaleTFCLoss, self).__init__()
        self.temperature = temperature
        self.num_patches = num_patches
        
        # Global projections
        self.global_time_proj = ProjectionHead(hidden_dim, hidden_dim // 2, 64)
        self.global_freq_proj = ProjectionHead(hidden_dim, hidden_dim // 2, 64)
        
        # NEW: Patch-level projections
        self.patch_time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 32)
        )
        self.patch_freq_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 32)
        )
        
        # NEW: Learnable loss weights
        self.loss_weights = nn.Parameter(torch.tensor([0.7, 0.3]))
    
    def forward(self, h_time, h_freq, h_time_seq, group_ids=None):
        batch_size = h_time.size(0)
        device = h_time.device
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Global TFC Loss
        z_time_g = F.normalize(self.global_time_proj(h_time), dim=-1)
        z_freq_g = F.normalize(self.global_freq_proj(h_freq), dim=-1)
        
        sim_tf = torch.matmul(z_time_g, z_freq_g.T) / self.temperature
        
        if group_ids is not None:
            same_group_mask = group_ids.unsqueeze(0) == group_ids.unsqueeze(1)
            diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
            mask_out = same_group_mask & ~diag_mask
            sim_tf = sim_tf.masked_fill(mask_out, float('-inf'))
        
        labels = torch.arange(batch_size, device=device)
        loss_global = (F.cross_entropy(sim_tf, labels) + 
                       F.cross_entropy(sim_tf.T, labels)) / 2
        
        # NEW: Patch-Level TFC Loss
        z_time_p = F.normalize(self.patch_time_proj(h_time_seq), dim=-1)
        h_freq_exp = h_freq.unsqueeze(1).expand(-1, self.num_patches, -1)
        z_freq_p = F.normalize(self.patch_freq_proj(h_freq_exp), dim=-1)
        
        pos_sim = (z_time_p * z_freq_p).sum(dim=-1).mean(dim=1)
        neg_sim = torch.bmm(z_time_p, z_freq_p.transpose(1, 2))
        mask = ~torch.eye(self.num_patches, dtype=torch.bool, device=device)
        neg_sim = neg_sim[:, mask].view(batch_size, -1)
        
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / self.temperature
        local_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        loss_local = F.cross_entropy(logits, local_labels)
        
        # Combine with learnable weights
        weights = F.softmax(self.loss_weights, dim=0)
        total_loss = weights[0] * loss_global + weights[1] * loss_local
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


# REMOVED: ForgettingMechanisms - Was disabled and hurt performance
# class ForgettingMechanisms(nn.Module):
#     """REMOVED: This class was disabled (use_forgetting=0) and hurt performance."""
#     pass


# REMOVED: LearnableFusion - Replaced by CrossModalFusion
# class LearnableFusion(nn.Module):
#     """REMOVED: Simple Concat+MLP fusion - replaced by CrossModalFusion"""
#     pass


# NEW: Cross-Modal Attention Fusion
class CrossModalFusion(nn.Module):
    """
    NEW: Bidirectional Cross-Attention between Time and Frequency representations.
    Replaces the simple Concat+MLP fusion (LearnableFusion class).
    
    MODIFIED: num_heads adjusted for d_model=128 (must divide evenly)
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(CrossModalFusion, self).__init__()
        self.hidden_dim = hidden_dim
        # FIX: Ensure num_heads divides hidden_dim evenly
        self.num_heads = min(num_heads, hidden_dim // 16)  # At least 16 dim per head
        if self.num_heads < 1:
            self.num_heads = 1
        
        # NEW: Time attends to Frequency
        self.time_to_freq_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=self.num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # NEW: Frequency attends to Time  
        self.freq_to_time_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout=dropout, 
            batch_first=True
        )
        
        # NEW: Gated combination
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Final projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.norm_time = nn.LayerNorm(hidden_dim)
        self.norm_freq = nn.LayerNorm(hidden_dim)
    
    def forward(self, h_time, h_freq, h_time_seq=None):
        """
        Args:
            h_time: [B*C, H] - Global time representation
            h_freq: [B*C, H] - Global frequency representation
            h_time_seq: [B*C, P, H] - Patch sequence (optional)
        """
        h_time_exp = h_time.unsqueeze(1)
        h_freq_exp = h_freq.unsqueeze(1)
        
        # NEW: Use patch sequence for richer cross-attention if available
        if h_time_seq is not None:
            h_time_enhanced, _ = self.time_to_freq_attn(
                query=h_time_seq,
                key=h_freq_exp,
                value=h_freq_exp
            )
            h_time_enhanced = h_time_enhanced.mean(dim=1)
        else:
            h_time_enhanced, _ = self.time_to_freq_attn(
                query=h_time_exp, key=h_freq_exp, value=h_freq_exp
            )
            h_time_enhanced = h_time_enhanced.squeeze(1)
        
        h_freq_enhanced, _ = self.freq_to_time_attn(
            query=h_freq_exp, key=h_time_exp, value=h_time_exp
        )
        h_freq_enhanced = h_freq_enhanced.squeeze(1)
        
        h_time_out = self.norm_time(h_time + h_time_enhanced)
        h_freq_out = self.norm_freq(h_freq + h_freq_enhanced)
        
        # NEW: Gated combination
        gate_input = torch.cat([h_time_out, h_freq_out], dim=-1)
        gate_weight = self.gate(gate_input)
        
        h_gated = gate_weight * h_time_out + (1 - gate_weight) * h_freq_out
        h_fused = self.out_proj(torch.cat([h_gated, h_time_out + h_freq_out], dim=-1))
        
        return h_fused, gate_weight


# REMOVED: ResidualForecastingHead - Replaced by FrequencyAwareForecastingHead
# class ResidualForecastingHead(nn.Module):
#     """REMOVED: Old forecasting head that ignored frequency features"""
#     pass


# NEW: Frequency-Aware Forecasting Head
class FrequencyAwareForecastingHead(nn.Module):
    """
    NEW: FIX for critical bug - original head ignored frequency information!
    This head properly utilizes both time and frequency representations.
    """
    def __init__(self, hidden_dim, num_patches, pred_len, input_len, dropout=0.1):
        super(FrequencyAwareForecastingHead, self).__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        
        # Time path: patches -> prediction
        patch_flat_dim = hidden_dim * num_patches
        self.time_head = nn.Sequential(
            nn.LayerNorm(patch_flat_dim),
            nn.Dropout(dropout),
            nn.Linear(patch_flat_dim, pred_len)
        )
        
        # NEW: Frequency path (was completely missing before!)
        self.freq_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len)
        )
        
        # NEW: Cross-modal refinement
        self.cross_refine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len)
        )
        
        # NEW: Learnable combination weights
        self.combine_weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))
        
        # Linear trend bypass
        self.linear_trend = nn.Linear(input_len, pred_len)
        nn.init.normal_(self.linear_trend.weight, mean=0.0, std=0.01)
    
    def forward(self, h_time_seq, h_time, h_freq, x_flat):
        """
        FIX: Now properly uses h_freq which was ignored before!
        """
        # Time-based prediction
        h_flat = h_time_seq.reshape(h_time_seq.size(0), -1)
        pred_time = self.time_head(h_flat)
        
        # NEW: Frequency-based prediction (captures periodic components)
        pred_freq = self.freq_head(h_freq)
        
        # NEW: Cross-modal prediction
        h_cross = torch.cat([h_time, h_freq], dim=-1)
        pred_cross = self.cross_refine(h_cross)
        
        # Linear trend
        trend = self.linear_trend(x_flat)
        
        # NEW: Learnable weighted combination
        weights = F.softmax(self.combine_weights, dim=0)
        pred = (weights[0] * pred_time + 
                weights[1] * pred_freq + 
                weights[2] * pred_cross + 
                trend)
        
        return pred


# NEW: Spectral-Temporal Disentanglement (Novel for Publication)
class SpectralTemporalDisentanglement(nn.Module):
    """
    NEW: NOVEL CONTRIBUTION for publication.
    
    Learns disentangled representations where:
    - Time branch captures non-periodic, transient dynamics
    - Frequency branch captures periodic, stationary patterns
    - Adaptive routing decides which pathway handles each prediction step
    - MI minimization encourages specialization
    """
    def __init__(self, hidden_dim, num_patches, pred_len, dropout=0.1):
        super(SpectralTemporalDisentanglement, self).__init__()
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        
        # Domain-Specific Encoders
        self.time_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.freq_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Disentanglement projections
        self.time_disentangle = nn.Linear(hidden_dim, hidden_dim // 2)
        self.freq_disentangle = nn.Linear(hidden_dim, hidden_dim // 2)
        self.shared_disentangle = nn.Linear(hidden_dim * 2, hidden_dim // 2)
        
        # Adaptive Router
        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * 3)
        )
        
        # Domain-Specific Predictors
        self.time_predictor = nn.Linear(hidden_dim // 2, pred_len)
        self.freq_predictor = nn.Linear(hidden_dim // 2, pred_len)
        self.shared_predictor = nn.Linear(hidden_dim // 2, pred_len)
        
        # MI Minimization Discriminator
        self.mi_discriminator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, h_time, h_freq, return_routing=False):
        batch_size = h_time.size(0)
        
        # Encode domain-specific features
        z_time = self.time_encoder(h_time)
        z_freq = self.freq_encoder(h_freq)
        
        # Disentangle into specialized subspaces
        z_time_d = self.time_disentangle(z_time)
        z_freq_d = self.freq_disentangle(z_freq)
        z_shared = self.shared_disentangle(torch.cat([h_time, h_freq], dim=-1))
        
        # Generate predictions from each pathway
        pred_time = self.time_predictor(z_time_d)
        pred_freq = self.freq_predictor(z_freq_d)
        pred_shared = self.shared_predictor(z_shared)
        
        # Compute adaptive routing weights
        router_input = torch.cat([z_time, z_freq], dim=-1)
        routing_logits = self.router(router_input)
        routing_weights = routing_logits.view(batch_size, self.pred_len, 3)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Route predictions
        preds_stacked = torch.stack([pred_time, pred_freq, pred_shared], dim=-1)
        pred_final = (preds_stacked * routing_weights).sum(dim=-1)
        
        # Compute MI regularization loss
        mi_loss = torch.tensor(0.0, device=h_time.device)
        if self.training:
            mi_loss = self._compute_mi_loss(z_time_d, z_freq_d)
        
        if return_routing:
            return pred_final, mi_loss, routing_weights
        
        return pred_final, mi_loss
    
    def _compute_mi_loss(self, z_time, z_freq):
        """Minimize mutual information using MINE."""
        batch_size = z_time.size(0)
        
        joint = torch.cat([z_time, z_freq], dim=-1)
        joint_score = self.mi_discriminator(joint)
        
        idx = torch.randperm(batch_size, device=z_time.device)
        marginal = torch.cat([z_time, z_freq[idx]], dim=-1)
        marginal_score = self.mi_discriminator(marginal)
        
        mi_estimate = joint_score.mean() - torch.log(torch.exp(marginal_score).mean() + 1e-8)
        
        return F.relu(mi_estimate)


class LightweightModel(nn.Module):
    """
    MODIFIED: Updated LightweightModel with all improvements.
    Compatible with d_model=128 (TimeDART benchmark settings).
    
    Changes:
    - NEW: CrossModalFusion replaces LearnableFusion
    - NEW: DeepMixerStack replaces simple mixer layers  
    - NEW: MultiScaleTFCLoss replaces TFC_Loss for pretraining
    - FIX: FrequencyAwareForecastingHead now uses frequency features
    - NEW: Optional SpectralTemporalDisentanglement for publication
    - REMOVED: ForgettingMechanisms (was disabled)
    """
    def __init__(self, in_channels, out_channels, task_name="pretrain", pred_len=None, 
                 num_classes=2, input_len=336, use_noise=False, 
                 use_forgetting=False, forgetting_type="activation", forgetting_rate=0.1,  # REMOVED: kept args for compatibility
                 tfc_weight=0.05, tfc_warmup_steps=0, use_real_imag=False,
                 projection_dim=128, patch_len=16, stride=8, depth=4, hidden_dim=128,  # MODIFIED: depth=4, hidden_dim=128 default
                 use_disentanglement=False):  # NEW: flag for novel module
        super(LightweightModel, self).__init__()
        self.task_name = task_name
        self.pred_len = pred_len
        self.num_classes = num_classes
        self.input_len = input_len
        self.use_noise = use_noise
        # REMOVED: self.use_forgetting - was disabled anyway
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.in_channels = in_channels
        self.use_disentanglement = use_disentanglement  # NEW
        
        self.patch_len = patch_len
        self.stride = stride
        
        self.tfc_weight = tfc_weight
        self.tfc_warmup_steps = tfc_warmup_steps
        self.register_buffer('_step_counter', torch.tensor(0, dtype=torch.long))
        
        # RevIN
        self.revin = RevIN(in_channels)
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(input_len, self.patch_len, self.stride, self.hidden_dim)
        
        # Compute exact number of patches
        dummy_input = torch.zeros(1, input_len)
        dummy_patches = self.patch_embed(dummy_input)
        self.num_patches = dummy_patches.shape[1]
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.hidden_dim) * 0.02)

        # NEW: Replace simple mixer layers with DeepMixerStack
        self.mixer_stack = DeepMixerStack(self.num_patches, self.hidden_dim, depth=self.depth, dropout=0.1)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Frequency backbone
        self.freq_backbone = FrequencyEncoder(
            input_len=input_len,
            hidden_dim=self.hidden_dim,
            dropout=0.2,
            use_real_imag=use_real_imag
        )
        
        # Projection heads for TFC
        self.time_projection = ProjectionHead(self.hidden_dim, self.hidden_dim // 2, projection_dim)
        self.freq_projection = ProjectionHead(self.hidden_dim, self.hidden_dim // 2, projection_dim)
        
        # NEW: Multi-scale TFC loss replaces simple TFC_Loss
        self.tfc_loss_fn = MultiScaleTFCLoss(self.hidden_dim, self.num_patches, temperature=0.2)
        
        # NEW: CrossModalFusion replaces LearnableFusion
        # FIX: num_heads=4 works for d_model=128 (128/4=32 dim per head)
        self.fusion = CrossModalFusion(self.hidden_dim, num_heads=4, dropout=0.1)

        # REMOVED: Forgetting mechanisms - was disabled and hurt performance
        
        # Task Heads
        if self.task_name == "pretrain":
            self.decoder = nn.Linear(self.hidden_dim, input_len)
        
        elif self.task_name == "finetune" and self.pred_len is not None:
            # NEW: Choose between FrequencyAwareForecastingHead or SpectralTemporalDisentanglement
            if self.use_disentanglement:
                # NEW: Novel module for publication
                self.forecast_head = SpectralTemporalDisentanglement(
                    self.hidden_dim, self.num_patches, self.pred_len, dropout=0.1
                )
            else:
                # FIX: FrequencyAwareForecastingHead properly uses frequency features
                self.forecast_head = FrequencyAwareForecastingHead(
                    self.hidden_dim, self.num_patches, self.pred_len, self.input_len, dropout=0.1
                )
            
            # Linear trend bypass (kept from original)
            self.linear_trend = nn.Linear(self.input_len, self.pred_len)
            nn.init.normal_(self.linear_trend.weight, mean=0.0, std=0.01)
   
        else:  # Classification
            self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.dropout = nn.Dropout(0.2)
            self.global_pool_cls = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes)
            )
    
    def get_tfc_weight(self):
        if self.tfc_warmup_steps <= 0:
            return self.tfc_weight
        progress = min(1.0, self._step_counter.item() / self.tfc_warmup_steps)
        return self.tfc_weight * progress
    
    def increment_step(self):
        self._step_counter += 1

    def add_noise(self, x, noise_level=0.1):
        std_dev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        noise = torch.randn_like(x)
        scaled_noise = noise * std_dev * noise_level
        return x + scaled_noise, scaled_noise

    def random_masking(self, x, mask_ratio=0.4):
        B, L, D = x.shape
        x = x.clone()
        mask = torch.rand(B, L, device=x.device) < mask_ratio
        mask = mask.unsqueeze(-1).repeat(1, 1, D)
        x_masked = x * (~mask)
        return x_masked, mask

    def forward(self, x, add_noise_flag=False, noise_level=0.1):
        batch_size, seq_len, num_features = x.size()
        
        # RevIN normalization
        if self.task_name != 'classification':
            x = self.revin(x, 'norm')   
        
        # Input augmentation
        if self.task_name == "pretrain":
            if self.use_noise and add_noise_flag:
                x_input, _ = self.add_noise(x, noise_level=noise_level)
            else:
                x_input, _ = self.random_masking(x, mask_ratio=0.4)
        else:
            x_input = x
        
        # Channel independence: Flatten
        x_permuted = x_input.permute(0, 2, 1) 
        x_flat = x_permuted.reshape(batch_size * num_features, seq_len)
        
        group_ids = torch.arange(batch_size, device=x.device).repeat_interleave(num_features)
        
        # DUAL BACKBONE
        
        # A: Frequency path
        h_freq = self.freq_backbone(x_flat)
        
        # B: Time path - NEW: Using DeepMixerStack instead of simple mixer layers
        x_patches = self.patch_embed(x_flat)
        x_patches = x_patches + self.pos_embedding[:, :x_patches.size(1), :]
        
        # NEW: Deep mixer stack with skip connections
        h_time_seq = self.mixer_stack(x_patches, h_freq)
        
        # Global pooling
        h_time = self.global_pool(h_time_seq.transpose(1, 2)).squeeze(-1)
        
        # NEW: Cross-modal fusion with attention
        h_fused, gate_weights = self.fusion(h_time, h_freq, h_time_seq)

        # REMOVED: Forgetting mechanism - was disabled

        # TASK HEADS
        
        if self.task_name == "pretrain":
            # NEW: Multi-scale TFC loss
            z_time = self.time_projection(h_time)
            z_freq = self.freq_projection(h_freq)
            loss_tfc = self.tfc_loss_fn(h_time, h_freq, h_time_seq, group_ids=group_ids) * self.get_tfc_weight()
            
            if self.training:
                self.increment_step()
            
            reconstruction = self.decoder(h_fused)
            reconstruction = reconstruction.view(batch_size, num_features, seq_len).permute(0, 2, 1)
            reconstruction = self.revin(reconstruction, 'denorm')
            
            return reconstruction, loss_tfc

        elif self.task_name == "finetune" and self.pred_len is not None:
            # FIX: Now properly uses frequency features!
            if self.use_disentanglement:
                # NEW: Novel disentanglement module
                pred, mi_loss = self.forecast_head(h_time, h_freq)
                # Store MI loss for potential logging
                self._last_mi_loss = mi_loss
            else:
                # FIX: FrequencyAwareForecastingHead uses h_freq
                pred = self.forecast_head(h_time_seq, h_time, h_freq, x_flat)
            
            # Reshape
            pred = pred.reshape(batch_size, num_features, -1).permute(0, 2, 1)
            pred = self.revin(pred, 'denorm')
            
            return pred

        else:  # Classification
            cnn_input = h_fused.view(batch_size, num_features, -1)
            x = F.relu(self.conv1(cnn_input))
            x = self.pool(x)
            x = self.dropout(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = self.dropout(x)
            x = F.relu(self.conv3(x))
            x = self.global_pool_cls(x).squeeze(-1)
            x = self.classifier(x)
            return x


class Model(nn.Module):
    """
    Wrapper for LightweightModel. 
    MODIFIED: Added use_disentanglement flag and depth arg for novel module.
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.input_len = args.input_len
        self.pred_len = args.pred_len
        self.task_name = args.task_name
        self.use_norm = args.use_norm
        self.use_noise = getattr(args, 'use_noise', False)
        self.noise_level = getattr(args, 'noise_level', 0.1)
        
        # TFC params
        tfc_weight = getattr(args, 'tfc_weight', 0.05)
        tfc_warmup_steps = getattr(args, 'tfc_warmup_steps', 0)
        use_real_imag = getattr(args, 'use_real_imag', False)
        projection_dim = getattr(args, 'projection_dim', 128)
        
        # Patch params
        patch_len = getattr(args, 'patch_len', 16)
        stride = getattr(args, 'stride', 8)
        
        # NEW: depth arg for controlling mixer depth
        depth = getattr(args, 'depth', 4)  # NEW: default 4 layers
        hidden_dim = getattr(args, 'd_model', 128)  # Keep d_model=128 for benchmark compatibility
        
        # NEW: Disentanglement flag
        use_disentanglement = getattr(args, 'use_disentanglement', False)

        self.model = LightweightModel(
            in_channels=args.enc_in,
            out_channels=args.enc_in,
            task_name=self.task_name,
            pred_len=self.pred_len if hasattr(args, 'pred_len') else None,
            num_classes=getattr(args, 'num_classes', 2),
            input_len=args.input_len,
            use_noise=self.use_noise,
            use_forgetting=False,  # REMOVED: Always False now
            forgetting_type='activation',
            forgetting_rate=0.1,
            tfc_weight=tfc_weight,
            tfc_warmup_steps=tfc_warmup_steps,
            use_real_imag=use_real_imag,
            projection_dim=projection_dim,
            patch_len=patch_len,
            stride=stride,
            depth=depth,  # NEW: pass depth
            hidden_dim=hidden_dim,
            use_disentanglement=use_disentanglement  # NEW
        )

    def pretrain(self, x):
        batch_size, input_len, num_features = x.size()
        if self.use_norm:
            means = torch.mean(x, dim=1, keepdim=True).detach()
            x = x - means
            stdevs = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = x / stdevs

        predict_x, loss_tfc = self.model(x, add_noise_flag=True, noise_level=self.noise_level)

        if self.use_norm:
            predict_x = predict_x * stdevs
            predict_x = predict_x + means

        return predict_x, loss_tfc

    def forecast(self, x):
        batch_size, input_len, num_features = x.size()
        if self.use_norm:
            means = torch.mean(x, dim=1, keepdim=True).detach()
            x = x - means
            stdevs = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = x / stdevs

        prediction = self.model(x, add_noise_flag=False)

        if self.use_norm:
            prediction = prediction * stdevs[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            prediction = prediction + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return prediction
    
    def forward(self, batch_x):
        if self.task_name == "pretrain":
            return self.pretrain(batch_x)
        elif self.task_name == "finetune":
            return self.forecast(batch_x)
        else:
            raise ValueError("task_name should be 'pretrain' or 'finetune'")


class ClsModel(nn.Module):
    """
    Classification model wrapper.
    MODIFIED: Added use_disentanglement flag and depth arg.
    """
    def __init__(self, args):
        super(ClsModel, self).__init__()
        self.input_len = args.input_len
        self.task_name = args.task_name
        self.num_classes = args.num_classes
        self.use_noise = getattr(args, 'use_noise', False)
        self.noise_level = getattr(args, 'noise_level', 0.1)
        
        tfc_weight = getattr(args, 'tfc_weight', 0.05)
        tfc_warmup_steps = getattr(args, 'tfc_warmup_steps', 0)
        use_real_imag = getattr(args, 'use_real_imag', False)
        projection_dim = getattr(args, 'projection_dim', 128)
        patch_len = getattr(args, 'patch_len', 16)
        stride = getattr(args, 'stride', 8)
        depth = getattr(args, 'depth', 4)  # NEW
        hidden_dim = getattr(args, 'd_model', 128)
        use_disentanglement = getattr(args, 'use_disentanglement', False)  # NEW
        
        self.model = LightweightModel(
            in_channels=args.enc_in,
            out_channels=args.enc_in,
            task_name=self.task_name,
            pred_len=None,
            num_classes=self.num_classes,
            input_len=args.input_len,
            use_noise=self.use_noise,
            use_forgetting=False,
            forgetting_type='activation',
            forgetting_rate=0.1,
            tfc_weight=tfc_weight,
            tfc_warmup_steps=tfc_warmup_steps,
            use_real_imag=use_real_imag,
            projection_dim=projection_dim,
            patch_len=patch_len,
            stride=stride,
            depth=depth,  # NEW
            hidden_dim=hidden_dim,
            use_disentanglement=use_disentanglement  # NEW
        )

    def pretrain(self, x):
        return self.model(x, add_noise_flag=True, noise_level=self.noise_level)

    def forecast(self, x):
        return self.model(x, add_noise_flag=False)
    
    def forward(self, batch_x):
        if self.task_name == "pretrain":
            return self.pretrain(batch_x)
        elif self.task_name == "finetune":
            return self.forecast(batch_x)
        else:
            raise ValueError("task_name should be 'pretrain' or 'finetune'")


# ==============================================================================
# SUMMARY OF CHANGES
# ==============================================================================
"""
REMOVED Classes:
- SpectrallyGatedMixerLayer: Replaced by HierarchicalMixerBlock
- ForgettingMechanisms: Was disabled (use_forgetting=0), hurt performance
- LearnableFusion: Replaced by CrossModalFusion  
- ResidualForecastingHead: Replaced by FrequencyAwareForecastingHead

NEW Classes:
- HierarchicalMixerBlock: Multi-scale mixer with local/medium/global receptive fields
- DeepMixerStack: Deeper backbone with skip connections
- CrossModalFusion: Bidirectional cross-attention between time and frequency
- MultiScaleTFCLoss: Global + patch-level TFC for better SSL
- FrequencyAwareForecastingHead: FIX - properly uses frequency features
- SpectralTemporalDisentanglement: NOVEL module for publication (optional)

CRITICAL FIX:
- Original forecasting head (lines 555-577) ignored h_freq completely!
- FrequencyAwareForecastingHead now properly uses both h_time and h_freq

NEW Args (add to run.py):
- --depth: Number of mixer layers (default: 4)
- --use_disentanglement: Enable novel SpectralTemporalDisentanglement module (0 or 1)

BENCHMARK COMPATIBLE:
- d_model=128 (as required by TimeDART benchmark)
- All other settings remain the same
"""