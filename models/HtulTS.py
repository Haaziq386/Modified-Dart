import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SpectrallyGatedMixerLayer(nn.Module):
    """
    Novelty: A Mixer layer where the Token Mixing is gated by Global Frequency Context.
    
    """
    def __init__(self, num_patches, hidden_dim, dropout=0.1):
        super(SpectrallyGatedMixerLayer, self).__init__()
        
        # 1. Token Mixing (Time interactions)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.token_mix = nn.Sequential(
            nn.Linear(num_patches, num_patches),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Channel Mixing (Feature interactions)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 3. Spectral Gate (The Novelty)
        # Projects global freq context [Batch, Hidden] -> [Batch, 1, Hidden]
        self.freq_gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, h_freq):
        # x: [Batch, Patches, Hidden]
        # h_freq: [Batch, Hidden]
        
        # --- Token Mixing Block ---
        # Swap axes for mixing across patches: [B, H, P]
        # This allows the model to "see" the whole sequence at once
        y = self.norm1(x).transpose(1, 2)
        y = self.token_mix(y).transpose(1, 2)
        
        # --- Spectral Gating ---
        # Inject frequency info to scale the time features dynamically
        gate = 1 + torch.tanh(self.freq_gate(h_freq).unsqueeze(1))
        y = y * gate
        
        x = x + y # Residual 1
        
        # --- Channel Mixing Block ---
        y = self.norm2(x)
        y = self.channel_mix(y)
        x = x + y # Residual 2
        
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
        
        # Calculate number of patches
        # Formula: (L - P) / S + 1
        # We handle padding dynamically in forward
        self.num_patches = int((seq_len - patch_len) / stride) + 2
        
        # Projection: Maps a single patch (size patch_len) to hidden_dim
        self.value_embedding = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch * Features, Seq_Len]
        n_vars = x.shape[1]
        
        # 1. Padding to ensure we cover the whole sequence
        if self.stride > 0:
            padding = self.stride - ((n_vars - self.patch_len) % self.stride)
            if padding < self.stride:
                 x = F.pad(x, (0, padding), "replicate")
        
        # 2. Patching (Unfold)
        # Output: [Batch*Feat, Num_Patches, Patch_Len]
        x_patched = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        
        # 3. Projection
        # Output: [Batch*Feat, Num_Patches, d_model]
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


class ForgettingMechanisms(nn.Module):
    """
    Forgetting mechanisms (Adaptive, Weight, Activation).
    """
    def __init__(self, hidden_dim, forgetting_type="activation", forgetting_rate=0.1):
        super(ForgettingMechanisms, self).__init__()
        self.forgetting_type = forgetting_type
        self.forgetting_rate = forgetting_rate
        self.hidden_dim = hidden_dim
        
        if forgetting_type == "activation":
            self.forget_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.Sigmoid()
            )
        elif forgetting_type == "weight":
            self.importance_scores = nn.Parameter(torch.ones(hidden_dim))
            self.decay_factor = nn.Parameter(torch.tensor(forgetting_rate))
        elif forgetting_type == "adaptive":
            self.forget_threshold = nn.Parameter(torch.tensor(0.5))
            self.forget_scale = nn.Parameter(torch.tensor(forgetting_rate))
    
    def forward(self, x, is_finetune=False):
        if not is_finetune:
            return x
            
        if self.forgetting_type == "activation":
            forget_mask = self.forget_gate(x)
            forget_factor = 1.0 - self.forgetting_rate * forget_mask
            return x * forget_factor
            
        elif self.forgetting_type == "weight":
            importance = torch.sigmoid(self.importance_scores)
            decay = torch.sigmoid(self.decay_factor)
            forget_factor = importance * (1.0 - decay)
            return x * forget_factor
            
        elif self.forgetting_type == "adaptive":
            activation_magnitude = torch.mean(torch.abs(x), dim=-1, keepdim=True)
            forget_prob = torch.sigmoid(self.forget_scale * (activation_magnitude - self.forget_threshold))
            if self.training:
                forget_mask = (torch.rand_like(forget_prob) < forget_prob).float()
            else:
                forget_mask = (forget_prob > 0.5).float()
            return x * (1.0 - forget_mask * self.forgetting_rate)
        
        return x


class LearnableFusion(nn.Module):
    def __init__(self, hidden_dim):
        super(LearnableFusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, h_time, h_freq):
        h_concat = torch.cat([h_time, h_freq], dim=-1)
        h_fused = self.fusion(h_concat)
        return h_fused


class ResidualForecastingHead(nn.Module):
    def __init__(self, hidden_dim, pred_len, dropout=0.1):
        super(ResidualForecastingHead, self).__init__()
        self.pred_len = pred_len
        head_width = min(2048, max(256, 4 * pred_len))
        
        self.fc1 = nn.Linear(hidden_dim, head_width)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(head_width)
        
        self.fc2 = nn.Linear(head_width, pred_len)
        self.dropout2 = nn.Dropout(dropout)
        
        if hidden_dim != pred_len:
            self.residual_proj = nn.Linear(hidden_dim, pred_len)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x):
        residual = self.residual_proj(x)
        out = self.fc1(x)
        out = F.relu(out)
        out = self.norm1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        out = out + residual
        return out


class LightweightModel(nn.Module):
    """
    Updated LightweightModel with:
    - RevIN for normalization
    - Patching + SpectrallyGatedMixer Stack (Time)
    - Global FFT Backbone (Freq)
    - Masking for Pretraining
    - Learnable Fusion & TFC
    """
    def __init__(self, in_channels, out_channels, task_name="pretrain", pred_len=None, 
                 num_classes=2, input_len=336, use_noise=False, 
                 use_forgetting=False, forgetting_type="activation", forgetting_rate=0.1,
                 tfc_weight=0.05, tfc_warmup_steps=0, use_real_imag=False,
                 projection_dim=128, patch_len=16, stride=8, depth=3, hidden_dim=256):
        super(LightweightModel, self).__init__()
        self.task_name = task_name
        self.pred_len = pred_len
        self.num_classes = num_classes
        self.input_len = input_len
        self.use_noise = use_noise
        self.use_forgetting = use_forgetting
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.in_channels = in_channels
        
        # Patch Config
        self.patch_len = patch_len
        self.stride = stride
        
        # TF-C weighting configuration
        self.tfc_weight = tfc_weight
        self.tfc_warmup_steps = tfc_warmup_steps
        self.register_buffer('_step_counter', torch.tensor(0, dtype=torch.long))
        
        # RevIN for normalization (Crucial for SOTA)
        self.revin = RevIN(in_channels)
        
        # 1. TIME-DOMAIN BACKBONE: Patch Embedding + Stack of Spectrally-Gated Mixer Layers
        self.patch_embed = PatchEmbedding(input_len, self.patch_len, self.stride, self.hidden_dim)
        
        # Compute exact number of patches
        dummy_input = torch.zeros(1, input_len)
        dummy_patches = self.patch_embed(dummy_input) # [1, Num_Patches, Hidden]
        self.num_patches = dummy_patches.shape[1]
        
        # Essential for Mixers to understand "order"
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.hidden_dim) * 0.02)

        # Stack of Spectrally-Gated Mixer Layers
        self.mixer_layers = nn.ModuleList([
            SpectrallyGatedMixerLayer(self.num_patches, self.hidden_dim, dropout=0.1)
            for _ in range(self.depth)
        ])
        
        # Global pooling for aggregating patches to single time representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 2. FREQUENCY-DOMAIN BACKBONE (Global)
        self.freq_backbone = FrequencyEncoder(
            input_len=input_len,
            hidden_dim=self.hidden_dim,
            dropout=0.2,
            use_real_imag=use_real_imag
        )
        
        # Projection heads for TF-C contrastive loss
        self.time_projection = ProjectionHead(self.hidden_dim, self.hidden_dim // 2, projection_dim)
        self.freq_projection = ProjectionHead(self.hidden_dim, self.hidden_dim // 2, projection_dim)
        
        self.tfc_loss_fn = TFC_Loss(temperature=0.2)
        self.fusion = LearnableFusion(self.hidden_dim)

        # Forgetting mechanisms
        if self.use_forgetting:
            self.forgetting_layer = ForgettingMechanisms(
                self.hidden_dim, forgetting_type, forgetting_rate
            )
        
        # Task Heads
        if self.task_name == "pretrain":
            # Decoder must map hidden_dim back to full sequence length
            self.decoder = nn.Linear(self.hidden_dim, input_len)
        
        elif self.task_name == "finetune" and self.pred_len is not None:
            # SOTA Head: Flatten-LayerNorm-Dropout-Linear (from PatchTST)
            # LayerNorm is batch-size invariant (works with batch size 1 or 16)
            self.head_ln = nn.LayerNorm(self.hidden_dim * self.num_patches)
            self.head_dropout = nn.Dropout(0.1)
            self.head_proj = nn.Linear(self.hidden_dim * self.num_patches, self.pred_len)
            # Linear trend branch to capture easy AR-style trends directly
            self.linear_trend = nn.Linear(self.input_len, self.pred_len)
            nn.init.normal_(self.linear_trend.weight, mean=0.0, std=0.01)
   
        else: # Classification
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
        """Randomly mask the input sequence."""
        B, L, D = x.shape
        x = x.clone()
        # Create mask: 1 = masked (dropped), 0 = kept
        mask = torch.rand(B, L, device=x.device) < mask_ratio
        mask = mask.unsqueeze(-1).repeat(1, 1, D)
        x_masked = x * (~mask) # Zero out masked values
        return x_masked, mask

    def forward(self, x, add_noise_flag=False, noise_level=0.1):
        batch_size, seq_len, num_features = x.size()
        
        # 1. REVERSIBLE INSTANCE NORMALIZATION (Crucial for SOTA)
        # ---------------------------------------------------
        # Do NOT normalize for classification tasks
        if self.task_name != 'classification':
            x = self.revin(x, 'norm')   
        
        # 2. INPUT AUGMENTATION (Masking vs Noise)
        # ---------------------------------------------------
        if self.task_name == "pretrain":
            # Default to Masking for SOTA results
            # Only use Noise if explicitly requested via args AND flag
            if self.use_noise and add_noise_flag:
                x_input, _ = self.add_noise(x, noise_level=noise_level)
            else:
                x_input, _ = self.random_masking(x, mask_ratio=0.4)
        else:
            x_input = x
        
        # 3. CHANNEL INDEPENDENCE: Flatten Batch and Features
        # ---------------------------------------------------
        # [B, L, C] -> [B, C, L] -> [B*C, L]
        x_permuted = x_input.permute(0, 2, 1) 
        x_flat = x_permuted.reshape(batch_size * num_features, seq_len)
        
        # Group IDs for TFC (used to group samples by original batch)
        group_ids = torch.arange(batch_size, device=x.device).repeat_interleave(num_features)
        
        # 4. DUAL BACKBONE
        # ---------------------------------------------------
        
        # A: FREQUENCY PATH (Global context)
        # [B*C, L] -> [B*C, Hidden]
        h_freq = self.freq_backbone(x_flat)
        
        # B: TIME PATH (Patching + Stack of Spectrally-Gated Mixer Layers)
        # 1. Patchify: [B*C, L] -> [B*C, Num_Patches, Hidden]
        x_patches = self.patch_embed(x_flat)
        
        # 2. Stack of Spectrally-Gated Mixer Layers with Frequency Gating
        # Pass frequency context to modulate each mixer layer
        h_time_seq = x_patches  # [B*C, Patches, Hidden]
        for mixer_layer in self.mixer_layers:
            # Each layer receives frequency context to gate token mixing
            h_time_seq = mixer_layer(h_time_seq, h_freq)
        
        # 3. Global Pooling to aggregate patches into single time representation
        # [B*C, Patches, Hidden] -> [B*C, Hidden, Patches] -> [B*C, Hidden, 1] -> [B*C, Hidden]
        h_time = self.global_pool(h_time_seq.transpose(1, 2)).squeeze(-1)
        
        # 5. FUSION & FORGETTING
        # ---------------------------------------------------
        h_fused = self.fusion(h_time, h_freq)

        if self.use_forgetting:
            is_finetune = (self.task_name == "finetune")
            h_fused = self.forgetting_layer(h_fused, is_finetune=is_finetune)

        # 6. TASK HEADS
        # ---------------------------------------------------
        
        # PRETRAIN
        if self.task_name == "pretrain":
            # Contrastive Loss between Time and Frequency representations
            z_time = self.time_projection(h_time)
            z_freq = self.freq_projection(h_freq)
            loss_tfc = self.tfc_loss_fn(z_time, z_freq, group_ids=group_ids) * self.get_tfc_weight()
            
            if self.training:
                self.increment_step()
            
            # Reconstruction: map fused representation back to sequence length
            # [B*C, Hidden] -> [B*C, L]
            reconstruction = self.decoder(h_fused)
            # Reshape: [B*C, L] -> [B, C, L] -> [B, L, C]
            reconstruction = reconstruction.view(batch_size, num_features, seq_len).permute(0, 2, 1)
            
            # Denormalize with RevIN
            reconstruction = self.revin(reconstruction, 'denorm')
            
            return reconstruction, loss_tfc

        # FINETUNE (Forecasting)
        elif self.task_name == "finetune" and self.pred_len is not None:
            # SOTA Head: Flatten-LayerNorm-Dropout-Linear
            # Use the sequence of patches from the mixer layers
            h_seq = h_time_seq  # [B*C, Patches, Hidden]
            # Flatten: [B*C, Patches * Hidden]
            x_out = h_seq.reshape(batch_size * num_features, -1)
            # Apply LayerNorm (BS-invariant, works with batch size 1 or 16)
            x_out = self.head_ln(x_out)
            # Apply Dropout
            x_out = self.head_dropout(x_out)
            # Project to prediction length
            pred = self.head_proj(x_out)  # [B*C, Pred_Len]
            # Linear trend bypass to capture simple trends without stressing the mixer
            trend_pred = self.linear_trend(x_flat)  # [B*C, Pred_Len]
            pred = pred + trend_pred
            # Reshape: [B*C, Pred] -> [B, C, Pred] -> [B, Pred, C]
            pred = pred.reshape(batch_size, num_features, -1).permute(0, 2, 1)
            
            # Denormalize with RevIN
            pred = self.revin(pred, 'denorm')
            
            return pred

        # CLASSIFICATION
        else:
            # Reshape fused representation for CNN: [B*C, Hidden] -> [B, C, Hidden]
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
    Handles Arg parsing and normalization.
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.input_len = args.input_len
        self.pred_len = args.pred_len
        self.task_name = args.task_name
        self.use_norm = args.use_norm
        self.use_noise = getattr(args, 'use_noise', False)
        self.noise_level = getattr(args, 'noise_level', 0.1)
        
        # Forgetting params
        self.use_forgetting = getattr(args, 'use_forgetting', False)
        self.forgetting_type = getattr(args, 'forgetting_type', 'activation')
        self.forgetting_rate = getattr(args, 'forgetting_rate', 0.1)
        
        # TFC params
        tfc_weight = getattr(args, 'tfc_weight', 0.05)
        tfc_warmup_steps = getattr(args, 'tfc_warmup_steps', 0)
        use_real_imag = getattr(args, 'use_real_imag', False)
        projection_dim = getattr(args, 'projection_dim', 128)
        
        # Patch params (Standard for ETTh1: P=16, S=8)
        patch_len = getattr(args, 'patch_len', 16)
        stride = getattr(args, 'stride', 8)
        
        # Mixer architecture params
        depth = getattr(args, 'depth', 3)  # Number of mixer layers
        hidden_dim = getattr(args, 'd_model', 256)  # Use d_model as hidden_dim

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
            depth=depth,
            hidden_dim=hidden_dim
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
    def __init__(self, args):
        super(ClsModel, self).__init__()
        self.input_len = args.input_len
        self.task_name = args.task_name
        self.num_classes = args.num_classes
        self.use_noise = getattr(args, 'use_noise', False)
        self.noise_level = getattr(args, 'noise_level', 0.1)
        self.use_forgetting = getattr(args, 'use_forgetting', False)
        
        # Params passed same as Model
        tfc_weight = getattr(args, 'tfc_weight', 0.05)
        tfc_warmup_steps = getattr(args, 'tfc_warmup_steps', 0)
        use_real_imag = getattr(args, 'use_real_imag', False)
        projection_dim = getattr(args, 'projection_dim', 128)
        patch_len = getattr(args, 'patch_len', 16)
        stride = getattr(args, 'stride', 8)
        depth = getattr(args, 'depth', 3)  # Number of mixer layers
        hidden_dim = getattr(args, 'd_model', 256)  # Use d_model as hidden_dim
        
        self.model = LightweightModel(
            in_channels=args.enc_in,
            out_channels=args.enc_in,
            task_name=self.task_name,
            pred_len=None,
            num_classes=self.num_classes,
            input_len=args.input_len,
            use_noise=self.use_noise,
            use_forgetting=self.use_forgetting,
            forgetting_type=getattr(args, 'forgetting_type', 'activation'),
            forgetting_rate=getattr(args, 'forgetting_rate', 0.1),
            tfc_weight=tfc_weight,
            tfc_warmup_steps=tfc_warmup_steps,
            use_real_imag=use_real_imag,
            projection_dim=projection_dim,
            patch_len=patch_len,
            stride=stride,
            depth=depth,
            hidden_dim=hidden_dim
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