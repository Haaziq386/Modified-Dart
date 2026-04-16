import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveFrequencyWarping(nn.Module):
    """
    Lightweight learnable monotonic frequency warping.

    Input/Output shapes:
    - [N, F, D] -> [N, F, D]
    - [N, F] -> [N, F]
    """
    def __init__(self, num_bins, eps=1e-6):
        super(AdaptiveFrequencyWarping, self).__init__()
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2")

        self.num_bins = int(num_bins)
        self.eps = float(eps)

        init_val = 1.0 / (self.num_bins - 1)
        inv_sp = math.log(math.exp(init_val) - 1.0)
        self.raw_increments = nn.Parameter(
            torch.full((self.num_bins - 1,), inv_sp, dtype=torch.float)
        )

    def _compute_grid(self):
        increments = F.softplus(self.raw_increments)
        zero = increments.new_zeros(1)
        cum = torch.cat([zero, torch.cumsum(increments, dim=0)], dim=0)
        total = cum[-1].clamp(min=self.eps)
        grid = (cum / total).clamp(0.0, 1.0)
        return grid

    def forward(self, freq_features):
        orig_ndim = freq_features.dim()
        if orig_ndim == 2:
            freq = freq_features.unsqueeze(-1)
        elif orig_ndim == 3:
            freq = freq_features
        else:
            raise ValueError("freq_features must be 2D or 3D tensor")

        _, f_bins, _ = freq.shape
        if f_bins != self.num_bins:
            raise ValueError(f"Expected {self.num_bins} frequency bins, got {f_bins}")

        grid = self._compute_grid()
        scaled = (grid * (f_bins - 1)).clamp(0.0, f_bins - 1.0)

        left_idx = torch.floor(scaled).long()
        right_idx = (left_idx + 1).clamp(max=f_bins - 1)
        weight = (scaled - left_idx.float()).unsqueeze(0).unsqueeze(-1)

        left_vals = torch.index_select(freq, dim=1, index=left_idx)
        right_vals = torch.index_select(freq, dim=1, index=right_idx)
        warped = (1.0 - weight) * left_vals + weight * right_vals

        if orig_ndim == 2:
            return warped.squeeze(-1)
        return warped


class FrequencyEncoder(nn.Module):
    """
    Frequency-domain encoder using FFT with improved stability.
    Takes time-domain input and produces frequency embeddings.
    """
    def __init__(
        self,
        input_len,
        hidden_dim=512,
        dropout=0.2,
        use_real_imag=False,
        use_warping=False,
    ):
        super(FrequencyEncoder, self).__init__()
        self.input_len = input_len
        self.hidden_dim = hidden_dim
        self.use_real_imag = use_real_imag
        self.use_warping = use_warping
        
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

        if self.use_warping:
            self.warper = AdaptiveFrequencyWarping(self.num_freq_bins)
    
    def forward(self, x, return_bins: bool = False):
        # x: [N, L] time-domain signal
        x_windowed = x * self.hann_window
        spectrum = torch.fft.rfft(x_windowed, dim=-1)
        
        if self.use_real_imag:
            real_part = spectrum.real
            imag_part = spectrum.imag
            freq_features = torch.stack([real_part, imag_part], dim=-1)
        else:
            amplitude = torch.log1p(torch.abs(spectrum))
            phase = torch.angle(spectrum)
            sin_phase = torch.sin(phase)
            cos_phase = torch.cos(phase)
            freq_features = torch.stack([amplitude, sin_phase, cos_phase], dim=-1)

        if self.use_warping:
            freq_features = self.warper(freq_features)
        
        # Per-sample normalization across frequency bins and channels
        mean = freq_features.mean(dim=(1, 2), keepdim=True)
        std = freq_features.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        freq_features = (freq_features - mean) / std

        freq_flat = freq_features.reshape(freq_features.size(0), -1)
        freq_flat = torch.nan_to_num(freq_flat, nan=0.0, posinf=1e6, neginf=-1e6)
        
        h_freq = self.freq_projection(freq_flat)
        if return_bins:
            return h_freq, freq_features
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
        seq_len = x.shape[1]
        
        # 1. Padding to ensure we cover the whole sequence
        if self.stride > 0:
            padding = self.stride - ((seq_len - self.patch_len) % self.stride)
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
    Time-Frequency Consistency Loss using INFONCE.
    """
    def __init__(self, temperature=0.2):
        super(TFC_Loss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z_time, z_freq):
        batch_size = z_time.size(0)
        device = z_time.device
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        z_time = F.normalize(z_time, dim=-1)
        z_freq = F.normalize(z_freq, dim=-1)
        
        sim_tf = torch.matmul(z_time, z_freq.T) / self.temperature
        
        labels = torch.arange(batch_size, device=device)
        loss_tf = F.cross_entropy(sim_tf, labels)
        loss_ft = F.cross_entropy(sim_tf.T, labels)
        
        loss = (loss_tf + loss_ft) / 2.0
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss


class CPCTFLoss(nn.Module):
    """
    Cross-modal predictive coding between time and frequency branches.
    """

    def __init__(
        self,
        freq_mask_ratio=0.2,
        time_mask_ratio=0.2,
        lambda_cpc=0.1,
        loss_type="l2",
        pos_emb_dim=64,
        hidden_dim=256,
        eps=1e-6,
    ):
        super(CPCTFLoss, self).__init__()
        self.freq_mask_ratio = float(freq_mask_ratio)
        self.time_mask_ratio = float(time_mask_ratio)
        self.lambda_cpc = float(lambda_cpc)
        self.loss_type = loss_type
        self.pos_emb_dim = int(pos_emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.eps = float(eps)

        self._built = False
        self.freq_predictor = None
        self.time_predictor = None
        self.freq_pos_mlp = None
        self.time_pos_mlp = None

    def _build(self, z_time_dim, z_freq_dim, feat_dim_freq, feat_dim_time):
        self.freq_pos_mlp = nn.Sequential(
            nn.Linear(1, self.pos_emb_dim),
            nn.ReLU(),
            nn.Linear(self.pos_emb_dim, self.pos_emb_dim),
        )
        self.time_pos_mlp = nn.Sequential(
            nn.Linear(1, self.pos_emb_dim),
            nn.ReLU(),
            nn.Linear(self.pos_emb_dim, self.pos_emb_dim),
        )

        in_dim_f = z_time_dim + self.pos_emb_dim
        hid_f = min(self.hidden_dim, max(in_dim_f, 64))
        self.freq_predictor = nn.Sequential(
            nn.Linear(in_dim_f, hid_f),
            nn.ReLU(),
            nn.Linear(hid_f, feat_dim_freq),
        )

        in_dim_t = z_freq_dim + self.pos_emb_dim
        hid_t = min(self.hidden_dim, max(in_dim_t, 64))
        self.time_predictor = nn.Sequential(
            nn.Linear(in_dim_t, hid_t),
            nn.ReLU(),
            nn.Linear(hid_t, feat_dim_time),
        )

        self._built = True

    def _sample_band_mask(self, bsz, num_bins, device):
        mask = torch.zeros(bsz, num_bins, dtype=torch.bool, device=device)
        mask_len = max(1, int(round(self.freq_mask_ratio * num_bins)))
        for b in range(bsz):
            if mask_len >= num_bins:
                start = 0
            else:
                start = torch.randint(0, num_bins - mask_len + 1, (1,), device=device).item()
            mask[b, start:start + mask_len] = True
        return mask

    def _sample_patch_mask(self, bsz, num_patches, device):
        mask = torch.zeros(bsz, num_patches, dtype=torch.bool, device=device)
        mask_count = max(1, int(round(self.time_mask_ratio * num_patches)))
        for b in range(bsz):
            if mask_count >= num_patches:
                idx = torch.arange(num_patches, device=device)
            else:
                idx = torch.randperm(num_patches, device=device)[:mask_count]
            mask[b, idx] = True
        return mask

    def forward(self, z_time, z_freq, h_time_patches, h_freq_bins, tfc_loss=None):
        if h_time_patches.dim() != 3 or h_freq_bins.dim() != 3:
            raise ValueError("h_time_patches must be [B, N, D] and h_freq_bins [B, F, D]")

        bsz = z_time.size(0)
        device = z_time.device
        num_patches = h_time_patches.size(1)
        num_bins = h_freq_bins.size(1)
        feat_dim_time = h_time_patches.size(2)
        feat_dim_freq = h_freq_bins.size(2)

        if not self._built:
            self._build(
                z_time_dim=z_time.size(1),
                z_freq_dim=z_freq.size(1),
                feat_dim_freq=feat_dim_freq,
                feat_dim_time=feat_dim_time,
            )
            self.to(device)

        mask_freq = self._sample_band_mask(bsz, num_bins, device=device)
        mask_time = self._sample_patch_mask(bsz, num_patches, device=device)

        pos_f = torch.linspace(0.0, 1.0, steps=num_bins, device=device).view(1, num_bins, 1)
        pos_f_emb = self.freq_pos_mlp(pos_f).expand(bsz, -1, -1)
        zt = z_time.unsqueeze(1).expand(-1, num_bins, -1)
        pred_f = self.freq_predictor(torch.cat([zt, pos_f_emb], dim=-1).reshape(bsz * num_bins, -1))
        pred_f = pred_f.reshape(bsz, num_bins, feat_dim_freq)

        pos_t = torch.linspace(0.0, 1.0, steps=num_patches, device=device).view(1, num_patches, 1)
        pos_t_emb = self.time_pos_mlp(pos_t).expand(bsz, -1, -1)
        zf = z_freq.unsqueeze(1).expand(-1, num_patches, -1)
        pred_t = self.time_predictor(torch.cat([zf, pos_t_emb], dim=-1).reshape(bsz * num_patches, -1))
        pred_t = pred_t.reshape(bsz, num_patches, feat_dim_time)

        loss_time_to_freq = torch.tensor(0.0, device=device)
        loss_freq_to_time = torch.tensor(0.0, device=device)

        total_masked_freq = int(mask_freq.sum().item())
        if total_masked_freq > 0:
            m = mask_freq.unsqueeze(-1).expand(-1, -1, feat_dim_freq)
            target = h_freq_bins[m].reshape(-1, feat_dim_freq)
            pred = pred_f[m].reshape(-1, feat_dim_freq)
            if self.loss_type == "l2":
                loss_val = F.mse_loss(pred, target, reduction="sum")
            else:
                loss_val = F.smooth_l1_loss(pred, target, reduction="sum")
            loss_time_to_freq = loss_val / (max(1, pred.numel()) + self.eps)

        total_masked_time = int(mask_time.sum().item())
        if total_masked_time > 0:
            m_t = mask_time.unsqueeze(-1).expand(-1, -1, feat_dim_time)
            target_t = h_time_patches[m_t].reshape(-1, feat_dim_time)
            pred_t_masked = pred_t[m_t].reshape(-1, feat_dim_time)
            if self.loss_type == "l2":
                loss_val = F.mse_loss(pred_t_masked, target_t, reduction="sum")
            else:
                loss_val = F.smooth_l1_loss(pred_t_masked, target_t, reduction="sum")
            loss_freq_to_time = loss_val / (max(1, pred_t_masked.numel()) + self.eps)

        loss_time_to_freq = torch.nan_to_num(loss_time_to_freq, nan=0.0, posinf=1e6, neginf=-1e6)
        loss_freq_to_time = torch.nan_to_num(loss_freq_to_time, nan=0.0, posinf=1e6, neginf=-1e6)

        tfc_loss_val = tfc_loss if tfc_loss is not None else torch.tensor(0.0, device=device)
        total = tfc_loss_val + self.lambda_cpc * (loss_time_to_freq + loss_freq_to_time)
        total = torch.nan_to_num(total, nan=0.0, posinf=1e6, neginf=-1e6)

        return {
            "loss_tfc": tfc_loss_val,
            "loss_time_to_freq": loss_time_to_freq,
            "loss_freq_to_time": loss_freq_to_time,
            "loss_total": total,
        }


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
    - Patching + Mixer Backbone (Time)
    - Global FFT Backbone (Freq)
    - Masking for Pretraining
    - Learnable Fusion & TFC
    """
    def __init__(self, in_channels, out_channels, task_name="pretrain", pred_len=None, 
                 num_classes=2, input_len=336, use_noise=False, 
                 use_forgetting=False, forgetting_type="activation", forgetting_rate=0.1,
                 tfc_weight=0.05, tfc_warmup_steps=0, use_real_imag=False,
                 use_warping=False, projection_dim=128, patch_len=16, stride=8,
                 hidden_dim=512, use_cpc=False, cpc_kwargs=None):
        super(LightweightModel, self).__init__()
        self.task_name = task_name
        self.pred_len = pred_len
        self.num_classes = num_classes
        self.input_len = input_len
        self.use_noise = use_noise
        self.use_forgetting = use_forgetting
        self.use_warping = use_warping
        self.hidden_dim = hidden_dim
        
        # Patch Config
        self.patch_len = patch_len
        self.stride = stride
        
        # TF-C weighting configuration
        self.tfc_weight = tfc_weight
        self.tfc_warmup_steps = tfc_warmup_steps
        self.register_buffer('_step_counter', torch.tensor(0, dtype=torch.long))
        
        # 1. TIME-DOMAIN BACKBONE: Patch Embedding + Mixer Aggregation
        self.patch_embed = PatchEmbedding(input_len, self.patch_len, self.stride, self.hidden_dim)
        
        # Calculate expected number of patches to size the Mixer correctly
        # Note: PatchEmbedding calculates exact patches dynamically, but the linear layer 
        # needs a fixed input size. We assume input_len is constant.
        # Max patches logic: ceil((L-P)/S) + 1 approx. 
        # We run a dummy pass to get exact dimension or calculate carefully.
        # Calculation: (336 - 16)/8 + 2 (padding) approx 42 patches.
        # To be safe, we can use a small adaptive pool or dynamic sizing if input varies, 
        # but for fixed input_len, we calculate:
        dummy_input = torch.zeros(1, input_len)
        dummy_patches = self.patch_embed(dummy_input) # [1, Num_Patches, Hidden]
        self.num_patches = dummy_patches.shape[1]
        
        # MLP Mixer to aggregate patches into one global time vector
        self.patch_mixer = nn.Sequential(
            nn.Linear(self.num_patches * self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 2. FREQUENCY-DOMAIN BACKBONE (Global)
        self.freq_backbone = FrequencyEncoder(
            input_len=input_len,
            hidden_dim=self.hidden_dim,
            dropout=0.2,
            use_real_imag=use_real_imag,
            use_warping=self.use_warping,
        )
        
        # Projection heads for TF-C contrastive loss
        self.time_projection = ProjectionHead(self.hidden_dim, self.hidden_dim // 2, projection_dim)
        self.freq_projection = ProjectionHead(self.hidden_dim, self.hidden_dim // 2, projection_dim)
        
        self.tfc_loss_fn = TFC_Loss(temperature=0.2)
        self.fusion = LearnableFusion(self.hidden_dim)

        self.use_cpc = use_cpc
        if self.use_cpc:
            cpc_kwargs = cpc_kwargs or {}
            self.cpctf_loss_fn = CPCTFLoss(**cpc_kwargs)
        else:
            self.cpctf_loss_fn = None
        self.last_loss_dict = None

        # Forgetting
        if self.use_forgetting:
            self.forgetting_layer = ForgettingMechanisms(
                self.hidden_dim, forgetting_type, forgetting_rate
            )
        
        # Task Heads
        if self.task_name == "pretrain":
            # Decoder must map hidden_dim back to full sequence length
            self.decoder = nn.Linear(self.hidden_dim, input_len)
        
        elif self.task_name == "finetune" and self.pred_len is not None:
            self.forecaster = ResidualForecastingHead(self.hidden_dim, self.pred_len, dropout=0.1)
   
        else: # Classification
            self.conv1 = nn.Conv1d(self.hidden_dim, 64, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.dropout = nn.Dropout(0.2)
            self.global_pool = nn.AdaptiveAvgPool1d(1)
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
        
        # 1. INPUT AUGMENTATION (Masking vs Noise)
        # ---------------------------------------------------
        if self.task_name == "pretrain" and self.training:
            # Pretraining augmentation: noise or masking
            if self.use_noise and add_noise_flag:
                x_input, _ = self.add_noise(x, noise_level=noise_level)
            else:
                x_input, _ = self.random_masking(x, mask_ratio=0.4)
        else:
            x_input = x
            
        # Flatten: [B, L, C] -> [B*C, L]
        x_permuted = x_input.permute(0, 2, 1) 
        x_flat = x_permuted.reshape(batch_size * num_features, seq_len)
        
        # 2. DUAL BACKBONE
        # ---------------------------------------------------
        
        # A: Time Path (Patching + Mixing)
        # 1. Patchify: [B*C, L] -> [B*C, Num_Patches, Hidden]
        x_patches = self.patch_embed(x_flat) 
        # 2. Flatten patches for Mixer: [B*C, Num_Patches * Hidden]
        x_patches_flat = x_patches.view(x_patches.size(0), -1)
        # 3. Mixer Aggregation -> [B*C, Hidden]
        h_time = self.patch_mixer(x_patches_flat)
        
        # B: Frequency Path (Global) -> [B*C, Hidden]
        h_freq, h_freq_bins = self.freq_backbone(x_flat, return_bins=True)
        
        # 3. FUSION & FORGETTING
        # ---------------------------------------------------
        h_fused = self.fusion(h_time, h_freq)

        if self.use_forgetting:
            is_finetune = (self.task_name == "finetune")
            h_fused = self.forgetting_layer(h_fused, is_finetune=is_finetune)

        # 4. TASK HEADS
        # ---------------------------------------------------
        
        # PRETRAIN
        if self.task_name == "pretrain":
            z_time = self.time_projection(h_time)
            z_freq = self.freq_projection(h_freq)
            loss_tfc = self.tfc_loss_fn(z_time, z_freq) * self.get_tfc_weight()
            
            if self.training:
                self.increment_step()

            if self.use_cpc and self.cpctf_loss_fn is not None:
                cpc_res = self.cpctf_loss_fn(z_time, z_freq, x_patches, h_freq_bins, tfc_loss=loss_tfc)
                total_aux_loss = cpc_res.get("loss_total", loss_tfc)
                self.last_loss_dict = cpc_res
            else:
                total_aux_loss = loss_tfc
                self.last_loss_dict = {"loss_tfc": loss_tfc}
            
            reconstruction = self.decoder(h_fused)
            # Reshape: [B*C, L] -> [B, L, C]
            reconstruction = reconstruction.view(batch_size, num_features, seq_len).permute(0, 2, 1)
            
            return reconstruction, total_aux_loss

        # FINETUNE (Forecasting)
        elif self.task_name == "finetune" and self.pred_len is not None:
            predictions = self.forecaster(h_fused)
            # Reshape: [B*C, Pred] -> [B, Pred, C]
            predictions = predictions.view(batch_size, num_features, self.pred_len).permute(0, 2, 1)
            return predictions

        # CLASSIFICATION
        else:
            # h_fused: [B*C, Hidden] -> reshape to [B, C, Hidden]
            h_fused_3d = h_fused.view(batch_size, num_features, self.hidden_dim)
            
            # Transpose for Conv1d: [B, C, Hidden] -> [B, Hidden, C]
            cnn_input = h_fused_3d.transpose(1, 2)
            
            # Avoid downsampling along the feature axis. Some classification datasets
            # only have 2-3 channels, so repeated pooling would collapse the length to 0.
            x = F.relu(self.conv1(cnn_input))
            x = self.dropout(x)
            x = F.relu(self.conv2(x))
            x = self.dropout(x)
            x = F.relu(self.conv3(x))
            x = self.global_pool(x).squeeze(-1)  # [B, 256]
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
        use_warping = getattr(args, 'use_warping', False)
        projection_dim = getattr(args, 'projection_dim', 128)

        # CPC-TF params
        use_cpc = getattr(args, 'use_cpc', False)
        cpc_kwargs = {
            'freq_mask_ratio': getattr(args, 'cpc_freq_mask_ratio', 0.2),
            'time_mask_ratio': getattr(args, 'cpc_time_mask_ratio', 0.2),
            'lambda_cpc': getattr(args, 'cpc_lambda', 0.1),
            'loss_type': getattr(args, 'cpc_loss_type', 'l2'),
            'pos_emb_dim': getattr(args, 'cpc_pos_emb_dim', 64),
            'hidden_dim': getattr(args, 'cpc_hidden_dim', 256),
        }
        
        # Patch params (Standard for ETTh1: P=16, S=8)
        patch_len = getattr(args, 'patch_len', 16)
        stride = getattr(args, 'stride', 8)

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
            use_warping=use_warping,
            projection_dim=projection_dim,
            patch_len=patch_len,
            stride=stride,
            hidden_dim=args.d_model,
            use_cpc=use_cpc,
            cpc_kwargs=cpc_kwargs,
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
            predict_x = predict_x * stdevs[:, 0, :].unsqueeze(1).repeat(1, self.input_len, 1)
            predict_x = predict_x + means[:, 0, :].unsqueeze(1).repeat(1, self.input_len, 1)

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
        use_warping = getattr(args, 'use_warping', False)
        projection_dim = getattr(args, 'projection_dim', 128)
        patch_len = getattr(args, 'patch_len', 16)
        stride = getattr(args, 'stride', 8)

        use_cpc = getattr(args, 'use_cpc', False)
        cpc_kwargs = {
            'freq_mask_ratio': getattr(args, 'cpc_freq_mask_ratio', 0.2),
            'time_mask_ratio': getattr(args, 'cpc_time_mask_ratio', 0.2),
            'lambda_cpc': getattr(args, 'cpc_lambda', 0.1),
            'loss_type': getattr(args, 'cpc_loss_type', 'l2'),
            'pos_emb_dim': getattr(args, 'cpc_pos_emb_dim', 64),
            'hidden_dim': getattr(args, 'cpc_hidden_dim', 256),
        }
        
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
            use_warping=use_warping,
            projection_dim=projection_dim,
            patch_len=patch_len,
            stride=stride,
            hidden_dim=args.d_model,
            use_cpc=use_cpc,
            cpc_kwargs=cpc_kwargs,
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