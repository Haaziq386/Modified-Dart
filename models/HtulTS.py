import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FrequencyEncoder(nn.Module):
    """
    Frequency-domain encoder using FFT with improved stability.
    Takes time-domain input and produces frequency embeddings.
    
    Improvements:
    - Hann window before FFT to reduce spectral leakage
    - log1p(abs(fft)) for amplitude (stable, handles varying magnitudes)
    - sin(phase) and cos(phase) instead of raw phase (avoids discontinuities)
    - Per-sample normalization of frequency features
    - Optional real/imag features via use_real_imag flag
    """
    def __init__(self, input_len, hidden_dim=512, dropout=0.2, use_real_imag=False):
        super(FrequencyEncoder, self).__init__()
        self.input_len = input_len
        self.hidden_dim = hidden_dim
        self.use_real_imag = use_real_imag
        
        # FFT output size: L//2 + 1 complex values
        self.num_freq_bins = input_len // 2 + 1
        
        # Feature size depends on representation mode:
        # - use_real_imag=True: real + imag = 2 * num_freq_bins
        # - use_real_imag=False: log_amp + sin(phase) + cos(phase) = 3 * num_freq_bins
        if use_real_imag:
            self.fft_output_size = 2 * self.num_freq_bins
        else:
            self.fft_output_size = 3 * self.num_freq_bins
        
        # Register Hann window as buffer (not a parameter, moves with model)
        self.register_buffer('hann_window', torch.hann_window(input_len))
        
        # Projection from frequency features to hidden_dim
        self.freq_projection = nn.Sequential(
            nn.Linear(self.fft_output_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: [N, L] time-domain signal (N samples, L sequence length)
        Returns:
            h_freq: [N, hidden_dim] frequency embeddings
        """
        # Apply Hann window to reduce spectral leakage
        # x: [N, L], hann_window: [L] -> broadcasts correctly
        x_windowed = x * self.hann_window  # [N, L]
        
        # Compute FFT
        spectrum = torch.fft.rfft(x_windowed, dim=-1)  # [N, num_freq_bins] complex
        
        if self.use_real_imag:
            # Use real and imaginary parts directly
            real_part = spectrum.real  # [N, num_freq_bins]
            imag_part = spectrum.imag  # [N, num_freq_bins]
            freq_features = torch.cat([real_part, imag_part], dim=-1)  # [N, 2*num_freq_bins]
        else:
            # Extract stable amplitude and phase representations
            # log1p(abs) for amplitude: handles varying magnitudes, avoids log(0)
            amplitude = torch.log1p(torch.abs(spectrum))  # [N, num_freq_bins]
            
            # sin(phase) and cos(phase) instead of raw phase
            # Avoids discontinuity at ±π and provides smooth gradients
            phase = torch.angle(spectrum)  # [N, num_freq_bins]
            sin_phase = torch.sin(phase)   # [N, num_freq_bins]
            cos_phase = torch.cos(phase)   # [N, num_freq_bins]
            
            # Concatenate: amplitude, sin(phase), cos(phase)
            freq_features = torch.cat([amplitude, sin_phase, cos_phase], dim=-1)  # [N, 3*num_freq_bins]
        
        # Per-sample normalization (zero mean, unit variance per sample)
        # This helps with varying signal magnitudes across samples
        mean = freq_features.mean(dim=-1, keepdim=True)
        std = freq_features.std(dim=-1, keepdim=True).clamp(min=1e-6)
        freq_features = (freq_features - mean) / std  # [N, fft_output_size]
        
        # Project to hidden_dim
        h_freq = self.freq_projection(freq_features)  # [N, hidden_dim]
        
        return h_freq


class TFC_Loss(nn.Module):
    """
    Time-Frequency Consistency Loss using NT-Xent (Normalized Temperature-scaled Cross Entropy).
    Maximizes similarity between time and frequency embeddings of the same sample
    while minimizing similarity with other samples.
    
    Improvements:
    - group_ids support to prevent channels from the same sample being treated as negatives
    - Proper masking of same-group negatives while keeping diagonal positives
    - Safe handling when masking removes all negatives (returns zero loss)
    """
    def __init__(self, temperature=0.2):
        super(TFC_Loss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z_time, z_freq, group_ids=None):
        """
        Args:
            z_time: [N, hidden_dim] time-domain embeddings
            z_freq: [N, hidden_dim] frequency-domain embeddings
            group_ids: [N] optional tensor indicating which group (e.g., original sample)
                       each embedding belongs to. Embeddings in the same group will not
                       be treated as negatives. If None, standard NT-Xent is used.
        Returns:
            loss: scalar NT-Xent loss
        """
        batch_size = z_time.size(0)
        device = z_time.device
        
        if batch_size < 2:
            # NT-Xent requires at least 2 samples for contrastive learning
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Normalize embeddings
        z_time = F.normalize(z_time, dim=-1)
        z_freq = F.normalize(z_freq, dim=-1)
        
        # Compute cosine similarity matrix
        # sim_tf[i,j] = similarity between z_time[i] and z_freq[j]
        sim_tf = torch.matmul(z_time, z_freq.T) / self.temperature  # [N, N]
        
        if group_ids is not None:
            # Create mask for same-group pairs (these should not be negatives)
            # group_ids: [N] -> same_group_mask[i,j] = True if group_ids[i] == group_ids[j]
            same_group_mask = group_ids.unsqueeze(0) == group_ids.unsqueeze(1)  # [N, N]
            
            # Diagonal elements are positives, keep them regardless of group
            diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
            
            # Mask to apply: same group but NOT the diagonal (i.e., same-group negatives)
            # These positions should be masked out (set to -inf)
            mask_out = same_group_mask & ~diag_mask  # [N, N]
            
            # Check if we have any valid negatives left
            # Valid negatives = not same group OR is diagonal
            valid_negatives = ~same_group_mask | diag_mask  # [N, N]
            # Each row must have at least one non-positive (negative) entry
            # Positives are on diagonal, so we need at least one valid entry that's not diagonal
            has_valid_negatives = (valid_negatives & ~diag_mask).any(dim=1)  # [N]
            
            if not has_valid_negatives.all():
                # Some rows have no valid negatives, return zero loss safely
                # This can happen if all samples are from the same group
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            # Apply mask: set same-group non-diagonal entries to -inf
            sim_tf = sim_tf.masked_fill(mask_out, float('-inf'))
        
        # Positive pairs are on the diagonal (same sample index)
        labels = torch.arange(batch_size, device=device)
        
        # Cross-entropy loss: time->freq
        loss_tf = F.cross_entropy(sim_tf, labels)
        
        # For freq->time, we need to handle the transposed matrix
        sim_ft = sim_tf.T  # [N, N] - transpose also transposes the mask
        loss_ft = F.cross_entropy(sim_ft, labels)
        
        # Check for NaN and return zero if detected
        loss = (loss_tf + loss_ft) / 2.0
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss


class ProjectionHead(nn.Module):
    """
    Small MLP projection head for contrastive learning.
    Maps backbone embeddings to a lower-dimensional space where contrastive loss is computed.
    This is standard practice in contrastive learning (SimCLR, MoCo, etc.)
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [N, input_dim] backbone embeddings
        Returns:
            z: [N, output_dim] projected embeddings for contrastive loss
        """
        return self.net(x)


class ForgettingMechanisms(nn.Module):
    """
    Forgetting mechanisms for neural networks.
    Applied only during finetuning to help adapt pretrained representations.
    
    Fixed: Broadcasting bugs for inputs with shape [N, D] (not [B, C, D])
    """
    def __init__(self, hidden_dim, forgetting_type="activation", forgetting_rate=0.1):
        super(ForgettingMechanisms, self).__init__()
        self.forgetting_type = forgetting_type
        self.forgetting_rate = forgetting_rate
        self.hidden_dim = hidden_dim
        
        if forgetting_type == "activation":
            # Learnable forgetting gates for activations
            self.forget_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.Sigmoid()
            )
        elif forgetting_type == "weight":
            # Weight-level forgetting with importance scores
            self.importance_scores = nn.Parameter(torch.ones(hidden_dim))
            self.decay_factor = nn.Parameter(torch.tensor(forgetting_rate))
        elif forgetting_type == "adaptive":
            # Adaptive forgetting based on gradient magnitude
            self.forget_threshold = nn.Parameter(torch.tensor(0.5))
            self.forget_scale = nn.Parameter(torch.tensor(forgetting_rate))
    
    def forward(self, x, is_finetune=False):
        """
        Args:
            x: [N, D] input features (N samples, D hidden_dim)
               Note: Input is 2D, NOT 3D. Broadcasting must account for this.
        Returns:
            x: [N, D] output with forgetting applied (if finetuning)
        """
        if not is_finetune:
            return x
            
        if self.forgetting_type == "activation":
            # Activation-level forgetting
            # x: [N, D], forget_gate outputs [N, D]
            forget_mask = self.forget_gate(x)  # [N, D]
            # Apply forgetting: multiply by (1 - forget_rate * forget_mask)
            forget_factor = 1.0 - self.forgetting_rate * forget_mask
            return x * forget_factor
            
        elif self.forgetting_type == "weight":
            # Weight-level forgetting (applied to features)
            # importance_scores: [D], decay_factor: scalar
            importance = torch.sigmoid(self.importance_scores)  # [D]
            decay = torch.sigmoid(self.decay_factor)  # scalar
            forget_factor = importance * (1.0 - decay)  # [D]
            # Broadcasting: x [N, D] * forget_factor [D] -> [N, D]
            return x * forget_factor  # Fixed: removed incorrect unsqueeze
            
        elif self.forgetting_type == "adaptive":
            # Adaptive forgetting based on activation magnitude
            # x: [N, D]
            activation_magnitude = torch.mean(torch.abs(x), dim=-1, keepdim=True)  # [N, 1]
            forget_prob = torch.sigmoid(self.forget_scale * (activation_magnitude - self.forget_threshold))  # [N, 1]
            # Use straight-through estimator for gradient flow during training
            if self.training:
                forget_mask = (torch.rand_like(forget_prob) < forget_prob).float()  # [N, 1]
            else:
                forget_mask = (forget_prob > 0.5).float()  # Deterministic at eval
            return x * (1.0 - forget_mask * self.forgetting_rate)
        
        return x


class LearnableFusion(nn.Module):
    """
    Learnable fusion module to combine time and frequency embeddings.
    Replaces simple element-wise addition with:
    concat -> Linear -> LayerNorm -> ReLU
    
    This allows the model to learn how to weight and combine the two modalities.
    """
    def __init__(self, hidden_dim):
        super(LearnableFusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, h_time, h_freq):
        """
        Args:
            h_time: [N, hidden_dim] time-domain embeddings
            h_freq: [N, hidden_dim] frequency-domain embeddings
        Returns:
            h_fused: [N, hidden_dim] fused embeddings
        """
        # Concatenate along feature dimension
        h_concat = torch.cat([h_time, h_freq], dim=-1)  # [N, 2*hidden_dim]
        # Learn optimal fusion
        h_fused = self.fusion(h_concat)  # [N, hidden_dim]
        return h_fused


class ResidualForecastingHead(nn.Module):
    """
    Lightweight 2-layer MLP with residual connection for forecasting.
    Improves gradient flow and representation quality compared to single-layer MLP.
    
    Output shape: [B, pred_len, C]
    """
    def __init__(self, hidden_dim, pred_len, dropout=0.1):
        super(ResidualForecastingHead, self).__init__()
        self.pred_len = pred_len
        
        # Compute intermediate width (bounded for efficiency)
        head_width = min(2048, max(256, 4 * pred_len))
        
        # First layer: expand
        self.fc1 = nn.Linear(hidden_dim, head_width)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(head_width)
        
        # Second layer: compress to pred_len
        self.fc2 = nn.Linear(head_width, pred_len)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual projection (if dimensions don't match)
        if hidden_dim != pred_len:
            self.residual_proj = nn.Linear(hidden_dim, pred_len)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x: [N, hidden_dim] fused embeddings (N = batch_size * num_features)
        Returns:
            out: [N, pred_len] predictions per channel
        """
        # Residual path
        residual = self.residual_proj(x)  # [N, pred_len]
        
        # Main path
        out = self.fc1(x)  # [N, head_width]
        out = F.relu(out)
        out = self.norm1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)  # [N, pred_len]
        out = self.dropout2(out)
        
        # Add residual
        out = out + residual  # [N, pred_len]
        
        return out

class LightweightModel(nn.Module):
    """
    Lightweight model with Time-Frequency Consistency (TF-C) architecture:
    - Time-domain MLP backbone + Frequency-domain encoder
    - Contrastive learning between time and frequency representations
    - Conv layers ONLY for classification task
    
    Improvements:
    - Projection heads for TF-C contrastive loss (standard practice)
    - Learnable fusion instead of simple addition
    - Residual forecasting head
    - Configurable TF-C weight with optional warm-up
    - Proper group_ids handling for channel flattening
    - FrequencyEncoder with improved stability (Hann window, log amplitude, sin/cos phase)
    """
    def __init__(self, in_channels, out_channels, task_name="pretrain", pred_len=None, 
                 num_classes=2, input_len=336, use_noise=False, 
                 use_forgetting=False, forgetting_type="activation", forgetting_rate=0.1,
                 tfc_weight=0.05, tfc_warmup_steps=0, use_real_imag=False,
                 projection_dim=128):
        super(LightweightModel, self).__init__()
        self.task_name = task_name
        self.pred_len = pred_len
        self.num_classes = num_classes
        self.input_len = input_len
        self.use_noise = use_noise
        self.use_forgetting = use_forgetting
        self.hidden_dim = 512
        
        # TF-C weighting configuration
        self.tfc_weight = tfc_weight
        self.tfc_warmup_steps = tfc_warmup_steps
        self.register_buffer('_step_counter', torch.tensor(0, dtype=torch.long))
        
        # TIME-DOMAIN BACKBONE (MLP)
        self.time_backbone = nn.Sequential(
            nn.Linear(input_len, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # FREQUENCY-DOMAIN BACKBONE (improved)
        self.freq_backbone = FrequencyEncoder(
            input_len=input_len,
            hidden_dim=self.hidden_dim,
            dropout=0.2,
            use_real_imag=use_real_imag
        )
        
        # Projection heads for TF-C contrastive loss
        # Contrastive loss should be computed on projected embeddings, not backbone outputs
        self.time_projection = ProjectionHead(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            output_dim=projection_dim
        )
        self.freq_projection = ProjectionHead(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            output_dim=projection_dim
        )
        
        # TF-C Contrastive Loss
        self.tfc_loss_fn = TFC_Loss(temperature=0.2)
        
        # Learnable fusion (replaces h_time + h_freq)
        self.fusion = LearnableFusion(self.hidden_dim)

        # Forgetting mechanisms (applied after backbone fusion)
        if self.use_forgetting:
            self.forgetting_layer = ForgettingMechanisms(
                self.hidden_dim, forgetting_type, forgetting_rate
            )
        
        # Task-specific heads
        if self.task_name == "pretrain":
            # Reconstruction head (from fused representation)
            self.decoder = nn.Linear(self.hidden_dim, input_len)
        
        elif self.task_name == "finetune" and self.pred_len is not None:
            # Enhanced forecasting head with residual connection
            self.forecaster = ResidualForecastingHead(
                hidden_dim=self.hidden_dim,
                pred_len=self.pred_len,
                dropout=0.1
            )
   
        else:  # Classification (completely separate Conv architecture)
            # Conv layers ONLY for classification
            self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.dropout = nn.Dropout(0.2)
            
            # Classifier
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
    
    def get_tfc_weight(self):
        """
        Get the current TF-C weight, with optional warm-up scaling.
        Returns the configured tfc_weight, optionally scaled by warm-up progress.
        """
        if self.tfc_warmup_steps <= 0:
            return self.tfc_weight
        
        # Linear warm-up
        progress = min(1.0, self._step_counter.item() / self.tfc_warmup_steps)
        return self.tfc_weight * progress
    
    def increment_step(self):
        """Increment the internal step counter (call once per training step)."""
        self._step_counter += 1

    def add_noise(self, x, noise_level=0.1):
        """
        Add Gaussian noise proportional to standard deviation
        x: [batch, seq_len, features]
        noise_level: scaling factor for noise (default 0.1 = 10% of std)
        """
        # Compute per-feature standard deviation
        std_dev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [batch, 1, features]
        
        # Sample Gaussian noise
        noise = torch.randn_like(x)  # [batch, seq_len, features]
        
        # Scale noise by std_dev and noise_level
        scaled_noise = noise * std_dev * noise_level
        
        # Add noise to signal
        x_noisy = x + scaled_noise
        
        return x_noisy, scaled_noise

    def forward(self, x, add_noise_flag=False, noise_level=0.1):
        """
        Forward pass for the TF-C model.
        
        Args:
            x: [batch_size, seq_len, num_features] input time series
            add_noise_flag: whether to add noise (for denoising pretraining)
            noise_level: noise magnitude as fraction of std
            
        Returns:
            For pretrain: (reconstruction, loss_tfc) where reconstruction is [B, seq_len, C]
            For finetune (forecasting): predictions [B, pred_len, C]
            For classification: logits [B, num_classes]
        """
        # Input: [batch_size, seq_len, num_features]
        batch_size, seq_len, num_features = x.size()
        
        # 1. DUAL BACKBONE PASS (Time + Frequency)
        # ---------------------------------------------------
        # Store clean signal for logic references if needed
        x_clean = x.clone()
        
        # Add noise only if strictly pretraining (applied before both encoders)
        if self.task_name == "pretrain" and add_noise_flag and self.use_noise:
            x, noise = self.add_noise(x, noise_level=noise_level)
            
        # Flatten: [batch, seq_len, features] -> [batch * features, seq_len]
        # This treats each channel independently
        x_permuted = x.permute(0, 2, 1)  # [B, C, L]
        x_flat = x_permuted.reshape(batch_size * num_features, seq_len)  # [B*C, L]
        
        # Construct group_ids for TF-C loss
        # group_ids[i] = which original sample (batch index) this flattened sample came from
        # This prevents channels from the same sample being treated as negatives
        group_ids = torch.arange(batch_size, device=x.device).repeat_interleave(num_features)  # [B*C]
        
        # Step A: Time-domain encoding
        h_time = self.time_backbone(x_flat)  # [B*C, hidden_dim]
        
        # Step B: Frequency-domain encoding
        h_freq = self.freq_backbone(x_flat)  # [B*C, hidden_dim]
        
        # Step C: Learnable fusion (replaces simple addition)
        h_fused = self.fusion(h_time, h_freq)  # [B*C, hidden_dim]

        # Apply forgetting if enabled (on fused representation)
        if self.use_forgetting:
            is_finetune = (self.task_name == "finetune")
            h_fused = self.forgetting_layer(h_fused, is_finetune=is_finetune)

        # 2. TASK-SPECIFIC HEADS
        # ---------------------------------------------------
        
        # CASE A: PRETRAINING (Reconstruction + TF-C Loss)
        if self.task_name == "pretrain":
            # Project embeddings for contrastive loss (don't use backbone outputs directly)
            z_time = self.time_projection(h_time)  # [B*C, projection_dim]
            z_freq = self.freq_projection(h_freq)  # [B*C, projection_dim]
            
            # Compute TF-C contrastive loss with group masking
            loss_tfc = self.tfc_loss_fn(z_time, z_freq, group_ids=group_ids)
            
            # Apply TF-C weight (with optional warm-up)
            loss_tfc = loss_tfc * self.get_tfc_weight()
            
            # Increment step counter for warm-up
            if self.training:
                self.increment_step()
            
            # Reconstruction (denoising) from fused representation
            reconstruction = self.decoder(h_fused)  # [B*C, seq_len]
            reconstruction = reconstruction.view(batch_size, num_features, seq_len)  # [B, C, L]
            reconstruction = reconstruction.permute(0, 2, 1)  # [B, L, C]
            
            return reconstruction, loss_tfc

        # CASE B: FORECASTING (Finetune with pred_len)
        elif self.task_name == "finetune" and self.pred_len is not None:
            predictions = self.forecaster(h_fused)  # [B*C, pred_len]
            predictions = predictions.view(batch_size, num_features, self.pred_len)  # [B, C, pred_len]
            predictions = predictions.permute(0, 2, 1)  # [B, pred_len, C]
            return predictions

        # CASE C: CLASSIFICATION (Finetune without pred_len)
        else:
            # Reshape fused features to be compatible with Conv1d
            # Backbone output: [Batch*Features, Hidden_Dim]
            # Conv Input Needed: [Batch, Channels, Length]
            # We map: Features -> Channels, Hidden_Dim -> Length
            
            # New shape: [Batch, Features, Hidden_Dim]
            cnn_input = h_fused.view(batch_size, num_features, -1)
            
            # Pass the PRETRAINED features into the CNN
            x = F.relu(self.conv1(cnn_input))
            x = self.pool(x)
            x = self.dropout(x)
            
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = self.dropout(x)
            
            x = F.relu(self.conv3(x))
            
            x = self.global_pool(x)
            x = x.squeeze(-1)
            x = self.classifier(x)
            
            return x


class Model(nn.Module):
    """
    TimeDART with shared linear backbone + task-specific heads.
    
    Wrapper for LightweightModel with normalization and TF-C support.
    
    New parameters (via args):
    - tfc_weight: weight for TF-C loss (default 0.05)
    - tfc_warmup_steps: steps over which to warm up TF-C weight (default 0)
    - use_real_imag: use real/imag FFT features instead of amp/phase (default False)
    - projection_dim: dimension of contrastive projection heads (default 128)
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.input_len = args.input_len
        self.pred_len = args.pred_len
        self.task_name = args.task_name
        self.use_norm = args.use_norm
        self.use_noise = getattr(args, 'use_noise', False)
        self.noise_level = getattr(args, 'noise_level', 0.1)
        
        # Forgetting mechanism parameters
        self.use_forgetting = getattr(args, 'use_forgetting', False)
        self.forgetting_type = getattr(args, 'forgetting_type', 'activation')  # 'activation', 'weight', 'adaptive'
        self.forgetting_rate = getattr(args, 'forgetting_rate', 0.1)
        
        # TF-C parameters (new)
        tfc_weight = getattr(args, 'tfc_weight', 0.05)
        tfc_warmup_steps = getattr(args, 'tfc_warmup_steps', 0)
        use_real_imag = getattr(args, 'use_real_imag', False)
        projection_dim = getattr(args, 'projection_dim', 128)

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
            projection_dim=projection_dim
        )

    def pretrain(self, x):
        """
        Pretrain forward pass with reconstruction and TF-C loss.
        
        Args:
            x: [B, input_len, C] input time series
        Returns:
            predict_x: [B, input_len, C] reconstructed time series
            loss_tfc: scalar TF-C contrastive loss (already weighted)
        """
        batch_size, input_len, num_features = x.size()
        
        # Normalization
        if self.use_norm:
            means = torch.mean(x, dim=1, keepdim=True).detach()
            x = x - means
            stdevs = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = x / stdevs

        # Forward with noise (model will denoise)
        # Returns: (reconstruction, loss_tfc)
        predict_x, loss_tfc = self.model(x, add_noise_flag=True, noise_level=self.noise_level)

        # Denormalization
        if self.use_norm:
            predict_x = predict_x * stdevs
            predict_x = predict_x + means

        return predict_x, loss_tfc

    def forecast(self, x):
        """
        Forecast forward pass.
        
        Args:
            x: [B, input_len, C] input time series
        Returns:
            prediction: [B, pred_len, C] forecasted time series
        """
        batch_size, input_len, num_features = x.size()
        
        # Normalization
        if self.use_norm:
            means = torch.mean(x, dim=1, keepdim=True).detach()
            x = x - means
            stdevs = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = x / stdevs

        # Forward pass without noise
        prediction = self.model(x, add_noise_flag=False)

        # Denormalization
        if self.use_norm:
            prediction = prediction * stdevs[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            prediction = prediction + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return prediction
    
    def forward(self, batch_x):
        """
        Main forward pass, dispatches to pretrain or forecast based on task_name.
        
        Args:
            batch_x: [B, input_len, C] input time series
        Returns:
            For pretrain: (reconstruction, loss_tfc)
            For finetune: predictions [B, pred_len, C]
        """
        if self.task_name == "pretrain":
            # Returns: (reconstruction, loss_tfc)
            return self.pretrain(batch_x)
        elif self.task_name == "finetune":
            return self.forecast(batch_x)
        else:
            raise ValueError("task_name should be 'pretrain' or 'finetune'")


class ClsModel(nn.Module):
    """
    Classification model wrapper with TF-C architecture.
    """
    def __init__(self, args):
        super(ClsModel, self).__init__()
        self.input_len = args.input_len
        self.task_name = args.task_name
        self.num_classes = args.num_classes
        self.use_noise = getattr(args, 'use_noise', False)
        self.noise_level = getattr(args, 'noise_level', 0.1)
        
        # Forgetting mechanism parameters
        self.use_forgetting = getattr(args, 'use_forgetting', False)
        self.forgetting_type = getattr(args, 'forgetting_type', 'activation')
        self.forgetting_rate = getattr(args, 'forgetting_rate', 0.1)
        
        # TF-C parameters (new)
        tfc_weight = getattr(args, 'tfc_weight', 0.05)
        tfc_warmup_steps = getattr(args, 'tfc_warmup_steps', 0)
        use_real_imag = getattr(args, 'use_real_imag', False)
        projection_dim = getattr(args, 'projection_dim', 128)
        
        self.model = LightweightModel(
            in_channels=args.enc_in,
            out_channels=args.enc_in,
            task_name=self.task_name,
            pred_len=None,
            num_classes=self.num_classes,
            input_len=args.input_len,
            use_noise=self.use_noise,
            use_forgetting=self.use_forgetting,
            forgetting_type=self.forgetting_type,
            forgetting_rate=self.forgetting_rate,
            tfc_weight=tfc_weight,
            tfc_warmup_steps=tfc_warmup_steps,
            use_real_imag=use_real_imag,
            projection_dim=projection_dim
        )

    def pretrain(self, x):
        # Add noise during pretrain
        # Returns: (reconstruction, loss_tfc)
        return self.model(x, add_noise_flag=True, noise_level=self.noise_level)

    def forecast(self, x):
        # No noise during finetune/classification
        return self.model(x, add_noise_flag=False)
    
    def forward(self, batch_x):
        if self.task_name == "pretrain":
            # Returns: (reconstruction, loss_tfc)
            return self.pretrain(batch_x)
        elif self.task_name == "finetune":
            return self.forecast(batch_x)
        else:
            raise ValueError("task_name should be 'pretrain' or 'finetune'")