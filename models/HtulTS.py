import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
    - Patching + Mixer Backbone (Time)
    - Global FFT Backbone (Freq)
    - Masking for Pretraining
    - Learnable Fusion & TFC
    """
    def __init__(self, in_channels, out_channels, task_name="pretrain", pred_len=None, 
                 num_classes=2, input_len=336, use_noise=False, 
                 use_forgetting=False, forgetting_type="activation", forgetting_rate=0.1,
                 tfc_weight=0.05, tfc_warmup_steps=0, use_real_imag=False,
                 projection_dim=128, patch_len=16, stride=8):
        super(LightweightModel, self).__init__()
        self.task_name = task_name
        self.pred_len = pred_len
        self.num_classes = num_classes
        self.input_len = input_len
        self.use_noise = use_noise
        self.use_forgetting = use_forgetting
        self.hidden_dim = 512
        
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
            use_real_imag=use_real_imag
        )
        
        # Projection heads for TF-C contrastive loss
        self.time_projection = ProjectionHead(self.hidden_dim, self.hidden_dim // 2, projection_dim)
        self.freq_projection = ProjectionHead(self.hidden_dim, self.hidden_dim // 2, projection_dim)
        
        self.tfc_loss_fn = TFC_Loss(temperature=0.2)
        self.fusion = LearnableFusion(self.hidden_dim)

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
            self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
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
        if self.task_name == "pretrain":
            # Default to Masking for SOTA results
            # Only use Noise if explicitly requested via args AND flag
            if self.use_noise and add_noise_flag:
                x_input, _ = self.add_noise(x, noise_level=noise_level)
            else:
                x_input, _ = self.random_masking(x, mask_ratio=0.4)
        else:
            x_input = x
            
        # Flatten: [B, L, C] -> [B*C, L]
        x_permuted = x_input.permute(0, 2, 1) 
        x_flat = x_permuted.reshape(batch_size * num_features, seq_len)
        
        # Group IDs for TFC
        group_ids = torch.arange(batch_size, device=x.device).repeat_interleave(num_features)
        
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
        h_freq = self.freq_backbone(x_flat)
        
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
            loss_tfc = self.tfc_loss_fn(z_time, z_freq, group_ids=group_ids) * self.get_tfc_weight()
            
            if self.training:
                self.increment_step()
            
            reconstruction = self.decoder(h_fused)
            # Reshape: [B*C, L] -> [B, L, C]
            reconstruction = reconstruction.view(batch_size, num_features, seq_len).permute(0, 2, 1)
            
            return reconstruction, loss_tfc

        # FINETUNE (Forecasting)
        elif self.task_name == "finetune" and self.pred_len is not None:
            predictions = self.forecaster(h_fused)
            # Reshape: [B*C, Pred] -> [B, Pred, C]
            predictions = predictions.view(batch_size, num_features, self.pred_len).permute(0, 2, 1)
            return predictions

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
            stride=stride
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
            stride=stride
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