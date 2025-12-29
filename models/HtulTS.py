import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyEncoder(nn.Module):
    """
    Frequency-domain encoder using FFT.
    Takes time-domain input and produces frequency embeddings.
    """
    def __init__(self, input_len, hidden_dim=512, dropout=0.2):
        super(FrequencyEncoder, self).__init__()
        self.input_len = input_len
        self.hidden_dim = hidden_dim
        
        # FFT output size: L//2 + 1 complex values -> 2*(L//2+1) real values (amp + phase or real + imag)
        self.fft_output_size = 2 * (input_len // 2 + 1)
        
        # Projection from frequency features to hidden_dim
        self.freq_projection = nn.Sequential(
            nn.Linear(self.fft_output_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: [N, L] time-domain signal
        Returns:
            h_freq: [N, hidden_dim] frequency embeddings
        """
        # Compute FFT
        spectrum = torch.fft.rfft(x, dim=-1)  # [N, L//2+1] complex
        
        # Extract amplitude and phase
        amplitude = torch.abs(spectrum)  # [N, L//2+1]
        phase = torch.angle(spectrum)     # [N, L//2+1]
        
        # Concatenate amplitude and phase
        freq_features = torch.cat([amplitude, phase], dim=-1)  # [N, 2*(L//2+1)]
        
        # Project to hidden_dim
        h_freq = self.freq_projection(freq_features)  # [N, hidden_dim]
        
        return h_freq


class TFC_Loss(nn.Module):
    """
    Time-Frequency Consistency Loss using NT-Xent (Normalized Temperature-scaled Cross Entropy).
    Maximizes similarity between time and frequency embeddings of the same sample
    while minimizing similarity with other samples.
    """
    def __init__(self, temperature=0.2):
        super(TFC_Loss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z_time, z_freq):
        """
        Args:
            z_time: [N, hidden_dim] time-domain embeddings
            z_freq: [N, hidden_dim] frequency-domain embeddings
        Returns:
            loss: scalar NT-Xent loss
        """
        batch_size = z_time.size(0)
        
        if batch_size < 2:
            # NT-Xent requires at least 2 samples for contrastive learning
            return torch.tensor(0.0, device=z_time.device, requires_grad=True)
        
        # Normalize embeddings
        z_time = F.normalize(z_time, dim=-1)
        z_freq = F.normalize(z_freq, dim=-1)
        
        # Compute cosine similarity matrix
        # sim_tf[i,j] = similarity between z_time[i] and z_freq[j]
        sim_tf = torch.matmul(z_time, z_freq.T) / self.temperature  # [N, N]
        sim_ft = sim_tf.T  # [N, N]
        
        # Positive pairs are on the diagonal (same sample index)
        labels = torch.arange(batch_size, device=z_time.device)
        
        # Cross-entropy loss: time->freq and freq->time
        loss_tf = F.cross_entropy(sim_tf, labels)
        loss_ft = F.cross_entropy(sim_ft, labels)
        
        # Average both directions
        loss = (loss_tf + loss_ft) / 2.0
        
        return loss


class ForgettingMechanisms(nn.Module):
    """
    Forgetting mechanisms for neural networks
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
        if not is_finetune:
            return x
            
        if self.forgetting_type == "activation":
            # Activation-level forgetting
            forget_mask = self.forget_gate(x)
            # Apply forgetting: multiply by (1 - forget_rate * forget_mask)
            forget_factor = 1.0 - self.forgetting_rate * forget_mask
            return x * forget_factor
            
        elif self.forgetting_type == "weight":
            # Weight-level forgetting (applied to features)
            importance = torch.sigmoid(self.importance_scores)
            decay = torch.sigmoid(self.decay_factor)
            forget_factor = importance * (1.0 - decay)
            return x * forget_factor.unsqueeze(0).unsqueeze(0)
            
        elif self.forgetting_type == "adaptive":
            # Adaptive forgetting based on activation magnitude
            activation_magnitude = torch.mean(torch.abs(x), dim=1, keepdim=True)
            forget_prob = torch.sigmoid(self.forget_scale * (activation_magnitude - self.forget_threshold))
            forget_mask = torch.bernoulli(forget_prob)
            return x * (1.0 - forget_mask * self.forgetting_rate)
        
        return x

class LightweightModel(nn.Module):
    """
    Lightweight model with Time-Frequency Consistency (TF-C) architecture:
    - Time-domain MLP backbone + Frequency-domain encoder
    - Contrastive learning between time and frequency representations
    - Conv layers ONLY for classification task
    """
    def __init__(self, in_channels, out_channels, task_name="pretrain", pred_len=None, 
                 num_classes=2, input_len=336, use_noise=False, 
                 use_forgetting=False, forgetting_type="activation", forgetting_rate=0.1):
        super(LightweightModel, self).__init__()
        self.task_name = task_name
        self.pred_len = pred_len
        self.num_classes = num_classes
        self.input_len = input_len
        self.use_noise = use_noise
        self.use_forgetting = use_forgetting
        self.hidden_dim = 512
        
        # TIME-DOMAIN BACKBONE (MLP)
        self.time_backbone = nn.Sequential(
            nn.Linear(input_len, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # FREQUENCY-DOMAIN BACKBONE
        self.freq_backbone = FrequencyEncoder(
            input_len=input_len,
            hidden_dim=self.hidden_dim,
            dropout=0.2
        )
        
        # TF-C Contrastive Loss
        self.tfc_loss_fn = TFC_Loss(temperature=0.2)

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
            head_width = min(2048, max(64, 4*int(self.pred_len)))

            # Enhanced forecasting head with forgetting
            self.forecaster = nn.Sequential(
                nn.Linear(self.hidden_dim, head_width),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(head_width, self.pred_len)
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
        x_permuted = x.permute(0, 2, 1)
        x_flat = x_permuted.reshape(batch_size * num_features, seq_len)
        
        # Step A: Time-domain encoding
        h_time = self.time_backbone(x_flat)  # [batch * features, hidden_dim]
        
        # Step B: Frequency-domain encoding
        h_freq = self.freq_backbone(x_flat)  # [batch * features, hidden_dim]
        
        # Step C: Fusion (element-wise sum)
        h_fused = h_time + h_freq  # [batch * features, hidden_dim]

        # Apply forgetting if enabled (on fused representation)
        if self.use_forgetting:
            is_finetune = (self.task_name == "finetune")
            h_fused = self.forgetting_layer(h_fused, is_finetune=is_finetune)

        # 2. TASK-SPECIFIC HEADS
        # ---------------------------------------------------
        
        # CASE A: PRETRAINING (Reconstruction + TF-C Loss)
        if self.task_name == "pretrain":
            # Compute TF-C contrastive loss
            loss_tfc = self.tfc_loss_fn(h_time, h_freq)
            
            # Reconstruction (denoising) from fused representation
            reconstruction = self.decoder(h_fused)  # [batch*features, seq_len]
            reconstruction = reconstruction.view(batch_size, num_features, seq_len)
            reconstruction = reconstruction.permute(0, 2, 1)  # [batch, seq_len, num_features]
            
            return reconstruction, loss_tfc

        # CASE B: FORECASTING (Finetune with pred_len)
        elif self.task_name == "finetune" and self.pred_len is not None:
            predictions = self.forecaster(h_fused)  # [batch*features, pred_len]
            predictions = predictions.view(batch_size, num_features, self.pred_len)
            predictions = predictions.permute(0, 2, 1)  # [batch, pred_len, num_features]
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
    TimeDART with shared linear backbone + task-specific heads
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
            forgetting_rate=self.forgetting_rate
        )

    def pretrain(self, x):
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
        if self.task_name == "pretrain":
            # Returns: (reconstruction, loss_tfc)
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
        
        # FIX: Added parameter extraction for ClsModel
        self.use_forgetting = getattr(args, 'use_forgetting', False)
        self.forgetting_type = getattr(args, 'forgetting_type', 'activation')
        self.forgetting_rate = getattr(args, 'forgetting_rate', 0.1)
        
        self.model = LightweightModel(
            in_channels=args.enc_in,
            out_channels=args.enc_in,
            task_name=self.task_name,
            pred_len=None,
            num_classes=self.num_classes,
            input_len=args.input_len,
            use_noise=self.use_noise,
            # FIX: Passing the extracted parameters to the inner model
            use_forgetting=self.use_forgetting,
            forgetting_type=self.forgetting_type,
            forgetting_rate=self.forgetting_rate
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