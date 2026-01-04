import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMBlock(nn.Module):
    """
    A simple, research-grade Neural State Space Model (SSM) block.

    - Learnable state dimension (state_dim)
    - Linear state transition: x_t = A x_{t-1} + B u_t
    - Output projection: y_t = C x_t
    - Processes input sequentially over time (for-loop)
    - Optional LayerNorm and dropout applied to states

    Notes:
    - Inputs: u of shape [batch, seq_len, in_channels]
    - States: [batch, seq_len, state_dim]
    - Returns: (states, outputs) where outputs = C(states)
    """
    def __init__(self,
                 in_channels: int,
                 state_dim: int = 64,
                 output_dim: int = None,
                 layer_norm: bool = True,
                 dropout: float = 0.0,
                 init_scale: float = 0.05):
        super(SSMBlock, self).__init__()
        self.in_channels = in_channels
        self.state_dim = state_dim
        self.output_dim = output_dim or state_dim

        # State transition matrix A (learnable)
        # We parametrize A directly as a small initialized matrix.
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * init_scale)

        # Input-to-state map B: u_t -> contribution to new state
        self.B = nn.Linear(in_channels, state_dim, bias=True)

        # Output projection C: state -> output (applied per time-step)
        self.C = nn.Linear(state_dim, self.output_dim)

        # Optional normalization and dropout on states
        self.norm = nn.LayerNorm(state_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else None

        # Activation used after transition
        self.act = nn.ReLU()

    def forward(self, u: torch.Tensor, initial_state: torch.Tensor = None,
                return_states: bool = True):
        """
        Forward pass through the SSM sequentially.

        Args:
            u: input sequence [batch, seq_len, in_channels]
            initial_state: optional initial state [batch, state_dim]
            return_states: whether to return the state trajectory

        Returns:
            states: [batch, seq_len, state_dim]
            outputs: [batch, seq_len, output_dim]
        """
        batch, seq_len, in_ch = u.shape
        device = u.device

        if initial_state is None:
            state = torch.zeros(batch, self.state_dim, device=device, dtype=u.dtype)
        else:
            state = initial_state

        states = []

        # Sequential update: for research clarity we use an explicit loop
        for t in range(seq_len):
            ut = u[:, t, :]  # [batch, in_channels]

            # Linear state transition: x_t = A x_{t-1} + B u_t
            # state @ A.T does batched matmul: [batch, state_dim] x [state_dim, state_dim]^T
            state = torch.matmul(state, self.A.t()) + self.B(ut)

            # Optional normalization, activation and dropout
            if self.norm is not None:
                state = self.norm(state)
            state = self.act(state)
            if self.dropout is not None:
                state = self.dropout(state)

            states.append(state.unsqueeze(1))

        states = torch.cat(states, dim=1)  # [batch, seq_len, state_dim]

        # Project states to outputs with C (applies to last dim)
        outputs = self.C(states)  # supports (..., state_dim)

        if return_states:
            return states, outputs
        else:
            return outputs


class SSMTimeSeriesModel(nn.Module):
    """
    Time series model that uses SSMBlock as a shared backbone for pretraining
    (reconstruction) and forecasting, and a Conv1D classifier for
    classification (unchanged architecture).

    Interface matches the existing `LightweightModel` for easy swapping.
    """
    def __init__(self, in_channels, out_channels, task_name="pretrain",
                 pred_len=None, num_classes=2, input_len=336,
                 state_dim: int = 128, layer_norm: bool = True, dropout: float = 0.1,
                 use_noise: bool = False):
        super(SSMTimeSeriesModel, self).__init__()
        self.task_name = task_name
        self.pred_len = pred_len
        self.num_classes = num_classes
        self.input_len = input_len
        self.in_channels = in_channels
        self.use_noise = use_noise

        # Shared SSM backbone: returns full state trajectory and projected outputs
        self.backbone = SSMBlock(in_channels=in_channels,
                                 state_dim=state_dim,
                                 output_dim=state_dim,
                                 layer_norm=layer_norm,
                                 dropout=dropout)

        # Heads
        if self.task_name == "pretrain":
            # Reconstruct the full input sequence from a fixed embedding (last state)
            self.decoder = nn.Linear(state_dim, input_len * in_channels)

        elif self.task_name == "finetune" and self.pred_len is not None:
            # Forecast pred_len steps for all channels from embedding
            self.forecaster = nn.Linear(state_dim, self.pred_len * in_channels)

        # Classification: reuse Conv1D classifier architecture (unchanged)
        # These are identical to the conv layers in the original LightweightModel
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def add_noise(self, x, noise_level=0.1):
        std_dev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        noise = torch.randn_like(x)
        scaled_noise = noise * std_dev * noise_level
        return x + scaled_noise, scaled_noise

    def forward(self, x, add_noise_flag=False, noise_level=0.1):
        """
        x: [batch, seq_len, in_channels]
        """
        batch_size, seq_len, num_features = x.size()

        if add_noise_flag and self.use_noise and self.task_name == "pretrain":
            x, _ = self.add_noise(x, noise_level=noise_level)

        # Backbone: get full state trajectory and outputs
        states, _ = self.backbone(x)  # states: [batch, seq_len, state_dim]

        # Fixed-size embedding per sample: use last state
        embedding = states[:, -1, :]  # [batch, state_dim]

        if self.task_name == "pretrain":
            recon_flat = self.decoder(embedding)  # [batch, input_len * in_channels]
            recon = recon_flat.view(batch_size, seq_len, num_features)
            return recon

        elif self.task_name == "finetune" and self.pred_len is not None:
            pred_flat = self.forecaster(embedding)  # [batch, pred_len * in_channels]
            preds = pred_flat.view(batch_size, self.pred_len, num_features)
            return preds

        else:
            # Classification: reuse Conv1D classifier exactly as before
            cnn_input = x.permute(0, 2, 1)  # [batch, channels, length]
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
    Lightweight model with shared linear backbone:
    - MLP backbone for both pretrain and forecasting
    - Conv layers ONLY for classification task
    """
    def __init__(self, in_channels, out_channels, task_name="pretrain", pred_len=None, 
                 num_classes=2, input_len=336, use_noise=False, 
                 use_forgetting=False, forgetting_type="activation", forgetting_rate=0.1, use_multivariate=True):
        super(LightweightModel, self).__init__()
        self.task_name = task_name
        self.pred_len = pred_len
        self.num_classes = num_classes
        self.input_len = input_len
        self.use_noise = use_noise
        self.use_forgetting = use_forgetting
        self.in_channels = in_channels
        self.use_multivariate = use_multivariate
        
        # SHARED LINEAR BACKBONE for pretrain and forecasting
        hidden_dim = 512
        
        self.linear_backbone = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Option 2: Direct multivariate processing
        self.multivariate_backbone = nn.Sequential(
            nn.Linear(input_len * in_channels, hidden_dim * 2),  # Flatten all features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Choose which backbone to use
        self.use_multivariate = True 

        # Forgetting mechanisms (applied after backbone)
        if self.use_forgetting:
            self.forgetting_layer = ForgettingMechanisms(
                hidden_dim, forgetting_type, forgetting_rate
            )
        
        # Task-specific heads
        if self.task_name == "pretrain":
            # Reconstruction head
            if hasattr(self, 'use_multivariate') and self.use_multivariate:
                # For multivariate: decode from hidden_dim to seq_len * num_features
                self.decoder = nn.Linear(hidden_dim, input_len * in_channels)
            else:
                # Original: decode from hidden_dim to seq_len (per feature)
                self.decoder = nn.Linear(hidden_dim, input_len)
        
        elif self.task_name == "finetune" and self.pred_len is not None:
            head_width = min(2048, max(64, 4*int(self.pred_len)))

            # Enhanced forecasting head
            if hasattr(self, 'use_multivariate') and self.use_multivariate:
                # For multivariate: predict all features at once
                self.forecaster = nn.Sequential(
                    nn.Linear(hidden_dim, head_width),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(head_width, self.pred_len * in_channels)
                )
            else:
                # Original: predict per feature
                self.forecaster = nn.Sequential(
                    nn.Linear(hidden_dim, head_width),
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
        
        # 1. SHARED BACKBONE PASS (Applies to ALL tasks now)
        # ---------------------------------------------------
        # Store clean signal for logic references if needed
        x_clean = x.clone()
        
        # Add noise only if strictly pretraining
        if self.task_name == "pretrain" and add_noise_flag and self.use_noise:
            x, noise = self.add_noise(x, noise_level=noise_level)
            
        if self.use_multivariate:
            # OPTION 2: Direct multivariate processing
            # Flatten all dimensions: [batch, seq_len, features] -> [batch, seq_len * features]
            x_flat_multivariate = x.reshape(batch_size, seq_len * num_features)
            
            # Pass through multivariate backbone
            # features shape: [batch, hidden_dim]
            features = self.multivariate_backbone(x_flat_multivariate)
            
        else:
            # OPTION 1: Original feature-wise processing
            # Flatten: [batch, seq_len, features] -> [batch * features, seq_len]
            x_permuted = x.permute(0, 2, 1)
            x_flat = x_permuted.reshape(batch_size * num_features, seq_len)
            
            # Pass through the original Linear Backbone
            # features shape: [batch * features, hidden_dim (512)]
            features = self.linear_backbone(x_flat)

        # Apply forgetting if enabled
        if self.use_forgetting:
            is_finetune = (self.task_name == "finetune")
            features = self.forgetting_layer(features, is_finetune=is_finetune)

        # 2. TASK-SPECIFIC HEADS
        # ---------------------------------------------------
        
        # CASE A: PRETRAINING (Reconstruction)
        if self.task_name == "pretrain":
            if self.use_multivariate:
                # For multivariate: features is [batch, hidden_dim]
                # Need to decode back to [batch, seq_len * num_features] then reshape
                reconstruction_flat = self.decoder(features)  # [batch, input_len * num_features]
                reconstruction = reconstruction_flat.view(batch_size, seq_len, num_features)
            else:
                # Original reconstruction logic
                reconstruction = self.decoder(features)  # [batch*features, seq_len]
                reconstruction = reconstruction.view(batch_size, num_features, seq_len)
                reconstruction = reconstruction.permute(0, 2, 1)  # [batch, seq_len, num_features]
            return reconstruction

        # CASE B: FORECASTING (Finetune with pred_len)
        elif self.task_name == "finetune" and self.pred_len is not None:
            if self.use_multivariate:
                # For multivariate: features is [batch, hidden_dim]
                predictions_flat = self.forecaster(features)  # [batch, pred_len * num_features]
                predictions = predictions_flat.view(batch_size, self.pred_len, num_features)
            else:
                # Original forecasting logic
                predictions = self.forecaster(features)  # [batch*features, pred_len]
                predictions = predictions.view(batch_size, num_features, self.pred_len)
                predictions = predictions.permute(0, 2, 1)  # [batch, pred_len, num_features]
            return predictions

        # CASE C: CLASSIFICATION (Finetune without pred_len)
        else:
            # Reshape features to be compatible with Conv1d
            # Backbone output: [Batch*Features, Hidden_Dim]
            # Conv Input Needed: [Batch, Channels, Length]
            # We map: Features -> Channels, Hidden_Dim -> Length
            
            # New shape: [Batch, Features, Hidden_Dim]
            cnn_input = features.view(batch_size, num_features, -1)
            
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

        # Use the SSM-based model as shared backbone+heads
        self.model = SSMTimeSeriesModel(
            in_channels=args.enc_in,
            out_channels=args.enc_in,
            task_name=self.task_name,
            pred_len=self.pred_len if hasattr(args, 'pred_len') else None,
            num_classes=getattr(args, 'num_classes', 2),
            input_len=args.input_len,
            state_dim=getattr(args, 'state_dim', 128),
            layer_norm=getattr(args, 'use_norm', True),
            dropout=getattr(args, 'dropout', 0.1),
            use_noise=self.use_noise
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
        predict_x = self.model(x, add_noise_flag=True, noise_level=self.noise_level)

        # Denormalization
        if self.use_norm:
            predict_x = predict_x * stdevs
            predict_x = predict_x + means

        return predict_x

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
        
        # Use SSM-based model for pretrain/finetune/classification
        self.model = SSMTimeSeriesModel(
            in_channels=args.enc_in,
            out_channels=args.enc_in,
            task_name=self.task_name,
            pred_len=None,
            num_classes=self.num_classes,
            input_len=args.input_len,
            state_dim=getattr(args, 'state_dim', 128),
            layer_norm=getattr(args, 'use_norm', True),
            dropout=getattr(args, 'dropout', 0.1),
            use_noise=self.use_noise
        )

    def pretrain(self, x):
        # Add noise during pretrain
        return self.model(x, add_noise_flag=True, noise_level=self.noise_level)

    def forecast(self, x):
        # No noise during finetune/classification
        return self.model(x, add_noise_flag=False)
    
    def forward(self, batch_x):
        if self.task_name == "pretrain":
            return self.pretrain(batch_x)
        elif self.task_name == "finetune":
            return self.forecast(batch_x)
        else:
            raise ValueError("task_name should be 'pretrain' or 'finetune'")