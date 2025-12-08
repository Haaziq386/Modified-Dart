import torch
import torch.nn as nn
import torch.nn.functional as F

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
                 use_forgetting=False, forgetting_type="activation", forgetting_rate=0.1):
        super(LightweightModel, self).__init__()
        self.task_name = task_name
        self.pred_len = pred_len
        self.num_classes = num_classes
        self.input_len = input_len
        self.use_noise = use_noise
        self.use_forgetting = use_forgetting
        
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

        # Forgetting mechanisms (applied after backbone)
        if self.use_forgetting:
            self.forgetting_layer = ForgettingMechanisms(
                hidden_dim, forgetting_type, forgetting_rate
            )
        
        # Task-specific heads
        if self.task_name == "pretrain":
            # Reconstruction head
            self.decoder = nn.Linear(hidden_dim, input_len)
        
        elif self.task_name == "finetune" and self.pred_len is not None:
            head_width = min(2048, max(64, 4*int(self.pred_len)))

            # Enhanced forecasting head with forgetting
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
        
        # For pretrain and forecasting, use shared linear backbone
        if self.task_name == "pretrain" or (self.task_name == "finetune" and self.pred_len is not None):
            # Store clean signal
            x_clean = x.clone()
            
            # Add noise during pretrain if enabled
            if self.task_name == "pretrain" and add_noise_flag and self.use_noise:
                x, noise = self.add_noise(x, noise_level=noise_level)
            
            # Process each feature independently with linear backbone
            x = x.permute(0, 2, 1)  # [batch, num_features, seq_len]
            x_flat = x.reshape(batch_size * num_features, seq_len)  # [batch*features, seq_len]
            
            # Apply shared linear backbone
            features = self.linear_backbone(x_flat)  # [batch*features, hidden_dim]
            
            # Apply forgetting mechanisms during finetune
            if self.use_forgetting:
                is_finetune = (self.task_name == "finetune")
                features = self.forgetting_layer(features, is_finetune=is_finetune)
            
            if self.task_name == "pretrain":
                # Reconstruction (denoising)
                reconstruction = self.decoder(features)  # [batch*features, seq_len]
                reconstruction = reconstruction.view(batch_size, num_features, seq_len)
                reconstruction = reconstruction.permute(0, 2, 1)  # [batch, seq_len, num_features]
                return reconstruction
            
            else:  # Forecasting
                # Predict future values
                predictions = self.forecaster(features)  # [batch*features, pred_len]
                predictions = predictions.view(batch_size, num_features, self.pred_len)
                predictions = predictions.permute(0, 2, 1)  # [batch, pred_len, num_features]
                return predictions
            
        else:  # Classification - uses Conv architecture
            # No noise for classification
            x = x.transpose(1, 2)  # [batch, num_features, seq_len]
            
            x = F.relu(self.conv1(x))
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
        
        self.model = LightweightModel(
            in_channels=args.enc_in,
            out_channels=args.enc_in,
            task_name=self.task_name,
            pred_len=None,
            num_classes=self.num_classes,
            input_len=args.input_len,
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