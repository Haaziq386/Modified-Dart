import torch
import torch.nn as nn
import torch.nn.functional as F
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np

class SeasonalTrendDecomposer:
    """
    Decompose time series into seasonal and trend components
    """
    def __init__(self, period=24, model='additive'):
        self.period = period
        self.model = model
    
    def decompose(self, x):
        """
        x: [batch_size, seq_len, num_features]
        Returns: trend, seasonal components
        """
        batch_size, seq_len, num_features = x.shape
        trend_components = []
        seasonal_components = []
        
        for b in range(batch_size):
            batch_trend = []
            batch_seasonal = []

            for f in range(num_features):
                series = x[b, :, f].cpu().numpy()
                
                # Handle series too short for decomposition
                if len(series) < 2 * self.period:
                    # Simple moving average for trend
                    trend = pd.Series(series).rolling(window=min(self.period, len(series)//2), center=True).mean()
                    trend = trend.fillna(method='bfill').fillna(method='ffill').values
                    seasonal = series - trend
                else:
                    try:
                        decomposition = seasonal_decompose(
                            series, 
                            model=self.model, 
                            period=self.period,
                            extrapolate_trend='freq'
                        )
                        trend = decomposition.trend
                        seasonal = decomposition.seasonal
                    except:
                        # Fallback to simple decomposition
                        trend = pd.Series(series).rolling(window=self.period, center=True).mean()
                        trend = trend.fillna(method='bfill').fillna(method='ffill').values
                        seasonal = series - trend
                
                batch_trend.append(trend)
                batch_seasonal.append(seasonal)
            
            trend_components.append(batch_trend)
            seasonal_components.append(batch_seasonal)
        
        trend_tensor = torch.tensor(trend_components, dtype=x.dtype, device=x.device)
        seasonal_tensor = torch.tensor(seasonal_components, dtype=x.dtype, device=x.device)
        
        return trend_tensor, seasonal_tensor

class LightweightModel(nn.Module):
    """
    Lightweight model with seasonal-trend decomposition
    """
    def __init__(self, in_channels, out_channels, task_name="pretrain", pred_len=None, 
                 num_classes=2, input_len=336, use_noise=False, use_decomposition=False, period=24):
        super(LightweightModel, self).__init__()
        self.task_name = task_name
        self.pred_len = pred_len
        self.num_classes = num_classes
        self.input_len = input_len
        self.use_noise = use_noise
        self.use_decomposition = use_decomposition
        
        # Decomposer
        if self.use_decomposition:
            self.decomposer = SeasonalTrendDecomposer(period=period)
        
        # SHARED LINEAR BACKBONE for pretrain and forecasting
        hidden_dim = 512

        if self.use_decomposition and (task_name == "pretrain" or (task_name == "finetune" and pred_len is not None)):
            # Separate backbones for trend and seasonal
            self.trend_backbone = nn.Sequential(
                nn.Linear(input_len, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            self.seasonal_backbone = nn.Sequential(
                nn.Linear(input_len, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            # Original shared backbone
            self.linear_backbone = nn.Sequential(
                nn.Linear(input_len, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        
        # Task-specific heads
        if self.task_name == "pretrain":
            if self.use_decomposition:
                # Separate decoders for trend and seasonal
                self.trend_decoder = nn.Linear(hidden_dim, input_len)
                self.seasonal_decoder = nn.Linear(hidden_dim, input_len)
            else:
                self.decoder = nn.Linear(hidden_dim, input_len)
        
        elif self.task_name == "finetune" and self.pred_len is not None:
            if self.use_decomposition:
                # Separate forecasters for trend and seasonal
                self.trend_forecaster = nn.Linear(hidden_dim, pred_len)
                self.seasonal_forecaster = nn.Linear(hidden_dim, pred_len)
            else:
                self.forecaster = nn.Linear(hidden_dim, pred_len)
        
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
        
        # For pretrain and forecasting with decomposition
        if (self.task_name == "pretrain" or (self.task_name == "finetune" and self.pred_len is not None)) and self.use_decomposition:
            
            # Decompose into trend and seasonal
            trend, seasonal = self.decomposer.decompose(x)
            
           # Add noise if enabled
            if self.task_name == "pretrain" and add_noise_flag and self.use_noise:
                trend, _ = self.add_noise(trend, noise_level=noise_level)
                seasonal, _ = self.add_noise(seasonal, noise_level=noise_level)
            
            # Process trend component
            trend = trend.permute(0, 2, 1)  # [batch, num_features, seq_len]
            trend_flat = trend.reshape(batch_size * num_features, seq_len)
            trend_features = self.trend_backbone(trend_flat)  # [batch*features, hidden_dim]
            
            # Process seasonal component
            seasonal = seasonal.permute(0, 2, 1)  # [batch, num_features, seq_len]
            seasonal_flat = seasonal.reshape(batch_size * num_features, seq_len)
            seasonal_features = self.seasonal_backbone(seasonal_flat)  # [batch*features, hidden_dim]
            
            if self.task_name == "pretrain":
                # Reconstruct trend and seasonal separately
                trend_recon = self.trend_decoder(trend_features)  # [batch*features, seq_len]
                seasonal_recon = self.seasonal_decoder(seasonal_features)  # [batch*features, seq_len]
                
                # Reshape and combine
                trend_recon = trend_recon.view(batch_size, num_features, seq_len).permute(0, 2, 1)
                seasonal_recon = seasonal_recon.view(batch_size, num_features, seq_len).permute(0, 2, 1)
                
                reconstruction = trend_recon + seasonal_recon
                return reconstruction
            
            else:  # Forecasting
                # Predict trend and seasonal separately
                trend_pred = self.trend_forecaster(trend_features)  # [batch*features, pred_len]
                seasonal_pred = self.seasonal_forecaster(seasonal_features)  # [batch*features, pred_len]
                
                # Reshape and combine
                trend_pred = trend_pred.view(batch_size, num_features, self.pred_len).permute(0, 2, 1)
                seasonal_pred = seasonal_pred.view(batch_size, num_features, self.pred_len).permute(0, 2, 1)
                
                predictions = trend_pred + seasonal_pred
                return predictions
            
        # Original processing (no decomposition or classification)
        elif self.task_name == "pretrain" or (self.task_name == "finetune" and self.pred_len is not None):
            # ...existing code for non-decomposition path...
            x_clean = x.clone()
            
            if self.task_name == "pretrain" and add_noise_flag and self.use_noise:
                x, noise = self.add_noise(x, noise_level=noise_level)
            
            x = x.permute(0, 2, 1)
            x_flat = x.reshape(batch_size * num_features, seq_len)
            features = self.linear_backbone(x_flat)
            
            if self.task_name == "pretrain":
                reconstruction = self.decoder(features)
                reconstruction = reconstruction.view(batch_size, num_features, seq_len)
                reconstruction = reconstruction.permute(0, 2, 1)
                return reconstruction
            else:
                predictions = self.forecaster(features)
                predictions = predictions.view(batch_size, num_features, self.pred_len)
                predictions = predictions.permute(0, 2, 1)
                return predictions
        
        else:  # Classification - unchanged
            x = x.transpose(1, 2)
            
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
    TimeDART with optional seasonal-trend decomposition
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.input_len = args.input_len
        self.pred_len = args.pred_len
        self.task_name = args.task_name
        self.use_norm = args.use_norm
        self.use_noise = getattr(args, 'use_noise', False)
        self.noise_level = getattr(args, 'noise_level', 0.1)
        self.use_decomposition = getattr(args, 'use_decomposition', False)
        self.period = getattr(args, 'period', 24)
        
        self.model = LightweightModel(
            in_channels=args.enc_in,
            out_channels=args.enc_in,
            task_name=self.task_name,
            pred_len=self.pred_len if hasattr(args, 'pred_len') else None,
            num_classes=getattr(args, 'num_classes', 2),
            input_len=args.input_len,
            use_noise=self.use_noise,
            use_decomposition=self.use_decomposition,
            period=self.period
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