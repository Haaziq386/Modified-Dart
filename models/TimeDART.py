import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightModel(nn.Module):
    """
    Lightweight model with task-specific architectures:
    - Conv1D for classification (needs spatial features)
    - Direct MLP for forecasting (simpler is better)
    """
    def __init__(self, in_channels, out_channels, task_name="pretrain", pred_len=None, num_classes=2, input_len=336):
        super(LightweightModel, self).__init__()
        self.task_name = task_name
        self.pred_len = pred_len
        self.num_classes = num_classes
        self.input_len = input_len
        
        # Task-specific architectures
        if self.task_name == "pretrain":
            # Use Conv for pretrain (works for both datasets)
            self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.dropout = nn.Dropout(0.2)
            
            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv1d(64, out_channels, kernel_size=1)
            )
        
        elif self.task_name == "finetune" and self.pred_len is not None:
            # PURE MLP for forecasting (no Conv!)
            # Simple channel-wise processing
            hidden_dim = 512
            self.forecaster = nn.Sequential(
                nn.Linear(input_len, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, pred_len)
            )
        
        else:
            # Conv for classification (needs spatial features)
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

    def forward(self, x):
        # Input: [batch_size, seq_len, num_features]
        batch_size, seq_len, num_features = x.size()
        
        if self.task_name == "pretrain":
            # Conv-based reconstruction
            x = x.transpose(1, 2)  # [batch, num_features, seq_len]
            
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = self.dropout(x)
            
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = self.dropout(x)
            
            x = F.relu(self.conv3(x))
            
            x = self.decoder(x)
            if x.size(-1) != seq_len:
                x = F.adaptive_avg_pool1d(x, seq_len)
            
            x = x.transpose(1, 2)
            return x
            
        elif self.task_name == "finetune" and self.pred_len is not None:
            # PURE MLP forecasting (NO CONV!)
            # Process each feature independently
            # x: [batch, seq_len, num_features]
            
            # Reshape for channel-wise processing
            x = x.permute(0, 2, 1)  # [batch, num_features, seq_len]
            x = x.reshape(batch_size * num_features, seq_len)  # [batch*features, seq_len]
            
            # Apply MLP
            predictions = self.forecaster(x)  # [batch*features, pred_len]
            
            # Reshape back
            predictions = predictions.view(batch_size, num_features, self.pred_len)  # [batch, features, pred_len]
            predictions = predictions.permute(0, 2, 1)  # [batch, pred_len, features]
            
            return predictions
            
        else:  # Classification
            # Conv-based classification
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
    TimeDART with task-adaptive architecture
    """

    def __init__(self, args):
        super(Model, self).__init__()
        self.input_len = args.input_len
        self.pred_len = args.pred_len
        self.task_name = args.task_name
        self.use_norm = args.use_norm
        
        self.model = LightweightModel(
            in_channels=args.enc_in,
            out_channels=args.enc_in,
            task_name=self.task_name,
            pred_len=self.pred_len if hasattr(args, 'pred_len') else None,
            num_classes=getattr(args, 'num_classes', 2),
            input_len=args.input_len
        )

    def pretrain(self, x):
        batch_size, input_len, num_features = x.size()
        
        if self.use_norm:
            means = torch.mean(x, dim=1, keepdim=True).detach()
            x = x - means
            stdevs = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = x / stdevs

        predict_x = self.model(x)

        if self.use_norm:
            predict_x = predict_x * stdevs
            predict_x = predict_x + means

        return predict_x

    def forecast(self, x):
        batch_size, input_len, num_features = x.size()
        
        if self.use_norm:
            means = torch.mean(x, dim=1, keepdim=True).detach()
            x = x - means
            stdevs = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = x / stdevs

        prediction = self.model(x)

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
        
        self.model = LightweightModel(
            in_channels=args.enc_in,
            out_channels=args.enc_in,
            task_name=self.task_name,
            pred_len=None,
            num_classes=self.num_classes,
            input_len=args.input_len
        )

    def pretrain(self, x):
        return self.model(x)

    def forecast(self, x):
        return self.model(x)
    
    def forward(self, batch_x):
        if self.task_name == "pretrain":
            return self.pretrain(batch_x)
        elif self.task_name == "finetune":
            return self.forecast(batch_x)
        else:
            raise ValueError("task_name should be 'pretrain' or 'finetune'")