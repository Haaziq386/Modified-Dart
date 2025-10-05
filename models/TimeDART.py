import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.TimeDART_EncDec import (
    ChannelIndependence,
    AddSosTokenAndDropLast,
    CausalTransformer,
    Diffusion,
    DenoisingPatchDecoder,
    DilatedConvEncoder,
    ClsEmbedding,
    ClsHead,
    OldClsHead,
    ClsFlattenHead,
    ARFlattenHead,
)
from layers.Embed import Patch, PatchEmbedding, PositionalEncoding, TokenEmbedding_TimeDART

def crop_or_pad(x, target_len):
    """Crop or pad x along the last dimension to match target_len."""
    current_len = x.size(-1)
    if current_len > target_len:
        x = x[..., :target_len]
    elif current_len < target_len:
        pad_size = target_len - current_len
        x = F.pad(x, (0, pad_size))
    return x

class UNet1D(nn.Module):
    """
    1D U-Net for time series data - works for both forecasting and classification
    """
    def __init__(self, in_channels, out_channels, task_name="pretrain", pred_len=None, num_classes=2):
        super(UNet1D, self).__init__()
        self.task_name = task_name
        self.pred_len = pred_len
        self.num_classes = num_classes
        
        # Encoder (Downsampling path)
        # Block 1: input_len -> input_len/2
        self.e11 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)  # [batch, in_channels, seq_len] -> [batch, 64, seq_len]
        self.e12 = nn.Conv1d(64, 64, kernel_size=3, padding=1)  # [batch, 64, seq_len] -> [batch, 64, seq_len]
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # [batch, 64, seq_len] -> [batch, 64, seq_len/2]

        # Block 2: seq_len/2 -> seq_len/4
        self.e21 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  # [batch, 64, seq_len/2] -> [batch, 128, seq_len/2]
        self.e22 = nn.Conv1d(128, 128, kernel_size=3, padding=1)  # [batch, 128, seq_len/2] -> [batch, 128, seq_len/2]
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # [batch, 128, seq_len/2] -> [batch, 128, seq_len/4]

        # Block 3: seq_len/4 -> seq_len/8
        self.e31 = nn.Conv1d(128, 256, kernel_size=3, padding=1)  # [batch, 128, seq_len/4] -> [batch, 256, seq_len/4]
        self.e32 = nn.Conv1d(256, 256, kernel_size=3, padding=1)  # [batch, 256, seq_len/4] -> [batch, 256, seq_len/4]
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # [batch, 256, seq_len/4] -> [batch, 256, seq_len/8]

        # Block 4: seq_len/8 -> seq_len/16
        self.e41 = nn.Conv1d(256, 512, kernel_size=3, padding=1)  # [batch, 256, seq_len/8] -> [batch, 512, seq_len/8]
        self.e42 = nn.Conv1d(512, 512, kernel_size=3, padding=1)  # [batch, 512, seq_len/8] -> [batch, 512, seq_len/8]
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)  # [batch, 512, seq_len/8] -> [batch, 512, seq_len/16]

        # Bottleneck
        self.e51 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)  # [batch, 512, seq_len/16] -> [batch, 1024, seq_len/16]
        self.e52 = nn.Conv1d(1024, 1024, kernel_size=3, padding=1)  # [batch, 1024, seq_len/16] -> [batch, 1024, seq_len/16]

        # Task-specific heads
        if self.task_name == "pretrain":
            # Decoder (Upsampling path) for reconstruction
            self.upconv1 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)  # [batch, 1024, seq_len/16] -> [batch, 512, seq_len/8]
            self.d11 = nn.Conv1d(1024, 512, kernel_size=3, padding=1)  # [batch, 1024, seq_len/8] -> [batch, 512, seq_len/8]
            self.d12 = nn.Conv1d(512, 512, kernel_size=3, padding=1)  # [batch, 512, seq_len/8] -> [batch, 512, seq_len/8]

            self.upconv2 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)  # [batch, 512, seq_len/8] -> [batch, 256, seq_len/4]
            self.d21 = nn.Conv1d(512, 256, kernel_size=3, padding=1)  # [batch, 512, seq_len/4] -> [batch, 256, seq_len/4]
            self.d22 = nn.Conv1d(256, 256, kernel_size=3, padding=1)  # [batch, 256, seq_len/4] -> [batch, 256, seq_len/4]

            self.upconv3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)  # [batch, 256, seq_len/4] -> [batch, 128, seq_len/2]
            self.d31 = nn.Conv1d(256, 128, kernel_size=3, padding=1)  # [batch, 256, seq_len/2] -> [batch, 128, seq_len/2]
            self.d32 = nn.Conv1d(128, 128, kernel_size=3, padding=1)  # [batch, 128, seq_len/2] -> [batch, 128, seq_len/2]

            self.upconv4 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)  # [batch, 128, seq_len/2] -> [batch, 64, seq_len]
            self.d41 = nn.Conv1d(128, 64, kernel_size=3, padding=1)  # [batch, 128, seq_len] -> [batch, 64, seq_len]
            self.d42 = nn.Conv1d(64, 64, kernel_size=3, padding=1)  # [batch, 64, seq_len] -> [batch, 64, seq_len]

            # Output layer for reconstruction
            self.outconv = nn.Conv1d(64, out_channels, kernel_size=1)  # [batch, 64, seq_len] -> [batch, out_channels, seq_len]
        
        elif self.task_name == "finetune" and self.pred_len is not None:
            # Forecasting head
            self.global_pool = nn.AdaptiveAvgPool1d(1)  # [batch, 1024, seq_len/16] -> [batch, 1024, 1]
            self.forecast_head = nn.Sequential(
                nn.Linear(1024, 512),  # [batch, 1024] -> [batch, 512]
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),  # [batch, 512] -> [batch, 256]
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, self.pred_len * out_channels)  # [batch, 256] -> [batch, pred_len * out_channels]
            )
        
        else:
            # Classification head
            self.global_pool = nn.AdaptiveAvgPool1d(1)  # [batch, 1024, seq_len/16] -> [batch, 1024, 1]
            self.classifier = nn.Sequential(
                nn.Linear(1024, 512),  # [batch, 1024] -> [batch, 512]
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),  # [batch, 512] -> [batch, 256]
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.num_classes)  # [batch, 256] -> [batch, num_classes]
            )

    def forward(self, x):
        # Input: [batch_size, seq_len, num_features]
        # Transpose for conv1d: [batch_size, num_features, seq_len]
        batch_size, seq_len, num_features = x.size()
        x = x.transpose(1, 2)
        
        # Encoder
        xe11 = F.relu(self.e11(x))  # [batch, num_features, seq_len] -> [batch, 64, seq_len]
        xe12 = F.relu(self.e12(xe11))  # [batch, 64, seq_len] -> [batch, 64, seq_len]
        xp1 = self.pool1(xe12)  # [batch, 64, seq_len] -> [batch, 64, seq_len/2]

        xe21 = F.relu(self.e21(xp1))  # [batch, 64, seq_len/2] -> [batch, 128, seq_len/2]
        xe22 = F.relu(self.e22(xe21))  # [batch, 128, seq_len/2] -> [batch, 128, seq_len/2]
        xp2 = self.pool2(xe22)  # [batch, 128, seq_len/2] -> [batch, 128, seq_len/4]

        xe31 = F.relu(self.e31(xp2))  # [batch, 128, seq_len/4] -> [batch, 256, seq_len/4]
        xe32 = F.relu(self.e32(xe31))  # [batch, 256, seq_len/4] -> [batch, 256, seq_len/4]
        xp3 = self.pool3(xe32)  # [batch, 256, seq_len/4] -> [batch, 256, seq_len/8]

        xe41 = F.relu(self.e41(xp3))  # [batch, 256, seq_len/8] -> [batch, 512, seq_len/8]
        xe42 = F.relu(self.e42(xe41))  # [batch, 512, seq_len/8] -> [batch, 512, seq_len/8]
        xp4 = self.pool4(xe42)  # [batch, 512, seq_len/8] -> [batch, 512, seq_len/16]

        # Bottleneck
        xe51 = F.relu(self.e51(xp4))  # [batch, 512, seq_len/16] -> [batch, 1024, seq_len/16]
        xe52 = F.relu(self.e52(xe51))  # [batch, 1024, seq_len/16] -> [batch, 1024, seq_len/16]
        
        if self.task_name == "pretrain":
            # Decoder with skip connections
            xu1 = self.upconv1(xe52)  # [batch, 1024, seq_len/16] -> [batch, 512, seq_len/8]
            xu1 = crop_or_pad(xu1, xe42.size(-1))  # Ensure same length
            xu11 = torch.cat([xu1, xe42], dim=1)  # [batch, 1024, seq_len/8]
            xd11 = F.relu(self.d11(xu11))  # [batch, 1024, seq_len/8] -> [batch, 512, seq_len/8]
            xd12 = F.relu(self.d12(xd11))  # [batch, 512, seq_len/8] -> [batch, 512, seq_len/8]

            xu2 = self.upconv2(xd12)  # [batch, 512, seq_len/8] -> [batch, 256, seq_len/4]
            xu2 = crop_or_pad(xu2, xe32.size(-1))  # Ensure same length
            xu22 = torch.cat([xu2, xe32], dim=1)  # [batch, 512, seq_len/4]
            xd21 = F.relu(self.d21(xu22))  # [batch, 512, seq_len/4] -> [batch, 256, seq_len/4]
            xd22 = F.relu(self.d22(xd21))  # [batch, 256, seq_len/4] -> [batch, 256, seq_len/4]

            xu3 = self.upconv3(xd22)  # [batch, 256, seq_len/4] -> [batch, 128, seq_len/2]
            xu3 = crop_or_pad(xu3, xe22.size(-1))  # Ensure same length
            xu33 = torch.cat([xu3, xe22], dim=1)  # [batch, 256, seq_len/2]
            xd31 = F.relu(self.d31(xu33))  # [batch, 256, seq_len/2] -> [batch, 128, seq_len/2]
            xd32 = F.relu(self.d32(xd31))  # [batch, 128, seq_len/2] -> [batch, 128, seq_len/2]

            xu4 = self.upconv4(xd32)  # [batch, 128, seq_len/2] -> [batch, 64, seq_len]
            xu4 = crop_or_pad(xu4, xe12.size(-1))
            xu44 = torch.cat([xu4, xe12], dim=1)  # [batch, 128, seq_len]
            xd41 = F.relu(self.d41(xu44))  # [batch, 128, seq_len] -> [batch, 64, seq_len]
            xd42 = F.relu(self.d42(xd41))  # [batch, 64, seq_len] -> [batch, 64, seq_len]

            # Output layer
            out = self.outconv(xd42)  # [batch, 64, seq_len] -> [batch, num_features, seq_len]
            
            # Transpose back to original format: [batch_size, seq_len, num_features]
            out = out.transpose(1, 2)
            return out
            
        elif self.task_name == "finetune" and self.pred_len is not None:
            # Forecasting
            pooled = self.global_pool(xe52)  # [batch, 1024, seq_len/16] -> [batch, 1024, 1]
            pooled = pooled.squeeze(-1)  # [batch, 1024, 1] -> [batch, 1024]
            out = self.forecast_head(pooled)  # [batch, 1024] -> [batch, pred_len * num_features]
            
            # Reshape to [batch_size, pred_len, num_features]
            out = out.view(batch_size, self.pred_len, num_features)
            return out
            
        else:  # Classification
            # Global pooling and classification
            pooled = self.global_pool(xe52)  # [batch, 1024, seq_len/16] -> [batch, 1024, 1]
            pooled = pooled.squeeze(-1)  # [batch, 1024, 1] -> [batch, 1024]
            out = self.classifier(pooled)  # [batch, 1024] -> [batch, num_classes]
            return out


class FlattenHead(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        pred_len: int,
        dropout: float,
    ):
        super(FlattenHead, self).__init__()
        self.pred_len = pred_len
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecast_head = nn.Linear(seq_len * d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, num_features, seq_len, d_model]
        :return: [batch_size, pred_len, num_features]
        """
        x = self.flatten(x)  # (batch_size, num_features, seq_len * d_model)
        x = self.forecast_head(x)  # (batch_size, num_features, pred_len)
        x = self.dropout(x)  # (batch_size, num_features, pred_len)
        x = x.permute(0, 2, 1)  # (batch_size, pred_len, num_features)
        return x


class Model(nn.Module):
    """
    TimeDART with U-Net architecture
    """

    def __init__(self, args):
        super(Model, self).__init__()
        self.input_len = args.input_len
        self.pred_len = args.pred_len
        self.task_name = args.task_name
        self.use_norm = args.use_norm
        
        # Initialize UNet with proper parameters
        self.unet = UNet1D(
            in_channels=args.enc_in,
            out_channels=args.enc_in,  # For reconstruction or forecasting same num features
            task_name=self.task_name,
            pred_len=self.pred_len if hasattr(args, 'pred_len') else None,
            num_classes=getattr(args, 'num_classes', 2)
        )

    def pretrain(self, x):
        # [batch_size, input_len, num_features]
        batch_size, input_len, num_features = x.size()
        
        # Optional normalization
        if self.use_norm:
            # Instance Normalization
            means = torch.mean(
                x, dim=1, keepdim=True
            ).detach()  # [batch_size, 1, num_features], detach from gradient
            x = x - means  # [batch_size, input_len, num_features]
            stdevs = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()  # [batch_size, 1, num_features]
            x = x / stdevs  # [batch_size, input_len, num_features]

        # Forward through UNet
        predict_x = self.unet(x)  # [batch_size, input_len, num_features]

        # Instance Denormalization
        if self.use_norm:
            predict_x = predict_x * (stdevs[:, 0, :].unsqueeze(1)).repeat(
                1, input_len, 1
            )  # [batch_size, input_len, num_features]
            predict_x = predict_x + (means[:, 0, :].unsqueeze(1)).repeat(
                1, input_len, 1
            )  # [batch_size, input_len, num_features]

        return predict_x

    def forecast(self, x):
        # [batch_size, input_len, num_features]
        batch_size, input_len, num_features = x.size()
        
        # Optional normalization
        if self.use_norm:
            means = torch.mean(x, dim=1, keepdim=True).detach()  # [batch_size, 1, num_features]
            x = x - means  # [batch_size, input_len, num_features]
            stdevs = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()  # [batch_size, 1, num_features]
            x = x / stdevs  # [batch_size, input_len, num_features]

        # Forward through UNet
        prediction = self.unet(x)  # [batch_size, pred_len, num_features]

        # Denormalization
        if self.use_norm:
            prediction = prediction * (stdevs[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)  # [batch_size, pred_len, num_features]
            prediction = prediction + (means[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)  # [batch_size, pred_len, num_features]

        return prediction
    
    def forward(self, batch_x):
        if self.task_name == "pretrain":
            return self.pretrain(batch_x)
        elif self.task_name == "finetune":
            dec_out = self.forecast(batch_x)
            return dec_out  # [batch_size, pred_len, num_features]
        else:
            raise ValueError("task_name should be 'pretrain' or 'finetune'")


class ClsModel(nn.Module):
    def __init__(self, args):
        super(ClsModel, self).__init__()
        self.input_len = args.input_len
        self.task_name = args.task_name
        self.num_classes = args.num_classes
        
        # Initialize UNet with proper parameters
        self.unet = UNet1D(
            in_channels=args.enc_in,
            out_channels=args.enc_in,
            task_name=self.task_name,
            pred_len=None,  # Not needed for classification
            num_classes=self.num_classes
        )

    def pretrain(self, x):
        # [batch_size, input_len, num_features]
        return self.unet(x)  # [batch_size, input_len, num_features]

    def forecast(self, x):
        # [batch_size, input_len, num_features]
        return self.unet(x)  # [batch_size, num_classes]
    
    def forward(self, batch_x):
        if self.task_name == "pretrain":
            return self.pretrain(batch_x)
        elif self.task_name == "finetune":
            return self.forecast(batch_x)
        else:
            raise ValueError("task_name should be 'pretrain' or 'finetune'")