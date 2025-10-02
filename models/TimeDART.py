import torch
import torch.nn as nn
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
    TimeDART
    """

    def __init__(self, args):
        super(Model, self).__init__()
        self.input_len = args.input_len

        # For Model Hyperparameters
        self.d_model = args.d_model
        self.num_heads = args.n_heads
        self.feedforward_dim = args.d_ff
        self.dropout = args.dropout
        self.device = args.device
        self.task_name = args.task_name
        self.pred_len = args.pred_len
        self.use_norm = args.use_norm
        self.channel_independence = ChannelIndependence()

        # Patch
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.patch = Patch(
            patch_len=self.patch_len,
            stride=self.stride,
        )
        self.seq_len = int((self.input_len - self.patch_len) / self.stride) + 1

        # Embedding
        self.enc_embedding = PatchEmbedding(
            patch_len=self.patch_len,
            d_model=self.d_model,
        )

        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            dropout=self.dropout,
        )

        sos_token = torch.randn(1, 1, self.d_model, device=self.device)
        self.sos_token = nn.Parameter(sos_token, requires_grad=True)

        self.add_sos_token_and_drop_last = AddSosTokenAndDropLast(
            sos_token=self.sos_token,
        )

        # Encoder (Casual Trasnformer)
        self.diffusion = Diffusion(
            time_steps=args.time_steps,
            device=self.device,
            scheduler=args.scheduler,
        )
        self.encoder = CausalTransformer(
            d_model=args.d_model,
            num_heads=args.n_heads,
            feedforward_dim=args.d_ff,
            dropout=args.dropout,
            num_layers=args.e_layers,
        )
        # self.encoder = DilatedConvEncoder(
        #     in_channels=self.d_model,
        #     channels=[self.d_model] * args.e_layers,
        #     kernel_size=3,
        # )

        # Create shared MLP for both pretrain and forecast
        patch_input_size = self.seq_len * self.patch_len  # seq_len * patch_len
        hidden_dim = 512

        self.shared_mlp = nn.Sequential(
            nn.Linear(patch_input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, patch_input_size)  # For pretrain: reconstruct patches
        )

        # Add forecasting head
        self.forecast_head = nn.Linear(hidden_dim, self.pred_len)

        # Decoder
        if self.task_name == "pretrain":
            self.denoising_patch_decoder = DenoisingPatchDecoder(
                d_model=args.d_model,
                num_layers=args.d_layers,
                num_heads=args.n_heads,
                feedforward_dim=args.d_ff,
                dropout=args.dropout,
                mask_ratio=args.mask_ratio,
            )

            self.projection = FlattenHead(
                seq_len=self.seq_len,
                d_model=self.d_model,
                pred_len=args.input_len,
                dropout=args.head_dropout,
            )
            # self.projection = ARFlattenHead(
            #     d_model=self.d_model,
            #     patch_len=self.patch_len,
            #     dropout=args.head_dropout,
            # )

        elif self.task_name == "finetune":
            self.head = FlattenHead(
                seq_len=self.seq_len,
                d_model=args.d_model,
                pred_len=args.pred_len,
                dropout=args.head_dropout,
            )

    def pretrain(self, x):
        # [batch_size, input_len, num_features]
        batch_size, input_len, num_features = x.size()
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

        # Channel Independence
        x = self.channel_independence(x)  # [batch_size * num_features, input_len, 1]
        # Patch
        x_patch = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]

        # Simple MLP
        x_patch_flat = x_patch.view(x_patch.size(0), -1)  # [batch_size * num_features, seq_len * patch_len]


        # Process through MLP
        predict_x_flat = self.shared_mlp(x_patch_flat)  # [batch_size * num_features, seq_len * patch_len]


        # Reshape back to patch format then to final output format
        predict_x_patch = predict_x_flat.view(x_patch.size())  # [batch_size * num_features, seq_len, patch_len]
        
        # Reconstruct original format (unpatch)
        predict_x = predict_x_patch.view(batch_size, num_features, -1)  # [batch_size, num_features, seq_len * patch_len]
        # Trim to input_len if needed
        predict_x = predict_x[:, :, :input_len]  # [batch_size, num_features, input_len]
        predict_x = predict_x.permute(0, 2, 1)  # [batch_size, input_len, num_features]

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
        batch_size, _, num_features = x.size()
        if self.use_norm:
            means = torch.mean(x, dim=1, keepdim=True).detach()
            x = x - means
            stdevs = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x = x / stdevs

        # Channel Independence
        x = self.channel_independence(x)  # [batch_size * num_features, input_len, 1]
        # Patch
        x_patch = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]

        # Simple MLP for forecasting
        x_patch_flat = x_patch.view(x_patch.size(0), -1)  # [batch_size * num_features, seq_len * patch_len]

        # Use the first two layers of the shared MLP (feature extraction)
        features = x_patch_flat
        for layer in self.shared_mlp[:3]:  # First Linear + ReLU + Second Linear
            features = layer(features)
        
        # Use forecasting head for final prediction
        pred_per_feature = self.forecast_head(features)  # [batch_size * num_features, pred_len]
        
        # Reshape to final format
        x = pred_per_feature.view(batch_size, num_features, self.pred_len)  # [batch_size, num_features, pred_len]
        x = x.permute(0, 2, 1)  # [batch_size, pred_len, num_features]

        # denormalization
        if self.use_norm:
            x = x * (stdevs[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)
            x = x + (means[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)

        return x
    
    def forward(self, batch_x):

        if self.task_name == "pretrain":
            return self.pretrain(batch_x)
        elif self.task_name == "finetune":
            dec_out = self.forecast(batch_x)
            return dec_out[:, -self.pred_len: , :]
        else:
            raise ValueError("task_name should be 'pretrain' or 'finetune'")


class ClsModel(nn.Module):
    def __init__(self, args):
        super(ClsModel, self).__init__()
        self.input_len = args.input_len

        # For Model Hyperparameters
        self.d_model = args.d_model
        self.num_heads = args.n_heads
        self.feedforward_dim = args.d_ff
        self.dropout = args.dropout
        self.device = args.device
        self.task_name = args.task_name
        self.num_classes = args.num_classes

        # Patch
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.seq_len = int((self.input_len - self.patch_len) / self.stride) + 2
        padding = self.seq_len * self.stride - self.input_len

        # Embedding
        self.enc_embedding = ClsEmbedding(
            num_features=args.enc_in,
            d_model=args.d_model,
            kernel_size=args.patch_len,
            stride=args.stride,
            padding=padding,
        )

        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            dropout=self.dropout,
        )

        sos_token = torch.randn(1, 1, self.d_model, device=self.device)
        self.sos_token = nn.Parameter(sos_token, requires_grad=True)

        self.add_sos_token_and_drop_last = AddSosTokenAndDropLast(
            sos_token=self.sos_token,
        )

        # Encoder (Casual Trasnformer)
        self.diffusion = Diffusion(
            time_steps=args.time_steps,
            device=self.device,
            scheduler=args.scheduler,
        )
        self.encoder = CausalTransformer(
            d_model=args.d_model,
            num_heads=args.n_heads,
            feedforward_dim=args.d_ff,
            dropout=args.dropout,
            num_layers=args.e_layers,
        )
        # self.encoder = DilatedConvEncoder(
        #     in_channels=self.d_model,
        #     channels=[self.d_model] * args.e_layers,
        #     kernel_size=3,
        # )

        # Decoder
        # if self.task_name == "pretrain":
        #     self.denoising_patch_decoder = DenoisingPatchDecoder(
        #         d_model=args.d_model,
        #         num_layers=args.d_layers,
        #         num_heads=args.n_heads,
        #         feedforward_dim=args.d_ff,
        #         dropout=args.dropout,
        #         mask_ratio=args.mask_ratio,
        #     )

        #     self.projection = ClsFlattenHead(
        #         seq_len=self.seq_len,
        #         d_model=self.d_model,
        #         pred_len=args.input_len,
        #         num_features=args.c_out,
        #         dropout=args.head_dropout,
        #     )

        # elif self.task_name == "finetune":
        #     self.head = OldClsHead(
        #         seq_len=self.seq_len,
        #         d_model=args.d_model,
        #         num_classes=args.num_classes,
        #         dropout=args.head_dropout,
        #     )
        # Create shared MLP for both pretrain and forecast
        input_size = args.input_len * args.enc_in  # input_len * num_features

        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        if self.task_name == "pretrain":
            self.reconstruction_head = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(512, input_size)
            )
        else:
            self.cls_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(128, args.num_classes)
            )
        # Add classification head for forecast
        # self.cls_head = nn.Linear(1024, args.num_classes)

    def pretrain(self, x):
        # [batch_size, input_len, num_features]
        # Instance Normalization
        # batch_size, input_len, num_features = x.size()
        # means = torch.mean(
        #     x, dim=1, keepdim=True
        # ).detach()  # [batch_size, 1, num_features], detach from gradient
        # x = x - means  # [batch_size, input_len, num_features]
        # stdevs = torch.sqrt(
        #     torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        # ).detach()  # [batch_size, 1, num_features]
        # x = x / stdevs  # [batch_size, input_len, num_features]

        # Simple MLP
        # [batch_size, input_len, num_features]
        batch_size, input_len, num_features = x.size()
        
        # Flatten input for MLP processing
        x_flat = x.view(batch_size, -1)  # [batch_size, input_len * num_features]

        features = self.shared_layers(x_flat)
        # Process through shared MLP
        predict_x_flat = self.reconstruction_head(features)  # [batch_size, input_len * num_features]

        # Reshape to final output format
        predict_x = predict_x_flat.view(batch_size, input_len, num_features)  # [batch_size, input_len, num_features]

        return predict_x

        # Instance Denormalization
        # predict_x = predict_x * (stdevs[:, 0, :].unsqueeze(1)).repeat(
        #     1, input_len, 1
        # )  # [batch_size, input_len, num_features]
        # predict_x = predict_x + (means[:, 0, :].unsqueeze(1)).repeat(
        #     1, input_len, 1
        # )  # [batch_size, input_len, num_features]

        # return predict_x

    def forecast(self, x):
        # batch_size, _, num_features = x.size()
        # means = torch.mean(x, dim=1, keepdim=True).detach()
        # x = x - means
        # stdevs = torch.sqrt(
        #     torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        # ).detach()
        # x = x / stdevs

        batch_size, input_len, num_features = x.size()
        
        # Flatten input for MLP processing
        x_flat = x.view(batch_size, -1)  # [batch_size, input_len * num_features]

        # Use the first two layers of the shared MLP (feature extraction)
        features = self.shared_layers(x_flat)
        
        # Use classification head for final prediction
        x = self.cls_head(features)  # [batch_size, num_classes]

        return x
    
    def forward(self, batch_x):

        if self.task_name == "pretrain":
            return self.pretrain(batch_x)
        elif self.task_name == "finetune":
            return self.forecast(batch_x)
        else:
            raise ValueError("task_name should be 'pretrain' or 'finetune'")