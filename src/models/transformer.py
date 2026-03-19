import torch
import torch.nn as nn
import torch.nn.functional as f
import math
import os
from mamba_ssm import Mamba2, Mamba


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class GeneralModelTransformer(nn.Module):
    def __init__(self, input_dim=3, model_dim=8, output_dim=4, num_layers=4, max_seq_len=512):
        super(GeneralModelTransformer, self).__init__()
        self.model_dim = model_dim
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, max_seq_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=4, dim_feedforward=256, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_linear = nn.Linear(model_dim, output_dim)
        # self.conv = nn.Conv1d(3, 4, kernel_size=(3,), padding=1, bias=False)


    def forward(self, inp):
        z = inp.permute(2, 0, 1)  # Change input shape to (seq_len, batch, input_dim)
        z = f.relu(self.input_linear(z)) # Apply the input linear layer
        z = self.pos_encoder(z)   # Apply positional encoding
        z = self.transformer_encoder(z)  # Apply the Transformer
        z = self.output_linear(z)
        z = z.permute(1, 2, 0)
        # z = self.conv(z)
        return z


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(CustomTransformerEncoderLayer, self).__init__(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class GeneralModelTransformerWithAttention(nn.Module):
    def __init__(self, input_dim=3, model_dim=8, output_dim=4, num_layers=4, max_seq_len=512):
        super(GeneralModelTransformerWithAttention, self).__init__()
        self.model_dim = model_dim
        self.input_linear = nn.Linear(input_dim, model_dim)
        custom_encoder_layer = CustomTransformerEncoderLayer(d_model=model_dim, nhead=4, dim_feedforward=256, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(custom_encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(model_dim, output_dim)

    def forward(self, inp):
        z = inp.permute(2, 0, 1) 
        z = f.relu(self.input_linear(z)) 
        attn_maps = []
        for layer in self.transformer_encoder.layers:
            z, attn = layer(z)  
            attn_maps.append(attn)
        z = self.output_linear(z)
        z = z.permute(1, 2, 0)
        return z, attn_maps


class ClassficationTransformer(nn.Module):
    def __init__(self, input_dim=3, model_dim=8, output_dim=3, num_layers=2, max_seq_len=50):
        super(ClassficationTransformer, self).__init__()
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, max_seq_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=4, dim_feedforward=256, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)


    def forward(self, inp):
        z = inp.permute(2, 0, 1)  
        z = f.relu(self.input_linear(z)) 
        z = self.pos_encoder(z)  
        z = self.transformer_encoder(z)  
        z = z.mean(dim=0)
        z = self.fc(z)
        return z

class GeneralModelTransformerwithstep(nn.Module):
    def __init__(self, input_dim=3, model_dim=8, output_dim=4, num_layers=4, 
                 max_seq_len=51, output_seq_len=25):
        super(GeneralModelTransformerwithstep, self).__init__()
        self.model_dim = model_dim
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, max_seq_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=4, dim_feedforward=256, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_seq_len = output_seq_len

        self.length_adjust = nn.Conv1d(
            in_channels=model_dim, 
            out_channels=model_dim, 
            kernel_size=3, 
            stride=max_seq_len//output_seq_len,
            padding=0
        )
        self.output_linear = nn.Linear(model_dim, output_dim)

    def forward(self, inp):
        z = inp.permute(2, 0, 1)  # (seq_len, batch, input_dim)
        z = f.relu(self.input_linear(z))  
        z = self.pos_encoder(z)  
        z = self.transformer_encoder(z)  

        z = z.permute(1, 2, 0)  # (batch, model_dim, seq_len)
        z = self.length_adjust(z)  
        z = z.permute(2, 0, 1)  # (output_seq_len, batch, model_dim)

        z = self.output_linear(z)
        z = z.permute(1, 2, 0)  # (batch, output_dim, output_seq_len)
        return z



class MambaBlock(nn.Module):
    """
    Wrap one Mamba layer + residual + LayerNorm
    """
    def __init__(self, d_model: int,
                 d_state: int = 128,
                 expansion: int = 2,
                 conv_kernel: int = 4):
        super().__init__()
        self.layer = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=conv_kernel,
            expand=expansion,
            headdim=48
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        return self.norm(self.layer(x) + x)


class GeneralModelMamba(nn.Module):
    """
    Drop‑in replacement for GeneralModelTransformer.
    Keeps exactly the same I/O shapes.
    """
    def __init__(self,
                 input_dim:  int = 3,
                 model_dim:  int = 192,
                 output_dim: int = 4,
                 num_layers: int = 8,
                 conv_kernel: int = 4,
                 d_state: int = 64,
                 expansion: int = 2):
        super().__init__()

        self.input_linear  = nn.Linear(input_dim, model_dim)
        self.blocks        = nn.ModuleList([
            MambaBlock(model_dim, d_state, expansion, conv_kernel)
            for _ in range(num_layers)
        ])
        self.output_linear = nn.Linear(model_dim, output_dim)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        z = inp.permute(2, 0, 1)          # (L, B, C_in)
        z = f.relu(self.input_linear(z))  # (L, B, D)

        z = z.permute(1, 0, 2)            # (B, L, D)
        for blk in self.blocks:
            z = blk(z)                    # (B, L, D)

        z = self.output_linear(z)         # (B, L, C_out)
        z = z.permute(0, 2, 1)            # (B, C_out, L)
        return z
