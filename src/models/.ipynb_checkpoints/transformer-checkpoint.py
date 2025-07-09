import torch
import torch.nn as nn
import torch.nn.functional as f

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
        # self.pos_encoder = PositionalEncoding(model_dim, max_seq_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=4, dim_feedforward=256, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_linear = nn.Linear(model_dim, output_dim)
        # self.conv = nn.Conv1d(3, 4, kernel_size=(3,), padding=1, bias=False)


    def forward(self, inp):
        z = inp.permute(2, 0, 1)  # Change input shape to (seq_len, batch, input_dim)
        z = f.relu(self.input_linear(z)) # Apply the input linear layer
        # z = self.pos_encoder(z)   # Apply positional encoding
        z = self.transformer_encoder(z)  # Apply the Transformer
        z = self.output_linear(z)
        z = z.permute(1, 2, 0)
        # z = self.conv(z)
        return z
        