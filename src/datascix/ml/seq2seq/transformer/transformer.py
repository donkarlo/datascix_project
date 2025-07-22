import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, max_len=500):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.input_linear = nn.Linear(input_dim, d_model)

        self.positional_encoding = self._generate_positional_encoding(max_len, d_model)
        self.register_buffer("pe", self.positional_encoding)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_linear = nn.Linear(d_model, input_dim)

    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # shape: (1, max_len, d_model)

    def forward(self, x, p):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        total_len = seq_len + p

        # Project input to d_model
        x_proj = self.input_linear(x)

        # Prepare zero vectors for future steps
        future_padding = torch.zeros((batch_size, p, self.d_model), device=x.device)
        x_proj_padded = torch.cat([x_proj, future_padding], dim=1)  # shape: (batch, seq_len + p, d_model)

        # Add positional encoding
        x_with_pe = x_proj_padded + self.pe[:, :total_len, :]

        # Create causal mask to block future information
        attn_mask = torch.triu(torch.full((total_len, total_len), float('-inf'), device=x.device), diagonal=1)

        # Pass through transformer
        out = self.transformer_encoder(x_with_pe, mask=attn_mask)

        # Project back to input_dim
        predictions = self.output_linear(out[:, -p:, :])  # take only the last p steps
        return predictions
