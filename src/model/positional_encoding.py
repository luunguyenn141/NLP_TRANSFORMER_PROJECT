import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Tạo ma trận PE (max_len, d_model) chứa toàn số 0
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Công thức Sin/Cos theo đề bài
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # Ensure the positional encoding buffer is large enough for x
        seq_len = x.size(1)
        # self.pe shape: (1, max_len, d_model)
        if seq_len > self.pe.size(1):
            # Recreate positional encodings with the larger length on the same device
            max_len = seq_len
            d_model = self.pe.size(2)
            pe = torch.zeros(max_len, d_model, device=x.device)
            position = torch.arange(0, max_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=x.device) *
                                 (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            # Replace the buffer (keep name 'pe' for consistency)
            self.register_buffer('pe', pe)

        # Cộng Embedding của từ với Positional Encoding
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)