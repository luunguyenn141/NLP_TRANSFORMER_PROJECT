import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .positional_encoding import PositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        
        # 1. Multi-Head Self Attention
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.self_attn_norm = nn.LayerNorm(d_model)
        
        # 2. Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        # src: [batch_size, src_len, d_model]
        # src_mask: [batch_size, 1, 1, src_len]
        
        # Apply Self Attention
        # _src là kết quả attention, _ là attention weights (không dùng ở đây)
        _src, _ = self.self_attn(src, src, src, src_mask)
        
        # Dropout + Add + Norm (Residual Connection)
        src = self.self_attn_norm(src + self.dropout(_src))
        
        # Apply Feed Forward
        _src = self.ffn(src)
        
        # Dropout + Add + Norm
        src = self.ffn_norm(src + self.dropout(_src))
        
        return src

class Encoder(nn.Module):
    # --- ĐÂY LÀ PHẦN BẠN ĐANG BỊ LỖI ---
    # Phải khai báo đầy đủ tham số để khớp với Transformer
    def __init__(self, input_dim, d_model, n_layers, n_heads, d_ff, dropout, max_len=5000):
        super().__init__()
        
        self.d_model = d_model
        
        # Embedding Layer
        self.embedding = nn.Embedding(input_dim, d_model)
        
        # Positional Encoding Layer
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Stack N lớp EncoderLayer
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        # src: [batch_size, src_len]
        
        # 1. Embedding + Scaling
        # Trong paper gốc, họ nhân embedding với sqrt(d_model)
        src = self.embedding(src) * (self.d_model ** 0.5)
        
        # 2. Cộng Positional Encoding
        src = self.pos_encoding(src)
        
        # 3. Qua các lớp Encoder Layers
        for layer in self.layers:
            src = layer(src, src_mask)
            
        return src