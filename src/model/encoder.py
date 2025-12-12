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
        # --- PRE-NORM CHANGE ---
        
        # 1. Self Attention Block
        # B1: Norm trước
        src_norm = self.self_attn_norm(src) 
        # B2: Tính Attention trên input đã norm
        _src, _ = self.self_attn(src_norm, src_norm, src_norm, src_mask)
        # B3: Residual Add vào input gốc (chưa norm)
        src = src + self.dropout(_src)
        
        # 2. FFN Block
        # B1: Norm trước
        src_norm = self.ffn_norm(src)
        # B2: Tính FFN
        _src = self.ffn(src_norm)
        # B3: Residual Add
        src = src + self.dropout(_src)
        
        return src

class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_heads, d_ff, dropout, max_len=5000):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # --- PRE-NORM CHANGE: Thêm lớp Norm cuối cùng ---
        self.final_norm = nn.LayerNorm(d_model) 
        
    def forward(self, src, src_mask):
        src = self.embedding(src) * (self.d_model ** 0.5)
        src = self.pos_encoding(src)
        
        for layer in self.layers:   
            src = layer(src, src_mask)
            
        # --- PRE-NORM CHANGE: Norm output cuối cùng ---
        src = self.final_norm(src)
        
        return src