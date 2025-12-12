import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.self_attn_norm = nn.LayerNorm(d_model)
        
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # --- PRE-NORM CHANGE ---

        # 1. Masked Self Attention
        trg_norm = self.self_attn_norm(trg) # Norm trước
        _trg, _ = self.self_attn(trg_norm, trg_norm, trg_norm, trg_mask)
        trg = trg + self.dropout(_trg) # Add vào gốc
        
        # 2. Cross Attention
        trg_norm = self.cross_attn_norm(trg) # Norm trước
        # Lưu ý: enc_src thường đã được Norm ở Encoder rồi nên không cần Norm lại, 
        # nhưng Query (trg_norm) thì bắt buộc phải lấy từ bước Norm này.
        _trg, _ = self.cross_attn(trg_norm, enc_src, enc_src, src_mask)
        trg = trg + self.dropout(_trg) # Add vào gốc
        
        # 3. Feed Forward
        trg_norm = self.ffn_norm(trg) # Norm trước
        _trg = self.ffn(trg_norm)
        trg = trg + self.dropout(_trg) # Add vào gốc
        
        return trg

class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, n_layers, n_heads, d_ff, dropout, max_len=5000):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(output_dim, d_model)
        
        # Import cục bộ để tránh lỗi vòng lặp (nếu có)
        from .positional_encoding import PositionalEncoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # --- PRE-NORM CHANGE: Thêm lớp Norm cuối cùng ---
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.embedding(trg) * (self.d_model ** 0.5)
        trg = self.pos_encoding(trg)
        trg = self.dropout(trg)
        
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        
        # --- PRE-NORM CHANGE: Norm trước khi vào Linear ---
        trg = self.final_norm(trg)
        
        output = self.fc_out(trg)
        return output