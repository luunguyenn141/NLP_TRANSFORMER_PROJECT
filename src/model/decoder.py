import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        
        #Masked Multi-Head-Self Attention
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.self_attn_norm = nn.LayerNorm(d_model)
        
        #Multi-head Cross-Attention
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        
        # 3. Feed Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg: inputs từ decoder (batch_size, trg_len, d_model)
        # enc_src: outputs từ encoder (batch_size, src_len, d_model)
        
        #Masked Self Attention
        _trg, _ = self.self_attn(trg, trg, trg, trg_mask)
        trg = self.self_attn_norm(trg + self.dropout(_trg))
        
        # Cross Attention
        # Query = trg, Key = enc_src, Value = enc_src
        _trg, _ = self.cross_attn(trg, enc_src, enc_src, src_mask)
        trg = self.cross_attn_norm(trg + self.dropout(_trg))
        
        #Feed Forward
        _trg = self.ffn(trg)
        trg = self.ffn_norm(trg + self.dropout(_trg))
        
        return trg

class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, n_layers, n_heads, d_ff, dropout, max_len=5000):
        super().__init__()
        
        self.d_model = d_model
        
        # Embedding và Positional Encoding (Cần import class PositionalEncoding nếu bạn tách riêng)
        self.embedding = nn.Embedding(output_dim, d_model)
        # Giả sử bạn đã có PositionalEncoding trong file positional_encoding.py
        from .positional_encoding import PositionalEncoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg: [batch_size, trg_len]
        
        # Embedding + Positional Encoding
        trg = self.embedding(trg) * (self.d_model ** 0.5)
        trg = self.pos_encoding(trg)
        trg = self.dropout(trg)
        
        # Qua N lớp DecoderLayer
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
            
        # Output cuối cùng [batch_size, trg_len, output_dim]
        output = self.fc_out(trg)
        return output