import os
import sys
import torch
import torch.nn as nn
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from configs.config import cfg
from src.model.attention import MultiHeadAttention
from src.model.positional_encoding import PositionalEncoding

class PositionwiseFeedForward(nn.Module):
    "Mạng FFN (Feed Forward Network) [cite: 24]"
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    "Multi-Head Attention + Add & Norm + FFN"
    def __init__(self, d_model, n_head, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Sub-layer 1: Self-Attention
        _x = x
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        
        # 2. Sub-layer 2: Feed Forward
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class Encoder(nn.Module):
    "Encoder Tổng: Chồng N lớp EncoderLayer lại với nhau"
    def __init__(self):
        super(Encoder, self).__init__()
        self.d_model = cfg.d_model
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pe = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)
        
        # Tạo danh sách các lớp EncoderLayer
        self.layers = nn.ModuleList([
            EncoderLayer(cfg.d_model, cfg.n_head, cfg.d_ff, cfg.dropout) 
            for _ in range(cfg.n_layer)
        ])
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, src, src_mask=None):
        # src shape: (batch_size, seq_len)
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pe(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

# --- PHẦN TEST (CHẠY THỬ) ---
if __name__ == "__main__":
    try:
        # Giả lập input (Batch=2, Câu dài 10 từ)
        src = torch.randint(0, 100, (2, 10))
        model = Encoder()
        out = model(src)
        print("✅ Input shape:", src.shape)
        print("✅ Encoder Output:", out.shape)
        
        if out.shape == (2, 10, 512):
            print("DONE")
    except Exception as e:
        print("ERROR:", e)