import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, 
                 src_pad_idx, trg_pad_idx, 
                 d_model=512, n_layers=6, n_heads=8, 
                 d_ff=2048, dropout=0.1, max_len=100):
        super().__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(trg_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        
        # Khởi tạo trọng số Xavier 
        if hasattr(self.decoder, 'embedding') and hasattr(self.decoder, 'fc_out'):
            self.decoder.fc_out.weight = self.decoder.embedding.weight
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        # src shape: [batch_size, src_len]
        # Mask các vị trí padding
        # mask shape: [batch_size, 1, 1, src_len]
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        # trg shape: [batch_size, trg_len]
        # Mask padding
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Mask tương lai (Look-ahead mask / No Peak mask)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()
        
        return trg_pad_mask & trg_sub_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        return output