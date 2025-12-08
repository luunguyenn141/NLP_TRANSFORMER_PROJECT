import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure project root is on `sys.path` so imports like `configs` and `src` resolve
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import cfg
from src.model.transformer import Transformer
from src.data.data_processing.dataset import TranslationDataset
from src.data.data_processing.vocabulary import Vocabulary

# --- CẤU HÌNH ĐƯỜNG DẪN (Khớp với run_pipeline.py) ---
TOKENIZED_DIR = "src/data/processed/tokenized"
VOCAB_DIR = "src/data/vocab"
CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# --- 1. SCHEDULER (Noam) ---
class NoamOpt:
    """
    Learning rate scheduler with warmup (Transformer style).
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """
    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1):
        self.optimizer = optimizer
        self._step = 0
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.d_model = d_model
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.d_model ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


# --- 2. Helpers ---
def load_vocab():
    print("⏳ Đang tải bộ từ điển...")
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.load(os.path.join(VOCAB_DIR, "src_vocab.json"))
    tgt_vocab.load(os.path.join(VOCAB_DIR, "tgt_vocab.json"))
    return src_vocab, tgt_vocab


def load_data(src_vocab, tgt_vocab, data_type):
    print(f"⏳ Đang tải dữ liệu: {data_type}...")
    en_path = os.path.join(TOKENIZED_DIR, f"{data_type}.tok.en")
    vi_path = os.path.join(TOKENIZED_DIR, f"{data_type}.tok.vi")

    with open(en_path, "r", encoding="utf-8") as f:
        src_data = [line.split() for line in f]
    with open(vi_path, "r", encoding="utf-8") as f:
        tgt_data = [line.split() for line in f]

    dataset = TranslationDataset(src_data, tgt_data, src_vocab, tgt_vocab)

    shuffle = True if data_type == "train" else False
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle,
                            num_workers=0, collate_fn=dataset.collate_fn, pin_memory=True)
    return dataloader


def train_epoch(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        if isinstance(batch, dict):
            src = batch['src'].to(device)
            trg = batch['tgt'].to(device)
        else:
            src, trg = batch[0].to(device), batch[1].to(device)

        trg_input = trg[:, :-1]
        trg_label = trg[:, 1:]

        optimizer.zero_grad()

        output = model(src, trg_input)

        output_dim = output.shape[-1]
        loss = criterion(output.contiguous().view(-1, output_dim),
                         trg_label.contiguous().view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        epoch_loss += loss.item()

        if i % 50 == 0:
            try:
                current_lr = optimizer.optimizer.param_groups[0]['lr']
            except Exception:
                # fallback if optimizer is a plain torch optimizer
                try:
                    current_lr = optimizer.param_groups[0]['lr']
                except Exception:
                    current_lr = 0.0
            print(f"   Step {i} | Loss: {loss.item():.4f} | LR: {current_lr:.7f}")

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            if isinstance(batch, dict):
                src = batch['src'].to(device)
                trg = batch['tgt'].to(device)
            else:
                src, trg = batch[0].to(device), batch[1].to(device)

            trg_input = trg[:, :-1]
            trg_label = trg[:, 1:]

            output = model(src, trg_input)
            output_dim = output.shape[-1]
            loss = criterion(output.contiguous().view(-1, output_dim),
                             trg_label.contiguous().view(-1))

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# --- 3. MAIN ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Bắt đầu huấn luyện trên thiết bị: {device}")

    src_vocab, tgt_vocab = load_vocab()
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    print(f"✅ Vocab size: Source={src_vocab_size}, Target={tgt_vocab_size}")

    src_pad_idx = src_vocab.to_index('<pad>')
    trg_pad_idx = tgt_vocab.to_index('<pad>')

    train_loader = load_data(src_vocab, tgt_vocab, "train")
    valid_loader = load_data(src_vocab, tgt_vocab, "tst2012")

    model = Transformer(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=tgt_vocab_size,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        d_model=cfg.d_model,
        n_layers=cfg.n_layer,
        n_heads=cfg.n_head,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_len=cfg.max_seq_len
    ).to(device)

    base_optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    model_opt = NoamOpt(base_optimizer, cfg.d_model, warmup_steps=4000, factor=1)

    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, label_smoothing=0.1)

    best_valid_loss = float('inf')

    print("\n🏁 BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN")
    for epoch in range(cfg.num_epochs):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, model_opt, criterion, device)
        valid_loss = evaluate(model, valid_loader, criterion, device)

        end_time = time.time()
        mins, secs = divmod(end_time - start_time, 60)

        print(f'Epoch: {epoch+1:02} | Time: {int(mins)}m {int(secs)}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print("\t--> 💾 Đã lưu Best Model!")

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'last_model.pth'))


if __name__ == "__main__":
    main()
