import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

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

# --- 1. CLASS EARLY STOPPING (MỚI) ---
class EarlyStopping:
    """
    Dừng training sớm nếu validation loss không cải thiện sau một số epoch nhất định (patience).
    """
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): Số epoch chờ đợi sau khi loss không cải thiện.
            verbose (bool): Nếu True, in ra thông báo mỗi khi validation loss cải thiện.
            delta (float): Thay đổi tối thiểu để được coi là cải thiện.
            path (str): Đường dẫn lưu checkpoint model tốt nhất.
            trace_func (function): Hàm dùng để in thông báo.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'Validation loss không giảm. EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Lưu model khi validation loss giảm.'''
        if self.verbose:
            self.trace_func(f'Validation loss giảm ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Đang lưu model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# --- 2. SCHEDULER (Noam) ---
class NoamOpt:
    """
    Learning rate scheduler with warmup (Transformer style).
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """
    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=2):
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


# --- 3. Helpers ---
def load_vocab():
    """Load vocabulary (prefer BPE if available)"""
    print("⏳ Đang tải bộ từ điển...")
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    
    # Try to load BPE vocab first
    src_bpe_path = os.path.join(VOCAB_DIR, "src_vocab_bpe.json")
    tgt_bpe_path = os.path.join(VOCAB_DIR, "tgt_vocab_bpe.json")
    
    if os.path.exists(src_bpe_path) and os.path.exists(tgt_bpe_path):
        print("   Sử dụng BPE vocabularies")
        src_vocab.load(src_bpe_path)
        tgt_vocab.load(tgt_bpe_path)
    else:
        # Fallback to word-level
        print("   Sử dụng word-level vocabularies")
        src_vocab.load(os.path.join(VOCAB_DIR, "src_vocab.json"))
        tgt_vocab.load(os.path.join(VOCAB_DIR, "tgt_vocab.json"))
    
    return src_vocab, tgt_vocab


def load_data(src_vocab, tgt_vocab, data_type):
    """Load dataset for training/validation"""
    print(f"⏳ Đang tải dữ liệu: {data_type}...")
    
    # Try BPE files first
    src_path = os.path.join(TOKENIZED_DIR, f"{data_type}.bpe.en")
    tgt_path = os.path.join(TOKENIZED_DIR, f"{data_type}.bpe.vi")
    
    if os.path.exists(src_path) and os.path.exists(tgt_path):
        print(f"   Sử dụng BPE tokenized files")
    else:
        # Fallback to word-level
        src_path = os.path.join(TOKENIZED_DIR, f"{data_type}.tok.en")
        tgt_path = os.path.join(TOKENIZED_DIR, f"{data_type}.tok.vi")
        print(f"   Sử dụng word-level tokenized files")

    with open(src_path, "r", encoding="utf-8") as f:
        src_data = [line.strip().split() for line in f]
    with open(tgt_path, "r", encoding="utf-8") as f:
        tgt_data = [line.strip().split() for line in f]

    print(f"   Loaded {len(src_data)} sentence pairs")

    dataset = TranslationDataset(
        src_data,
        tgt_data,
        src_vocab,
        tgt_vocab,
        max_len=cfg.max_seq_len
    )

    shuffle = True if data_type == "train" else False
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle,
                            num_workers=0, collate_fn=dataset.collate_fn, pin_memory=True)
    return dataloader


def train_epoch(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        # Your batch already has src, trg_input, trg_label from collate_fn
        src = batch['src'].to(device)
        trg_input = batch['trg_input'].to(device)
        trg_label = batch['trg_label'].to(device)

        optimizer.zero_grad()

        output = model(src, trg_input)

        output_dim = output.shape[-1]
        loss = criterion(output.contiguous().view(-1, output_dim),
                         trg_label.contiguous().view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        epoch_loss += loss.item()

        if i % 100 == 0:
            # Get learning rate from NoamOpt
            if isinstance(optimizer, NoamOpt):
                current_lr = optimizer._rate
            else:
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
        for batch in iterator:
            src = batch['src'].to(device)
            trg_input = batch['trg_input'].to(device)
            trg_label = batch['trg_label'].to(device)

            output = model(src, trg_input)
            output_dim = output.shape[-1]
            loss = criterion(output.contiguous().view(-1, output_dim),
                             trg_label.contiguous().view(-1))

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# --- 4. MAIN ---
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
    optimizer = NoamOpt(base_optimizer, cfg.d_model, warmup_steps=2000, factor=2)

    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, label_smoothing=0.1)

    # --- Cấu hình Early Stopping ---
    best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    early_stopping = EarlyStopping(patience=3, verbose=True, path=best_model_path)
    
    # Kiểm tra checkpoint cũ
    if os.path.exists(best_model_path):
        print(f"Đang nạp lại checkpoint từ: {best_model_path}")
        try:
            state_dict = torch.load(best_model_path, map_location=device)
            model.load_state_dict(state_dict)
            print("Đã khôi phục trạng thái mô hình thành công! Tiếp tục train...")
        except Exception as e:
            print(f"Lỗi khi load checkpoint: {e}")
            print("Sẽ train lại từ đầu.")
    else:
        print("Không tìm thấy checkpoint cũ. Bắt đầu huấn luyện từ đầu.")
        
    
    # Lists để lưu lịch sử cho biểu đồ
    train_losses = []
    valid_losses = []
    train_ppls = []
    valid_ppls = []
    
    print("\nBẮT ĐẦU VÒNG LẶP HUẤN LUYỆN")
    for epoch in range(cfg.num_epochs):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss = evaluate(model, valid_loader, criterion, device)
        
        # Tính Perplexity (PPL) = exp(loss)
        train_ppl = math.exp(train_loss)
        valid_ppl = math.exp(valid_loss)

        # LƯU HISTORY
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_ppls.append(train_ppl)
        valid_ppls.append(valid_ppl)

        end_time = time.time()
        mins, secs = divmod(end_time - start_time, 60)

        print(f'Epoch: {epoch+1:02} | Time: {int(mins)}m {int(secs)}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {valid_ppl:7.3f}')

        # GỌI EARLY STOPPING
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("⛔ Early stopping triggered! Dừng huấn luyện do validation loss không giảm sau 3 epochs.")
            break

        # Lưu checkpoint cuối cùng (đề phòng)
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'last_model.pth'))

    # VẼ BIỂU ĐỒ (LOSS & PPL)
    print("Đang vẽ biểu đồ huấn luyện...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Biểu đồ 1: Loss
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(valid_losses, label='Validation Loss', color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Epochs')
    ax1.legend()
    ax1.grid(True)

    # Biểu đồ 2: Perplexity
    ax2.plot(train_ppls, label='Train PPL', color='green')
    ax2.plot(valid_ppls, label='Validation PPL', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Perplexity over Epochs')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    # Lưu biểu đồ thành file ảnh thay vì chỉ show
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'training_metrics.png'))
    plt.show()
    print(f"Đã lưu biểu đồ tại: {os.path.join(CHECKPOINT_DIR, 'training_metrics.png')}")

if __name__ == "__main__":
    main()