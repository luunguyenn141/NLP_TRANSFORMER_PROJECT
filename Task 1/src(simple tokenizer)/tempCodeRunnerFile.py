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


# --- 1. SCHEDULER (Noam) ---
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
    
    # [MỚI] Lấy accumulation_steps từ config (mặc định là 1 nếu bạn chưa set)
    # Giúp giả lập Batch Size lớn (ví dụ: batch 64 * 4 steps = 256)
    accum_steps = getattr(cfg, 'accumulation_steps', 1)
    
    optimizer.zero_grad() # Reset gradient trước khi vào vòng lặp

    for i, batch in enumerate(iterator):
        # --- 1. XỬ LÝ DỮ LIỆU ĐẦU VÀO (Giữ nguyên logic cũ) ---
        if isinstance(batch, dict):
            src = batch.get('src')
            if 'trg_input' in batch and 'trg_label' in batch:
                trg_input = batch['trg_input']
                trg_label = batch['trg_label']
                src = src.to(device)
                trg_input = trg_input.to(device)
                trg_label = trg_label.to(device)
            else:
                # Fallback cho code cũ
                trg = batch.get('tgt') or batch.get('trg')
                if trg is None:
                    raise KeyError("Batch dict does not contain 'tgt' or 'trg_input'/'trg_label' keys")
                src = src.to(device)
                trg = trg.to(device)
                trg_input = trg[:, :-1]
                trg_label = trg[:, 1:]
        else:
            # Fallback cho tuple
            src, trg = batch[0].to(device), batch[1].to(device)
            trg_input = trg[:, :-1]
            trg_label = trg[:, 1:]

        # --- 2. FORWARD PASS ---
        output = model(src, trg_input)
        output_dim = output.shape[-1]
        
        # --- 3. TÍNH LOSS ---
        loss = criterion(output.contiguous().view(-1, output_dim),
                         trg_label.contiguous().view(-1))

        # [MỚI] Chia nhỏ loss để trung bình gradient trong quá trình tích lũy
        loss = loss / accum_steps 
        
        # --- 4. BACKWARD ---
        loss.backward()

        # --- 5. OPTIMIZER STEP (Có điều kiện) ---
        # Chỉ update trọng số sau mỗi accum_steps hoặc ở batch cuối cùng
        if (i + 1) % accum_steps == 0 or (i + 1) == len(iterator):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad() # Reset gradient sau khi update

        # Cộng dồn loss (nhân ngược lại để log giá trị thực tế)
        epoch_loss += loss.item() * accum_steps

        # --- 6. LOGGING ---
        if i % 100 == 0:
            try:
                # Lấy LR an toàn cho cả NoamOpt lẫn torch.optim thường
                if hasattr(optimizer, 'optimizer'): 
                    current_lr = optimizer.optimizer.param_groups[0]['lr']
                else:
                    current_lr = optimizer.param_groups[0]['lr']
            except Exception:
                current_lr = 0.0
            
            # Loss hiển thị là loss thực tế của batch đó
            actual_loss = loss.item() * accum_steps
            print(f"   Step {i} | Loss: {actual_loss:.4f} | LR: {current_lr:.7f}")

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            # Support multiple batch formats for backward compatibility
            if isinstance(batch, dict):
                src = batch.get('src')
                if 'trg_input' in batch and 'trg_label' in batch:
                    trg_input = batch['trg_input']
                    trg_label = batch['trg_label']
                    src = src.to(device)
                    trg_input = trg_input.to(device)
                    trg_label = trg_label.to(device)
                else:
                    trg = batch.get('tgt') or batch.get('trg')
                    if trg is None:
                        raise KeyError("Batch dict does not contain 'tgt' or 'trg_input'/'trg_label' keys")
                    src = src.to(device)
                    trg = trg.to(device)
                    trg_input = trg[:, :-1]
                    trg_label = trg[:, 1:]
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
    model_opt = NoamOpt(base_optimizer, cfg.d_model, warmup_steps=2000, factor=2)

    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, label_smoothing=0.1)

    best_valid_loss = float('inf')


    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    
    start_epoch = 0 # Mặc định bắt đầu từ 0
    
    if os.path.exists(checkpoint_path):
        print(f"Đang nạp lại checkpoint từ: {checkpoint_path}")
        try:
            # Load trọng số vào model
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            print("Đã khôi phục trạng thái mô hình thành công! Tiếp tục train...")
            
        except Exception as e:
            print(f"Lỗi khi load checkpoint: {e}")
            print("Sẽ train lại từ đầu.")
    else:
        print("Không tìm thấy checkpoint cũ. Bắt đầu huấn luyện từ đầu.")
        
    
    train_losses = []
    valid_losses = []
    
    print("\nBẮT ĐẦU VÒNG LẶP HUẤN LUYỆN")
    for epoch in range(cfg.num_epochs):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, model_opt, criterion, device)
        valid_loss = evaluate(model, valid_loader, criterion, device)
        
        # --- LƯU HISTORY ---
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        end_time = time.time()
        mins, secs = divmod(end_time - start_time, 60)

        print(f'Epoch: {epoch+1:02} | Time: {int(mins)}m {int(secs)}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print("\t--> Đã lưu Best Model!")

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'last_model.pth'))

    # Vẽ biểu đồ loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
