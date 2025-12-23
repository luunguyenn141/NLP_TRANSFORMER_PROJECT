import os
import torch
import sys
from torchtext.data.metrics import bleu_score
from tqdm import tqdm

# Setup đường dẫn import
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import cfg
from src.model.transformer import Transformer
from src.data.data_processing.vocabulary import Vocabulary
from src.beam_search import beam_search_decode

# --- CẤU HÌNH ---
TOKENIZED_DIR = "src/data/processed/tokenized"
VOCAB_DIR = "src/data/vocab"
CHECKPOINT_PATH = "checkpoints/best_model.pth" # Load model tốt nhất
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_SET = "tst2013" # Tên tập dữ liệu dùng để test (theo file list bạn cung cấp)

def load_vocab():
    print("⏳ Đang tải bộ từ điển...")
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.load(os.path.join(VOCAB_DIR, "src_vocab.json"))
    tgt_vocab.load(os.path.join(VOCAB_DIR, "tgt_vocab.json"))
    return src_vocab, tgt_vocab

def load_test_data(data_type):
    print(f"⏳ Đang tải dữ liệu test: {data_type}...")
    en_path = os.path.join(TOKENIZED_DIR, f"{data_type}.tok.en")
    vi_path = os.path.join(TOKENIZED_DIR, f"{data_type}.tok.vi")

    with open(en_path, "r", encoding="utf-8") as f:
        src_data = [line.strip().split() for line in f]
    with open(vi_path, "r", encoding="utf-8") as f:
        tgt_data = [line.strip().split() for line in f]
        
    return src_data, tgt_data

def translate_sentence(sentence_tokens, model, src_vocab, tgt_vocab, device, max_len=100):
    model.eval()
    
    # Convert tokens to indices
    src_indices = [src_vocab.to_index(token) for token in sentence_tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device) # [1, src_len]
    
    # Tạo mask cho src
    src_mask = model.make_src_mask(src_tensor)
    
    sos_idx = tgt_vocab.to_index('<sos>')
    eos_idx = tgt_vocab.to_index('<eos>')
    
    # Thực hiện Beam Search
    pred_indices = beam_search_decode(model, src_tensor, src_mask, max_len, sos_idx, eos_idx, device, beam_width=3)
    
    # Convert indices back to tokens (bỏ <sos>)
    pred_tokens = [tgt_vocab.to_token(idx) for idx in pred_indices if idx not in [sos_idx, eos_idx]]
    
    return pred_tokens

def calculate_bleu(data, model, src_vocab, tgt_vocab, device):
    src_data, tgt_data = data
    
    candidates = []
    references = []
    
    print("🚀 Bắt đầu dịch và tính toán BLEU...")
    for src_tokens, tgt_tokens in tqdm(zip(src_data, tgt_data), total=len(src_data)):
        # Dự đoán
        pred_tokens = translate_sentence(src_tokens, model, src_vocab, tgt_vocab, device)
        
        candidates.append(pred_tokens)
        references.append([tgt_tokens]) # BLEU yêu cầu list các references (nested list)

    # Tính BLEU score
    score = bleu_score(candidates, references)
    return score * 100

def main():
    # 1. Load Vocab
    src_vocab, tgt_vocab = load_vocab()
    src_pad_idx = src_vocab.to_index('<pad>')
    trg_pad_idx = tgt_vocab.to_index('<pad>')
    
    # 2. Build Model Structure
    model = Transformer(
        src_vocab_size=len(src_vocab),
        trg_vocab_size=len(tgt_vocab),
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        d_model=cfg.d_model,
        n_layers=cfg.n_layer,
        n_heads=cfg.n_head,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_len=cfg.max_seq_len
    ).to(DEVICE)
    
    # 3. Load Checkpoint (use non-strict load to tolerate small architecture changes)
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        load_res = model.load_state_dict(ckpt, strict=False)
        print(f"✅ Đã load model từ {CHECKPOINT_PATH}")
        # Print missing/unexpected keys to aid debugging
        try:
            missing = load_res.missing_keys if hasattr(load_res, 'missing_keys') else load_res.get('missing_keys')
            unexpected = load_res.unexpected_keys if hasattr(load_res, 'unexpected_keys') else load_res.get('unexpected_keys')
        except Exception:
            missing = None
            unexpected = None

        if missing:
            print(f"⚠️ Missing keys in checkpoint (initialized randomly): {missing}")
        if unexpected:
            print(f"⚠️ Unexpected keys found in checkpoint (ignored): {unexpected}")
    else:
        print(f"Không tìm thấy checkpoint tại {CHECKPOINT_PATH}")
        return

    # 4. Load Data & Evaluate
    test_data = load_test_data(TEST_SET)
    bleu = calculate_bleu(test_data, model, src_vocab, tgt_vocab, DEVICE)
    
    print(f"\n------------------------------------------------")
    print(f"BLEU Score trên tập {TEST_SET}: {bleu:.2f}")
    print(f"------------------------------------------------")

if __name__ == "__main__":
    main()