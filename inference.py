import os
import sys
import torch
import re

# Đảm bảo import được các module trong project
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import cfg
from src.model.transformer import Transformer
from src.data.data_processing.vocabulary import Vocabulary
from src.data.data_processing.tokenizer import SimpleTokenizer

# --- CẤU HÌNH ---
VOCAB_DIR = "src/data/vocab"
CHECKPOINT_PATH = "checkpoints/best_model.pth" # Đường dẫn model tốt nhất
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- HÀM BEAM SEARCH (Đưa vào đây để tiện chạy độc lập) ---
def beam_search_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device, beam_width=3):
    model.eval()
    with torch.no_grad():
        enc_src = model.encoder(src, src_mask)
    
    # Beam khởi tạo: (score, sequence)
    beam = [(0.0, [start_symbol])] 
    
    for _ in range(max_len):
        candidates = []
        for score, seq in beam:
            if seq[-1] == end_symbol:
                candidates.append((score, seq))
                continue
            
            trg_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor)
            
            with torch.no_grad():
                output = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
                prob = output[:, -1, :]
                log_prob = torch.log_softmax(prob, dim=-1)
            
            topk_log_probs, topk_indices = torch.topk(log_prob, beam_width)
            
            for i in range(beam_width):
                sym = topk_indices[0][i].item()
                added_score = topk_log_probs[0][i].item()
                candidates.append((score + added_score, seq + [sym]))
        
        beam = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
        if all(seq[-1] == end_symbol for _, seq in beam):
            break
            
    return beam[0][1]

# --- HÀM XỬ LÝ CHÍNH ---
def load_resources():
    print("⏳ Đang tải tài nguyên (Vocab, Model)...")
    
    # 1. Load Vocab
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    # Đường dẫn file json dựa trên cấu trúc thư mục bạn đã upload
    src_vocab.load(os.path.join(VOCAB_DIR, "src_vocab.json")) #
    tgt_vocab.load(os.path.join(VOCAB_DIR, "tgt_vocab.json")) #
    
    # 2. Load Tokenizer
    tokenizer = SimpleTokenizer() #
    
    # 3. Load Model
    src_pad_idx = src_vocab.to_index('<pad>')
    trg_pad_idx = tgt_vocab.to_index('<pad>')
    
    model = Transformer(
        src_vocab_size=len(src_vocab),
        trg_vocab_size=len(tgt_vocab),
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        d_model=cfg.d_model,     #
        n_layers=cfg.n_layer,    #
        n_heads=cfg.n_head,      #
        d_ff=cfg.d_ff,           #
        dropout=cfg.dropout,     #
        max_len=cfg.max_seq_len  #
    ).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        # Try to load state dict permissively to support older checkpoints
        load_res = model.load_state_dict(ckpt, strict=False)
        # Report any missing / unexpected keys to help debugging
        missing = load_res.missing_keys if hasattr(load_res, 'missing_keys') else load_res.get('missing_keys')
        unexpected = load_res.unexpected_keys if hasattr(load_res, 'unexpected_keys') else load_res.get('unexpected_keys')
        print(f"✅ Đã load model từ {CHECKPOINT_PATH}")
        if missing:
            print(f"⚠️ Missing keys in checkpoint (model had these keys but checkpoint did not): {missing}")
        if unexpected:
            print(f"⚠️ Unexpected keys in checkpoint (checkpoint had these keys not used by model): {unexpected}")
    else:
        print(f"❌ CẢNH BÁO: Không tìm thấy {CHECKPOINT_PATH}. Model chưa được huấn luyện!")
    
    return model, src_vocab, tgt_vocab, tokenizer

def translate_input(sentence, model, src_vocab, tgt_vocab, tokenizer, device):
    # 1. Tokenize & Preprocess
    tokens = tokenizer.tokenize(sentence)
    
    # 2. Convert to Indices
    src_indices = [src_vocab.to_index(token) for token in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device) # [1, seq_len]
    
    # 3. Create Mask
    src_mask = model.make_src_mask(src_tensor) #
    
    # 4. Beam Search Decoding
    sos_idx = tgt_vocab.to_index('<sos>')
    eos_idx = tgt_vocab.to_index('<eos>')
    
    pred_indices = beam_search_decode(
        model, src_tensor, src_mask, 
        max_len=100, 
        start_symbol=sos_idx, 
        end_symbol=eos_idx, 
        device=device,
        beam_width=5 # Bạn có thể chỉnh beam_width tại đây
    )
    
    # 5. Convert Indices to Text
    pred_tokens = [tgt_vocab.to_token(idx) for idx in pred_indices if idx not in [sos_idx, eos_idx]]
    translated_text = tokenizer.detokenize(pred_tokens) #
    
    return translated_text

# --- MAIN LOOP ---
def main():
    model, src_vocab, tgt_vocab, tokenizer = load_resources()
    
    print("\n" + "="*40)
    print("🤖 DEMO DỊCH MÁY (BEAM SEARCH)")
    print("Nhập 'q' hoặc 'quit' để thoát.")
    print("="*40 + "\n")
    
    while True:
        try:
            src_text = input("Mời nhập câu tiếng Anh: ")
            if src_text.lower() in ['q', 'quit', 'exit']:
                print("Tạm biệt!")
                break
            
            if not src_text.strip():
                continue
                
            translation = translate_input(src_text, model, src_vocab, tgt_vocab, tokenizer, DEVICE)
            
            print(f"-> Bản dịch tiếng Việt: {translation}")
            print("-" * 30)
            
        except KeyboardInterrupt:
            print("\nĐã dừng chương trình.")
            break
        except Exception as e:
            print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()