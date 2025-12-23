import os
import sys
import torch

# --- 1. SETUP ĐƯỜNG DẪN IMPORT ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # Trỏ về thư mục gốc NLP_TRANSFORMER_PROJECT
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import các module custom
from configs.config import cfg
from src.model.transformer import Transformer
from src.data.data_processing.vocabulary import Vocabulary
from src.data.data_processing.tokenizer import BPETokenizer

# --- 2. CẤU HÌNH ---
# Lưu ý: Cấu trúc thư mục phải khớp với lúc train
VOCAB_DIR = os.path.join(project_root,"NLP_TRANSFORMER_PROJECT", "src", "data", "vocab")
CHECKPOINT_PATH = os.path.join(project_root,"NLP_TRANSFORMER_PROJECT", "checkpoints", "best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. HÀM BEAM SEARCH ---
def beam_search_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device, beam_width=3):
    model.eval()
    with torch.no_grad():
        enc_src = model.encoder(src, src_mask)
    
    # Beam: list of tuples (cumulative_log_prob, sequence_list)
    beam = [(0.0, [start_symbol])] 
    
    for _ in range(max_len):
        candidates = []
        for score, seq in beam:
            # Nếu chuỗi đã kết thúc bằng end_symbol, giữ nguyên
            if seq[-1] == end_symbol:
                candidates.append((score, seq))
                continue
            
            # Chuẩn bị input cho decoder
            trg_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor)
            
            with torch.no_grad():
                output = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
                # Lấy dự đoán cho token cuối cùng
                prob = output[:, -1, :]
                log_prob = torch.log_softmax(prob, dim=-1)
            
            # Lấy top k ứng viên tốt nhất
            topk_log_probs, topk_indices = torch.topk(log_prob, beam_width)
            
            for i in range(beam_width):
                sym = topk_indices[0][i].item()
                added_score = topk_log_probs[0][i].item()
                candidates.append((score + added_score, seq + [sym]))
        
        # Sắp xếp và chọn ra top beam_width ứng viên có điểm cao nhất
        beam = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
        
        # Nếu tất cả các beam đều đã kết thúc, dừng sớm
        if all(seq[-1] == end_symbol for _, seq in beam):
            break
            
    # Trả về chuỗi có điểm cao nhất
    return beam[0][1]

# --- 4. HÀM LOAD TÀI NGUYÊN (QUAN TRỌNG) ---
def load_resources():
    print(f"⏳ Đang tải tài nguyên từ: {VOCAB_DIR}")
    
    # --- A. Load Vocab (Mapping ID <-> Token) ---
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    
    src_vocab_path = os.path.join(VOCAB_DIR, "src_vocab.json")
    tgt_vocab_path = os.path.join(VOCAB_DIR, "tgt_vocab.json")
    
    if not os.path.exists(src_vocab_path) or not os.path.exists(tgt_vocab_path):
        raise FileNotFoundError("Không tìm thấy file vocab. Hãy chạy pipeline trước!")

    src_vocab.load(src_vocab_path)
    tgt_vocab.load(tgt_vocab_path)
    
    # Fix lỗi <unk> nếu file json bị lỗi
    if "<unk>" not in src_vocab.stoi:
        src_vocab.stoi["<unk>"] = 1 # Giả định index 1, hoặc len(stoi)
        print("⚠️ Cảnh báo: Đã tự động vá lỗi thiếu <unk> trong src_vocab")

    # --- B. Load Tokenizer (BPE Model) ---
    # Cần 2 tokenizer riêng cho Source (Vi) và Target (En)
    src_tokenizer = BPETokenizer(vocab_size=cfg.vocab_size)
    tgt_tokenizer = BPETokenizer(vocab_size=cfg.vocab_size)
    
    src_bpe_path = os.path.join(VOCAB_DIR, "src_bpe.json")
    tgt_bpe_path = os.path.join(VOCAB_DIR, "tgt_bpe.json")
    
    if os.path.exists(src_bpe_path):
        src_tokenizer.load(src_bpe_path)
    else:
        raise FileNotFoundError(f"Thiếu file {src_bpe_path}. Hãy chạy pipeline step 2!")

    if os.path.exists(tgt_bpe_path):
        tgt_tokenizer.load(tgt_bpe_path)
    else:
        raise FileNotFoundError(f"Thiếu file {tgt_bpe_path}")

    # --- C. Load Model ---
    model = Transformer(
        src_vocab_size=len(src_vocab),
        trg_vocab_size=len(tgt_vocab),
        src_pad_idx=src_vocab.to_index('<pad>'),
        trg_pad_idx=tgt_vocab.to_index('<pad>'),
        d_model=cfg.d_model,
        n_layers=cfg.n_layer,
        n_heads=cfg.n_head,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_len=cfg.max_seq_len
    ).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"⏳ Đang load checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        # Load state_dict với strict=False để tránh lỗi nhỏ không tương thích
        try:
            model.load_state_dict(ckpt)
            print("✅ Load model thành công!")
        except Exception as e:
            print(f"⚠️ Load model có cảnh báo (có thể do sai lệch kích thước vocab): {e}")
            model.load_state_dict(ckpt, strict=False)
    else:
        print("❌ KHÔNG TÌM THẤY CHECKPOINT! Model sẽ dịch ngẫu nhiên.")
    
    return model, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer

# --- 5. HÀM DỊCH ---
def translate_sentence(sentence, model, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, device):
    model.eval()
    
    # 1. Tokenize (Sử dụng BPE tokenizer của Source - Tiếng Việt)
    # tokenize() trả về list các sub-words (string)
    src_tokens = src_tokenizer.tokenize(sentence)
    
    # 2. Convert to Indices
    # Map sub-words sang ID dùng src_vocab
    src_ids = [src_vocab.to_index(t) for t in src_tokens]
    
    # Thêm batch dimension
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device) # [1, seq_len]
    
    # 3. Tạo Mask
    src_mask = model.make_src_mask(src_tensor)
    
    # 4. Beam Search
    sos_idx = tgt_vocab.to_index('<sos>')
    eos_idx = tgt_vocab.to_index('<eos>')
    
    # Kết quả trả về là list các ID
    pred_ids = beam_search_decode(
        model, src_tensor, src_mask, 
        max_len=cfg.max_seq_len,
        start_symbol=sos_idx, 
        end_symbol=eos_idx, 
        device=device,
        beam_width=3,
        no_repeat_ngram_size=3
    )
    
    # 5. Convert IDs to Text (Sử dụng BPE tokenizer của Target - Tiếng Anh)
    # Loại bỏ SOS và EOS trước khi decode
    pred_ids_clean = [idx for idx in pred_ids if idx not in [sos_idx, eos_idx]]
    
    # Cách 1: Map ID -> Token String -> Detokenize (Dùng hàm detokenize cũ)
    # pred_tokens = [tgt_vocab.to_token(idx) for idx in pred_ids_clean]
    # translated_text = tgt_tokenizer.detokenize(pred_tokens)

    # Cách 2 (Khuyên dùng với BPE): Dùng hàm decode trực tiếp của thư viện
    # Tuy nhiên, do cấu trúc project đang tách rời Vocab và Tokenizer, ta dùng cách 1 cho an toàn
    # Hoặc nếu bạn đã update detokenize như tôi hướng dẫn trước đó:
    pred_tokens = [tgt_vocab.to_token(idx) for idx in pred_ids_clean]
    translated_text = tgt_tokenizer.detokenize(pred_tokens)
    
    return translated_text

# --- 6. MAIN ---
def main():
    try:
        model, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer = load_resources()
    except Exception as e:
        print(f"❌ Lỗi khởi tạo: {e}")
        return

    print("\n" + "="*50)
    print(f"🌏 DEMO DỊCH MÁY: {cfg.src_lang.upper()} -> {cfg.tgt_lang.upper()}")
    print("Nhập 'q' để thoát.")
    print("="*50 + "\n")
    
    while True:
        try:
            src_text = input(f"Nhập câu ({cfg.src_lang}): ")
            if src_text.lower() in ['q', 'quit', 'exit']:
                break
            
            if not src_text.strip():
                continue
                
            translation = translate_sentence(
                src_text, model, src_vocab, tgt_vocab, 
                src_tokenizer, tgt_tokenizer, DEVICE
            )
            
            print(f"-> Dịch ({cfg.tgt_lang}): {translation}")
            print("-" * 30)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Lỗi khi dịch: {e}")

if __name__ == "__main__":
    main()