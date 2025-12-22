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

# --- CẤU HÌNH ---
VOCAB_DIR = "src/data/vocab"
CHECKPOINT_PATH = "checkpoints/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- HÀM BEAM SEARCH ---
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
    """Load model, vocabulary, and tokenizer"""
    print("⏳ Đang tải tài nguyên (Vocab, Model)...")
    
    # 1. Load Vocab (ưu tiên BPE)
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    
    # Check for BPE vocab
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
    
    # 2. Load Model
    src_pad_idx = src_vocab.to_index('<pad>')
    trg_pad_idx = tgt_vocab.to_index('<pad>')
    
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
    
    # 3. Load checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            load_res = model.load_state_dict(ckpt, strict=False)
            print(f"✅ Đã load model từ {CHECKPOINT_PATH}")
            
            # Print warnings
            if hasattr(load_res, 'missing_keys'):
                if load_res.missing_keys:
                    print(f"⚠️  Missing keys: {load_res.missing_keys}")
                if load_res.unexpected_keys:
                    print(f"⚠️  Unexpected keys: {load_res.unexpected_keys}")
        except Exception as e:
            print(f"❌ Lỗi khi load model: {e}")
            return None, None, None, None
    else:
        print(f"❌ Không tìm thấy checkpoint tại {CHECKPOINT_PATH}")
        return None, None, None, None
    
    model.eval()
    print("✅ Model đã sẵn sàng cho inference")
    
    return model, src_vocab, tgt_vocab

def preprocess_text(text, is_vietnamese=True):
    """Simple text preprocessing"""
    text = text.lower().strip()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def tokenize_for_inference(text, vocab, is_vietnamese=True):
    """Tokenize text for inference"""
    # Preprocess
    text = preprocess_text(text, is_vietnamese)
    
    # For BPE vocab: split by space (assuming text already BPE tokenized during training)
    # For word-level vocab: split by space
    tokens = text.split()
    
    # If using BPE and token is not in vocab, try to handle it
    if vocab.bpe_tokenizer is not None:
        # Use BPE tokenizer if available
        return vocab.bpe_tokenizer.tokenize(text)
    
    return tokens

def translate_input(sentence, model, src_vocab, tgt_vocab, device):
    """Translate a sentence"""
    try:
        # Tokenize
        tokens = tokenize_for_inference(sentence, src_vocab, is_vietnamese=False)
        
        if not tokens:
            return "Lỗi: Không thể tokenize câu đầu vào"
        
        # Convert to indices
        src_indices = [src_vocab.to_index(token) for token in tokens]
        
        # Filter out unknown tokens (optional)
        src_indices = [idx for idx in src_indices if idx != src_vocab.unk_id]
        
        if not src_indices:
            return "Lỗi: Tất cả tokens đều không biết"
        
        src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
        src_mask = model.make_src_mask(src_tensor)
        
        sos_idx = tgt_vocab.to_index('<sos>')
        eos_idx = tgt_vocab.to_index('<eos>')
        
        # Perform beam search
        pred_indices = beam_search_decode(
            model, src_tensor, src_mask, 
            max_len=cfg.max_seq_len,
            start_symbol=sos_idx,
            end_symbol=eos_idx,
            device=device,
            beam_width=5
        )
        
        # Convert indices to tokens
        pred_tokens = []
        for idx in pred_indices:
            if idx == sos_idx:
                continue
            if idx == eos_idx:
                break
            token = tgt_vocab.to_token(idx)
            if token not in ['<pad>', '<unk>', '<sos>', '<eos>']:
                pred_tokens.append(token)
        
        # Join tokens
        if tgt_vocab.bpe_tokenizer is not None:
            # Use BPE detokenizer
            translated_text = tgt_vocab.bpe_tokenizer.detokenize(pred_tokens)
        else:
            # Simple join for word-level
            translated_text = ' '.join(pred_tokens)
        
        # Post-processing
        translated_text = translated_text.strip()
        if translated_text and translated_text[-1] not in '.!?':
            translated_text += '.'
        
        return translated_text
        
    except Exception as e:
        return f"Lỗi khi dịch: {e}"

# --- MAIN LOOP ---
def main():
    # Load resources
    resources = load_resources()
    if resources[0] is None:
        return
    
    model, src_vocab, tgt_vocab = resources
    
    print("\n" + "="*50)
    print("🤖 DEMO DỊCH MÁY VIỆT-ANH")
    print("Nhập câu tiếng Việt để dịch sang tiếng Anh")
    print("Lệnh: 'q' hoặc 'quit' để thoát, 'beam N' để đổi beam width")
    print("="*50 + "\n")
    
    beam_width = 5
    
    while True:
        try:
            src_text = input("📝 Câu tiếng Việt: ").strip()
            
            if not src_text:
                continue
            
            # Check for commands
            if src_text.lower() in ['q', 'quit', 'exit']:
                print("👋 Tạm biệt!")
                break
            
            if src_text.lower().startswith('beam '):
                try:
                    new_beam = int(src_text.split()[1])
                    if 1 <= new_beam <= 10:
                        beam_width = new_beam
                        print(f"✅ Đã đổi beam width thành {beam_width}")
                    else:
                        print("⚠️ Beam width phải từ 1 đến 10")
                except:
                    print("⚠️ Lệnh: beam N (N từ 1-10)")
                continue
            
            # Translate
            print(f"🔍 Đang dịch (beam={beam_width})...")
            translation = translate_input(src_text, model, src_vocab, tgt_vocab, DEVICE)
            
            print(f"✅ Bản dịch tiếng Anh: {translation}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Đã dừng chương trình.")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()