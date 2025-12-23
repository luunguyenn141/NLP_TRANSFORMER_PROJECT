import os
import sys
import torch
from tqdm import tqdm
import sacrebleu  

# Setup đường dẫn
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import cfg
from src.model.transformer import Transformer
from src.data.data_processing.vocabulary import Vocabulary
from src.data.data_processing.tokenizer import BPETokenizer
from src.beam_search import beam_search_decode

# --- CẤU HÌNH ---
VOCAB_DIR = os.path.join(project_root, "src/data/vocab")
PROCESSED_DIR = os.path.join(project_root, "src/data/processed") 
CHECKPOINT_PATH = os.path.join(project_root, "checkpoints/best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_SET = "tst2013" 

def load_resources():
    print("⏳ Đang tải tài nguyên...")
    # 1. Load Vocab
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.load(os.path.join(VOCAB_DIR, "src_vocab.json"))
    tgt_vocab.load(os.path.join(VOCAB_DIR, "tgt_vocab.json"))
    
    # 2. Load Tokenizer 
    # Tokenizer target (En) dùng để ghép từ tiếng Anh lại
    tgt_tokenizer = BPETokenizer(vocab_size=cfg.vocab_size)
    tgt_bpe_path = os.path.join(VOCAB_DIR, "tgt_bpe.json")
    
    # Chúng ta cũng cần src tokenizer để tokenize input test data
    src_tokenizer = BPETokenizer(vocab_size=cfg.vocab_size)
    src_bpe_path = os.path.join(VOCAB_DIR, "src_bpe.json")
    
    if os.path.exists(tgt_bpe_path): tgt_tokenizer.load(tgt_bpe_path)
    if os.path.exists(src_bpe_path): src_tokenizer.load(src_bpe_path)

    return src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer

def load_test_sentences(data_type):
    """
    Load dữ liệu dưới dạng câu raw (string) thay vì token đã cắt sẵn.
    """
    print(f"⏳ Đang tải dữ liệu test gốc: {data_type}...")
    
    # Đường dẫn file đã clean (chưa tokenized BPE)
    src_path = os.path.join(PROCESSED_DIR, f"{data_type}.clean.en") # Source là tiếng Việt
    tgt_path = os.path.join(PROCESSED_DIR, f"{data_type}.clean.vi") # Target là tiếng Anh
    
    with open(src_path, "r", encoding="utf-8") as f:
        src_sentences = [line.strip() for line in f]
        
    with open(tgt_path, "r", encoding="utf-8") as f:
        tgt_sentences = [line.strip() for line in f]
        
    return src_sentences, tgt_sentences

def translate_sentence(sentence, model, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, device):
    model.eval()
    
    # 1. Tokenize src sentence 
    src_tokens = src_tokenizer.tokenize(sentence)
    
    # 2. Convert to Indices
    src_indices = [src_vocab.to_index(token) for token in src_tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    
    sos_idx = tgt_vocab.to_index('<sos>')
    eos_idx = tgt_vocab.to_index('<eos>')
    
    # 3. Beam Search
    pred_indices = beam_search_decode(
        model, src_tensor, src_mask, 
        max_len=cfg.max_seq_len, 
        start_symbol=sos_idx, 
        end_symbol=eos_idx, 
        device=device, 
        beam_width=3
    )
    
    # 4. Convert Indices back to Tokens -> String
    pred_token_strs = [tgt_vocab.to_token(idx) for idx in pred_indices if idx not in [sos_idx, eos_idx]]
    
    # 5. Detokenize 
    pred_sentence = tgt_tokenizer.detokenize(pred_token_strs)
    
    return pred_sentence

def main():
    # 1. Setup
    src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer = load_resources()
    
    # 2. Build Model
    model = Transformer(
        src_vocab_size=len(src_vocab),
        trg_vocab_size=len(tgt_vocab),
        src_pad_idx=src_vocab.to_index('<pad>'),
        trg_pad_idx=tgt_vocab.to_index('<pad>'),
        d_model=cfg.d_model, n_layers=cfg.n_layer, n_heads=cfg.n_head,
        d_ff=cfg.d_ff, dropout=cfg.dropout, max_len=cfg.max_seq_len
    ).to(DEVICE)
    
    # 3. Load Checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE), strict=False)
    else:
        print("❌ Không tìm thấy model checkpoint!")
        return

    # 4. Load Data
    src_sentences, tgt_sentences = load_test_sentences(TEST_SET)

    hypotheses = [] # Chứa các câu máy dịch
    references = [] # Chứa các câu đáp án gốc
    
    print(f"🚀 Bắt đầu đánh giá trên {len(src_sentences)} câu...")
    
    for src, tgt in tqdm(zip(src_sentences, tgt_sentences), total=len(src_sentences)):
        # Máy dịch
        pred = translate_sentence(src, model, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, DEVICE)
        hypotheses.append(pred)
        references.append(tgt)
        
    # 5. Calculate BLEU Score using sacrebleu
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    
    print(f"\n" + "="*40)
    print(f"KẾT QUẢ ĐÁNH GIÁ ({TEST_SET})")
    print(f"SacreBLEU Score: {bleu.score:.2f}")
    print("="*40)
    
    # Some example outputs
    print("\nVí dụ:")
    for i in range(min(3, len(hypotheses))):
        print(f"Src: {src_sentences[i]}")
        print(f"Ref: {references[i]}")
        print(f"Hyp: {hypotheses[i]}")
        print("-" * 20)

if __name__ == "__main__":
    main()