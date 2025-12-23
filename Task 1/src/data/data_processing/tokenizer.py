import os
from typing import List
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class BPETokenizer:
    """
    Wrapper cho BPE Tokenizer (sử dụng thư viện tokenizers của HuggingFace).
    """
    def __init__(self, vocab_size=30000, min_frequency=2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        # Khởi tạo mô hình BPE
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        # Sử dụng Whitespace pre-tokenizer để tách theo khoảng trắng trước
        self.tokenizer.pre_tokenizer = Whitespace()

    def train(self, files: List[str], save_path: str):
        """Train BPE model từ danh sách các file text."""
        print(f"Training BPE tokenizer on {files}...")
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            # Đảm bảo các token đặc biệt không bị tách
            special_tokens=["<pad>", "<unk>", "<sos>", "<eos>"],
            show_progress=True
        )
        self.tokenizer.train(files, trainer)
        self.tokenizer.save(save_path)
        print(f"Saved BPE tokenizer to {save_path}")

    def load(self, model_path: str):
        """Load BPE model đã train."""
        self.tokenizer = Tokenizer.from_file(model_path)

    def tokenize(self, text: str) -> List[str]:
        """
        Tách câu thành list các sub-words (string).
        """
        output = self.tokenizer.encode(text)
        return output.tokens

    def detokenize(self, tokens: List[str]) -> str:
        """
        Ghép tokens lại sử dụng Decoder chuẩn của thư viện.
        Cơ chế: Convert tokens -> IDs -> Decode (tự động xử lý ghép từ).
        """
        if not tokens:
            return ""
    
        # 1. Chuyển list các string tokens về list các IDs
        # token_to_id trả về None nếu không tìm thấy, nên ta map về <unk>
        ids = [self.tokenizer.token_to_id(t) for t in tokens]
        
        # Lọc bỏ các None nếu có (để an toàn)
        ids = [i for i in ids if i is not None]
        
        # skip_special_tokens=True giúp loại bỏ tự động các <pad>, <sos>, <eos>
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        
        return text