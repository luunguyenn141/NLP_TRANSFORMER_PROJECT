import re
from typing import List

class SimpleTokenizer:
    """
    Tokenizer word-level đơn giản (không dùng BPE/SentencePiece).
    - Lowercase
    - Tách dấu câu
    - Tách từ bằng regex
    """

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def clean_text(self, text: str) -> str:
        """Làm sạch câu: bỏ khoảng trắng thừa, chuẩn hóa dấu câu."""
        if self.lowercase:
            text = text.lower()

        # Thêm khoảng trắng quanh dấu câu
        text = re.sub(r"([.,!?;:\-])", r" \1 ", text)

        # Bỏ khoảng trắng thừa
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tách câu thành tokens (word-level)."""
        text = self.clean_text(text)
        return text.split()

    def detokenize(self, tokens: List[str]) -> str:
        """Ghép tokens thành câu lại như cũ."""
        return " ".join(tokens)


if __name__ == "__main__":
    tok = SimpleTokenizer()
    s = "Hello, world! Đây là mô hình Transformer :)"
    tokens = tok.tokenize(s)
    print(tokens)
    print(tok.detokenize(tokens))