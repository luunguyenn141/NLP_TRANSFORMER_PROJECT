# preprocessing.py
# Load, clean, normalize raw bilingual data
# This module prepares parallel sentences before tokenization

import re
from typing import List, Tuple

class Preprocessor:
    """
    Preprocessor for parallel bilingual data
    - Lowercasing
    - Removing special characters
    - Normalizing spaces
    - Aligning EN–VI sentence pairs
    """

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def clean_sentence(self, sentence: str) -> str:
        """Basic cleaning steps for a sentence."""
        if self.lowercase:
            sentence = sentence.lower()

        # Remove weird characters (keep letters, digits, punctuation)
        sentence = re.sub(r"[^a-zA-Z0-9\u00C0-\u1EF9.,!?;:'\-\s]", "", sentence)

        # Normalize multiple spaces
        sentence = re.sub(r"\s+", " ", sentence).strip()

        return sentence

    def load_parallel_data(self, src_path: str, tgt_path: str) -> Tuple[List[str], List[str]]:
        """
        Load bilingual corpus (line-aligned)
        Returns: (src_sentences, tgt_sentences)
        """
        with open(src_path, "r", encoding="utf-8") as f_src:
            src_lines = f_src.read().strip().splitlines()

        with open(tgt_path, "r", encoding="utf-8") as f_tgt:
            tgt_lines = f_tgt.read().strip().splitlines()

        assert len(src_lines) == len(tgt_lines), "Source and target files must have same number of lines"

        # Clean sentences
        src_cleaned = [self.clean_sentence(s) for s in src_lines]
        tgt_cleaned = [self.clean_sentence(s) for s in tgt_lines]

        return src_cleaned, tgt_cleaned

    def save_processed(self, src: List[str], tgt: List[str], src_out: str, tgt_out: str):
        """Save cleaned bilingual data to processed folder."""
        with open(src_out, "w", encoding="utf-8") as f:
            f.write("\n".join(src))

        with open(tgt_out, "w", encoding="utf-8") as f:
            f.write("\n".join(tgt))


# Simple test
if __name__ == "__main__":
    p = Preprocessor()
    print(p.clean_sentence("Hello!!!   This   is a   TEST 123     "))