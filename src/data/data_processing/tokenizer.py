"""
BPE Tokenizer with SentencePiece for Vietnamese-English Translation
"""
import os
import json
import re
import sentencepiece as spm
from typing import List, Tuple, Optional

class BPETokenizer:
    """
    BPE Tokenizer wrapper for SentencePiece models
    """
    def __init__(self, model_path: str, lowercase: bool = True):
        """
        Args:
            model_path: Path to .model file
            lowercase: Whether to lowercase input
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.lowercase = lowercase
        
        # Get special tokens info
        self.pad_token = self.sp.id_to_piece(0)
        self.unk_token = self.sp.id_to_piece(1)
        self.sos_token = self.sp.id_to_piece(2)
        self.eos_token = self.sp.id_to_piece(3)
        
        self.pad_id = 0
        self.unk_id = 1
        self.sos_id = 2
        self.eos_id = 3
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into BPE tokens"""
        if self.lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize with BPE
        tokens = self.sp.encode_as_pieces(text)
        
        return tokens
    
    def detokenize(self, tokens: List[str]) -> str:
        """Detokenize BPE tokens back to text"""
        if not tokens:
            return ""
        
        # Filter out special tokens
        filtered_tokens = []
        for token in tokens:
            if token not in [self.pad_token, self.sos_token, self.eos_token, self.unk_token]:
                filtered_tokens.append(token)
        
        if not filtered_tokens:
            return ""
        
        # Decode
        text = self.sp.decode_pieces(filtered_tokens)
        
        # Clean up
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix common BPE artifacts
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r"(\w)' (\w)", r"\1'\2", text)  # Fix apostrophes
        
        return text
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token IDs"""
        if self.lowercase:
            text = text.lower()
        
        if add_special_tokens:
            return self.sp.encode(text, out_type=int, add_bos=True, add_eos=True)
        return self.sp.encode(text, out_type=int)
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        if skip_special_tokens:
            # Filter out special token IDs
            filtered_ids = [id for id in ids if id > 3]
            if not filtered_ids:
                return ""
            return self.sp.decode(filtered_ids)
        return self.sp.decode(ids)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.sp.get_piece_size()
    
    def save_info(self, filepath: str):
        """Save tokenizer info to JSON"""
        info = {
            'model_path': filepath,
            'vocab_size': self.get_vocab_size(),
            'special_tokens': {
                'pad': {'token': self.pad_token, 'id': self.pad_id},
                'unk': {'token': self.unk_token, 'id': self.unk_id},
                'sos': {'token': self.sos_token, 'id': self.sos_id},
                'eos': {'token': self.eos_token, 'id': self.eos_id}
            },
            'lowercase': self.lowercase
        }
        
        with open(filepath.replace('.model', '_info.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

class BPETokenizerManager:
    """
    Manager for training and loading BPE tokenizers for bilingual data
    """
    def __init__(self, config):
        self.config = config
        self.src_tokenizer = None
        self.tgt_tokenizer = None
        
    def train(self, src_text_path: str, tgt_text_path: str, 
              vocab_size: int = 30000, character_coverage: float = 1.0):
        """
        Train BPE tokenizers for both languages
        
        Args:
            src_text_path: Path to source language text file
            tgt_text_path: Path to target language text file
            vocab_size: Vocabulary size for each language
            character_coverage: Character coverage (1.0 for English, 0.9995 for Vietnamese)
        """
        # Create output directory
        os.makedirs("src/data/vocab/bpe_models", exist_ok=True)
        
        # Train source tokenizer (Vietnamese)
        print(f"Training Vietnamese BPE tokenizer (vocab_size={vocab_size})...")
        src_model_prefix = "src/data/vocab/bpe_models/bpe_vi"
        
        spm.SentencePieceTrainer.train(
            input=src_text_path,
            model_prefix=src_model_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=0.9995,  # For Vietnamese
            max_sentence_length=self.config.max_seq_len,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<sos>',
            eos_piece='<eos>',
            normalization_rule_name='nmt_nfkc',
            split_digits=True,
            add_dummy_prefix=True,
            remove_extra_whitespaces=True,
            num_threads=4
        )
        
        # Train target tokenizer (English)
        print(f"Training English BPE tokenizer (vocab_size={vocab_size})...")
        tgt_model_prefix = "src/data/vocab/bpe_models/bpe_en"
        
        spm.SentencePieceTrainer.train(
            input=tgt_text_path,
            model_prefix=tgt_model_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=1.0,  # For English
            max_sentence_length=self.config.max_seq_len,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<sos>',
            eos_piece='<eos>',
            normalization_rule_name='nmt_nfkc',
            split_digits=True,
            add_dummy_prefix=True,
            remove_extra_whitespaces=True,
            num_threads=4
        )
        
        print("✅ BPE tokenizers trained successfully!")
        return src_model_prefix + '.model', tgt_model_prefix + '.model'
    
    def load(self, src_model_path: str, tgt_model_path: str):
        """Load trained BPE tokenizers"""
        self.src_tokenizer = BPETokenizer(src_model_path, lowercase=True)
        self.tgt_tokenizer = BPETokenizer(tgt_model_path, lowercase=True)
        
        print(f"✅ BPE tokenizers loaded:")
        print(f"   Source (VI): vocab_size={self.src_tokenizer.get_vocab_size()}")
        print(f"   Target (EN): vocab_size={self.tgt_tokenizer.get_vocab_size()}")
        
        return self.src_tokenizer, self.tgt_tokenizer

# Test function
if __name__ == "__main__":
    # Test with a sample sentence
    test_sentence = "I don't know what's happening, but we'll figure it out!"
    
    # You'll need to train models first or load existing ones
    print("Testing BPE Tokenizer with sample sentence:")
    print(f"Input: {test_sentence}")
    
    # Example of loading (uncomment when you have trained models)
    # tokenizer = BPETokenizer("path/to/bpe_en.model")
    # tokens = tokenizer.tokenize(test_sentence)
    # print(f"Tokens: {tokens}")
    # print(f"Detokenized: {tokenizer.detokenize(tokens)}")