import json
from collections import Counter
from typing import List, Union

class Vocabulary:
    def __init__(self, min_freq=2, max_size=30000, specials=None, bpe_tokenizer=None):
        """
        Vocabulary for NLP tasks with BPE support.
        
        Args:
            min_freq: minimum frequency to keep a token
            max_size: maximum vocabulary size
            specials: list of special tokens
            bpe_tokenizer: Optional BPE tokenizer (if using BPE)
        """
        self.min_freq = min_freq
        self.max_size = max_size
        self.bpe_tokenizer = bpe_tokenizer
        
        # Special tokens
        if specials is None:
            specials = ["<pad>", "<unk>", "<sos>", "<eos>"]
        self.specials = specials

        # Token → ID
        self.stoi = {}
        # ID → Token
        self.itos = []
        
        # Special token IDs
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        
        # Will be set after build_vocab or load
        self.pad_id = None
        self.unk_id = None
        self.sos_id = None
        self.eos_id = None

    def build_vocab(self, tokenized_texts):
        """
        Build vocabulary from tokenized sentences.
        
        Args:
            tokenized_texts: list of list, e.g. [["i", "love", "you"], ...]
        """
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        # Start with special tokens
        self.itos = list(self.specials)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        
        # Set special token IDs
        self.pad_id = self.stoi.get(self.pad_token, 0)
        self.unk_id = self.stoi.get(self.unk_token, 1)
        self.sos_id = self.stoi.get(self.sos_token, 2)
        self.eos_id = self.stoi.get(self.eos_token, 3)

        # Add tokens above min_freq and up to max_size
        for tok, freq in counter.most_common():
            if freq < self.min_freq:
                continue
            if len(self.itos) >= self.max_size:
                break
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)
            
        print(f"✅ Vocabulary built: {len(self)} tokens")

    def build_vocab_from_bpe(self):
        """
        Build vocabulary directly from BPE tokenizer.
        Use this when using BPE instead of word-level tokenization.
        """
        if self.bpe_tokenizer is None:
            raise ValueError("BPE tokenizer not provided")
        
        vocab_size = self.bpe_tokenizer.get_vocab_size()
        
        # Build from BPE tokenizer
        self.itos = []
        self.stoi = {}
        
        for i in range(vocab_size):
            token = self.bpe_tokenizer.sp.id_to_piece(i)
            self.itos.append(token)
            self.stoi[token] = i
        
        # Set special token IDs from BPE
        self.pad_id = self.bpe_tokenizer.pad_id
        self.unk_id = self.bpe_tokenizer.unk_id
        self.sos_id = self.bpe_tokenizer.sos_id
        self.eos_id = self.bpe_tokenizer.eos_id
        
        print(f"✅ BPE Vocabulary loaded: {len(self)} tokens")

    def __len__(self):
        return len(self.itos)

    def token_to_id(self, token):
        return self.stoi.get(token, self.unk_id)
    
    # Backwards-compatible alias expected by some scripts
    def to_index(self, token):
        """Alias for token_to_id kept for backwards compatibility."""
        return self.token_to_id(token)
    
    def to_token(self, idx):
        """Alias for id_to_token kept for backwards compatibility."""
        return self.id_to_token(idx)

    def id_to_token(self, idx):
        if idx < 0 or idx >= len(self.itos):
            return self.unk_token
        return self.itos[idx]

    def encode(self, tokens: Union[List[str], str]):
        """Encode tokens to IDs"""
        if isinstance(tokens, str):
            # If it's a string, tokenize it first if we have BPE tokenizer
            if self.bpe_tokenizer:
                return self.bpe_tokenizer.encode(tokens)
            else:
                tokens = tokens.split()
        
        return [self.token_to_id(tok) for tok in tokens]

    def decode(self, ids: List[int], skip_special_tokens: bool = True):
        """Decode IDs to tokens"""
        if self.bpe_tokenizer:
            return self.bpe_tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        
        tokens = [self.id_to_token(i) for i in ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.specials]
        return tokens

    def save(self, filepath):
        """Save vocabulary to JSON file"""
        data = {
            "itos": self.itos,
            "stoi": self.stoi,
            "min_freq": self.min_freq,
            "max_size": self.max_size,
            "special_tokens": {
                "pad": {"token": self.pad_token, "id": self.pad_id},
                "unk": {"token": self.unk_token, "id": self.unk_id},
                "sos": {"token": self.sos_token, "id": self.sos_id},
                "eos": {"token": self.eos_token, "id": self.eos_id}
            }
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load(self, filepath):
        """Load vocabulary from JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.itos = data["itos"]
        self.stoi = data["stoi"]
        self.min_freq = data["min_freq"]
        self.max_size = data["max_size"]
        
        # Load special tokens info
        specials_data = data.get("special_tokens")
        if specials_data:
            self.pad_id = specials_data["pad"]["id"]
            self.unk_id = specials_data["unk"]["id"]
            self.sos_id = specials_data["sos"]["id"]
            self.eos_id = specials_data["eos"]["id"]
        else:
            # Fallback for old format
            self.pad_id = self.stoi.get("<pad>", 0)
            self.unk_id = self.stoi.get("<unk>", 1)
            self.sos_id = self.stoi.get("<sos>", 2)
            self.eos_id = self.stoi.get("<eos>", 3)
        
        print(f"✅ Vocabulary loaded: {len(self)} tokens")
        
    def load_bpe_vocab(self, bpe_model_path: str):
        """Load BPE vocabulary from model file"""
        from .tokenizer import BPETokenizer
        self.bpe_tokenizer = BPETokenizer(bpe_model_path)
        self.build_vocab_from_bpe()