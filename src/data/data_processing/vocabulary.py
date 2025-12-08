import json
from collections import Counter

class Vocabulary:
    def __init__(self, min_freq=2, max_size=30000, specials=None):
        """
        Vocabulary for NLP tasks.

        Args:
            min_freq: minimum frequency to keep a token
            max_size: maximum vocabulary size
            specials: list of special tokens
        """
        self.min_freq = min_freq
        self.max_size = max_size

        # Special tokens
        if specials is None:
            specials = ["<pad>", "<unk>", "<sos>", "<eos>"]
        self.specials = specials

        # Token → ID
        self.stoi = {}
        # ID → Token
        self.itos = []

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

        # Add tokens above min_freq and up to max_size
        for tok, freq in counter.most_common():
            if freq < self.min_freq:
                continue
            if len(self.itos) >= self.max_size:
                break
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)

    def __len__(self):
        return len(self.itos)

    def token_to_id(self, token):
        return self.stoi.get(token, self.stoi["<unk>"])

    def id_to_token(self, idx):
        if idx < 0 or idx >= len(self.itos):
            return "<unk>"
        return self.itos[idx]

    def encode(self, tokens):
        return [self.token_to_id(tok) for tok in tokens]

    def decode(self, ids):
        return [self.id_to_token(i) for i in ids]

    def save(self, filepath):
        """Save vocabulary to JSON file"""
        data = {
            "itos": self.itos,
            "stoi": self.stoi,
            "min_freq": self.min_freq,
            "max_size": self.max_size
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