import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    """
    Dataset song ngữ dành cho Transformer
    """

    def __init__(self, source_sentences, target_sentences, src_vocab, tgt_vocab, max_len=100):
        assert len(source_sentences) == len(target_sentences), "Nguồn và đích phải có cùng số câu"

        self.src_sentences = source_sentences
        self.tgt_sentences = target_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def pad_sequence(self, seq, pad_id):
        if len(seq) < self.max_len:
            seq = seq + [pad_id] * (self.max_len - len(seq))
        else:
            seq = seq[: self.max_len]
        return seq

    def __getitem__(self, idx):
        src_tokens = self.src_sentences[idx]
        tgt_tokens = self.tgt_sentences[idx]

        # Encode token → ID
        src_ids = self.src_vocab.encode(src_tokens)
        tgt_ids = self.tgt_vocab.encode(tgt_tokens)

        # Padding
        src_ids = self.pad_sequence(src_ids, self.src_vocab.stoi["<pad>"])
        tgt_ids = self.pad_sequence(tgt_ids, self.tgt_vocab.stoi["<pad>"])

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long)
        }

    # -------------------------------
    # ADD THIS — collate function
    # -------------------------------
    def collate_fn(self, batch):
        """
        Gom batch thành tensor.
        batch: list[ {"src": tensor, "tgt": tensor} ]
        """
        src_batch = torch.stack([item["src"] for item in batch])
        tgt_batch = torch.stack([item["tgt"] for item in batch])

        return {
            "src": src_batch,   # (batch, seq_len)
            "tgt": tgt_batch    # (batch, seq_len)
        }
