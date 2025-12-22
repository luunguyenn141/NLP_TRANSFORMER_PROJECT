import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    """Bilingual dataset for Transformer with BPE support."""

    def __init__(self, source_sentences, target_sentences, src_vocab, tgt_vocab, max_len=100):
        assert len(source_sentences) == len(target_sentences), "Source and target must have same number of lines"

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

        # Encode tokens to IDs
        # Note: Vocabulary.encode() now handles both word-level and BPE tokens
        src_ids = self.src_vocab.encode(src_tokens)
        tgt_ids = self.tgt_vocab.encode(tgt_tokens)

        # Get special token IDs
        sos_id = self.tgt_vocab.sos_id
        eos_id = self.tgt_vocab.eos_id
        
        if sos_id is None or eos_id is None:
            raise KeyError("Vocabulary must contain <sos> and <eos> tokens with IDs")

        # Add special tokens for decoder: <sos> ... <eos>
        tgt_ids = [sos_id] + tgt_ids + [eos_id]

        # Padding / truncation to fixed max_len
        pad_id = self.src_vocab.pad_id
        src_ids = self.pad_sequence(src_ids, pad_id)
        tgt_ids = self.pad_sequence(tgt_ids, pad_id)

        # Prepare decoder input and labels (shifted)
        trg_input = tgt_ids[:-1]
        trg_label = tgt_ids[1:]

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "trg_input": torch.tensor(trg_input, dtype=torch.long),
            "trg_label": torch.tensor(trg_label, dtype=torch.long),
        }

    def collate_fn(self, batch):
        """Collate list of samples into batched tensors."""
        src_batch = torch.stack([item["src"] for item in batch])
        trg_input_batch = torch.stack([item["trg_input"] for item in batch])
        trg_label_batch = torch.stack([item["trg_label"] for item in batch])

        return {
            "src": src_batch,
            "trg_input": trg_input_batch,
            "trg_label": trg_label_batch,
        }