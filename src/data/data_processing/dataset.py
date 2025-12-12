import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    """Bilingual dataset for Transformer.

    Returns dicts with keys: 'src', 'trg_input', 'trg_label'.
    """

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

        src_ids = self.src_vocab.encode(src_tokens)
        tgt_ids = self.tgt_vocab.encode(tgt_tokens)

        # Add special tokens for decoder: <sos> ... <eos>
        sos_id = self.tgt_vocab.stoi.get("<sos>")
        eos_id = self.tgt_vocab.stoi.get("<eos>")
        if sos_id is None or eos_id is None:
            raise KeyError("Vocabulary must contain <sos> and <eos> tokens")

        tgt_ids = [sos_id] + tgt_ids + [eos_id]

        # Padding / truncation to fixed max_len
        src_ids = self.pad_sequence(src_ids, self.src_vocab.stoi["<pad>"])
        tgt_ids = self.pad_sequence(tgt_ids, self.tgt_vocab.stoi["<pad>"])

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
