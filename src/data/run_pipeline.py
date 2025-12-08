import os
from data_processing.preprocessing import Preprocessor
from data_processing.tokenizer import SimpleTokenizer
from data_processing.vocabulary import Vocabulary
from data_processing.dataset import TranslationDataset
from data_processing.statistics import DataStatistics
from data_processing.dataloader import create_dataloader


RAW_DIR = "src/data/raw"
PROCESSED_DIR = "src/data/processed"
TOKENIZED_DIR = "src/data/processed/tokenized"
VOCAB_DIR = "src/data/vocab"


def ensure_dirs():
    for d in [PROCESSED_DIR, TOKENIZED_DIR, VOCAB_DIR]:
        os.makedirs(d, exist_ok=True)


def step1_clean_data():
    print("\n=== STEP 1: Cleaning raw data ===")
    p = Preprocessor()

    pairs = [
        ("train.en", "train.vi"),
        ("tst2012.en", "tst2012.vi"),
        ("tst2013.en", "tst2013.vi"),
    ]

    for en_file, vi_file in pairs:
        src_path = os.path.join(RAW_DIR, en_file)
        tgt_path = os.path.join(RAW_DIR, vi_file)

        src_clean, tgt_clean = p.load_parallel_data(src_path, tgt_path)

        out_src = os.path.join(PROCESSED_DIR, en_file.replace(".en", ".clean.en"))
        out_tgt = os.path.join(PROCESSED_DIR, vi_file.replace(".vi", ".clean.vi"))

        p.save_processed(src_clean, tgt_clean, out_src, out_tgt)
        print(f"Saved cleaned → {out_src}, {out_tgt}")


def step2_tokenize():
    print("\n=== STEP 2: Tokenizing cleaned data ===")
    tok = SimpleTokenizer()

    for prefix in ["train", "tst2012", "tst2013"]:
        en_path = os.path.join(PROCESSED_DIR, f"{prefix}.clean.en")
        vi_path = os.path.join(PROCESSED_DIR, f"{prefix}.clean.vi")

        with open(en_path, "r", encoding="utf-8") as f:
            en_clean = [line.strip() for line in f]

        with open(vi_path, "r", encoding="utf-8") as f:
            vi_clean = [line.strip() for line in f]

        en_tokens = [tok.tokenize(s) for s in en_clean]
        vi_tokens = [tok.tokenize(s) for s in vi_clean]

        # Save tokenized
        tok_en_path = os.path.join(TOKENIZED_DIR, f"{prefix}.tok.en")
        tok_vi_path = os.path.join(TOKENIZED_DIR, f"{prefix}.tok.vi")

        with open(tok_en_path, "w", encoding="utf-8") as f:
            for t in en_tokens:
                f.write(" ".join(t) + "\n")

        with open(tok_vi_path, "w", encoding="utf-8") as f:
            for t in vi_tokens:
                f.write(" ".join(t) + "\n")

        print(f"Saved tokenized → {tok_en_path}, {tok_vi_path}")

    return True


def step3_build_vocab():
    print("\n=== STEP 3: Building vocabulary ===")

    train_en_file = os.path.join(TOKENIZED_DIR, "train.tok.en")
    train_vi_file = os.path.join(TOKENIZED_DIR, "train.tok.vi")

    with open(train_en_file, "r", encoding="utf-8") as f:
        en_tokens = [line.split() for line in f]

    with open(train_vi_file, "r", encoding="utf-8") as f:
        vi_tokens = [line.split() for line in f]

    src_vocab = Vocabulary(min_freq=2)
    tgt_vocab = Vocabulary(min_freq=2)

    src_vocab.build_vocab(en_tokens)
    tgt_vocab.build_vocab(vi_tokens)

    # Save vocab
    src_vocab.save(os.path.join(VOCAB_DIR, "src_vocab.json"))
    tgt_vocab.save(os.path.join(VOCAB_DIR, "tgt_vocab.json"))

    print(f"Saved vocab → {VOCAB_DIR}/src_vocab.json, tgt_vocab.json")

    return src_vocab, tgt_vocab


def step4_build_dataset(src_vocab, tgt_vocab):
    print("\n=== STEP 4: Building dataset + dataloader ===")

    train_en = os.path.join(TOKENIZED_DIR, "train.tok.en")
    train_vi = os.path.join(TOKENIZED_DIR, "train.tok.vi")

    with open(train_en, "r", encoding="utf-8") as f:
        en_tokens = [line.split() for line in f]

    with open(train_vi, "r", encoding="utf-8") as f:
        vi_tokens = [line.split() for line in f]

    dataset = TranslationDataset(en_tokens, vi_tokens, src_vocab, tgt_vocab)
    dataloader = create_dataloader(dataset, batch_size=32)

    print("Dataset size:", len(dataset))
    print("Batch example:")
    batch = next(iter(dataloader))
    print(batch["src"].shape, batch["tgt"].shape)

    return dataset, dataloader


def step5_statistics():
    print("\n=== STEP 5: Data Statistics ===")
    stats = DataStatistics(
        raw_data_dir=RAW_DIR,
        save_report_dir="src/data/processed/statistics"
    )
    stats.run()


if __name__ == "__main__":
    ensure_dirs()
    step1_clean_data()
    step2_tokenize()
    src_vocab, tgt_vocab = step3_build_vocab()
    step4_build_dataset(src_vocab, tgt_vocab)
    step5_statistics()

    print("\n===== PIPELINE COMPLETED SUCCESSFULLY =====")
