import os
import json
from collections import Counter


class DataStatistics:
    def __init__(self, raw_data_dir, save_report_dir=None):
        """
        raw_data_dir: thư mục chứa train.en, train.vi, tst....en, tst....vi
        save_report_dir: thư mục lưu báo cáo (optional)
        """
        self.raw_data_dir = raw_data_dir
        self.save_report_dir = save_report_dir

        if save_report_dir and not os.path.exists(save_report_dir):
            os.makedirs(save_report_dir)

    def read_file(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def get_sentence_lengths(self, sentences):
        """
        Trả về danh sách tuple: (len_words, len_chars)
        """
        return [(len(s.split()), len(s)) for s in sentences]

    def compute_statistics(self, sentences):
        lengths = self.get_sentence_lengths(sentences)

        word_lengths = [w for (w, _) in lengths]
        char_lengths = [c for (_, c) in lengths]

        stats = {
            "num_sentences": len(sentences),
            "avg_words": sum(word_lengths) / len(word_lengths) if word_lengths else 0,
            "avg_chars": sum(char_lengths) / len(char_lengths) if char_lengths else 0,
            "max_words": max(word_lengths) if word_lengths else 0,
            "min_words": min(word_lengths) if word_lengths else 0,
            "max_chars": max(char_lengths) if char_lengths else 0,
            "min_chars": min(char_lengths) if char_lengths else 0,
            "word_length_distribution": dict(Counter(word_lengths)),
            "char_length_distribution": dict(Counter(char_lengths)),
        }

        return stats

    def process_dataset(self, prefix):
        """
        prefix: train | tst2012 | tst2013
        """
        en_path = os.path.join(self.raw_data_dir, f"{prefix}.en")
        vi_path = os.path.join(self.raw_data_dir, f"{prefix}.vi")

        if not os.path.exists(en_path) or not os.path.exists(vi_path):
            print(f"[WARNING] Missing dataset: {prefix}")
            return None

        en_sentences = self.read_file(en_path)
        vi_sentences = self.read_file(vi_path)

        stats = {
            "english": self.compute_statistics(en_sentences),
            "vietnamese": self.compute_statistics(vi_sentences),
        }

        if self.save_report_dir:
            save_path = os.path.join(self.save_report_dir, f"{prefix}_stats.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=4, ensure_ascii=False)
            print(f"Saved report → {save_path}")

        return stats

    def run(self):
        prefixes = ["train", "tst2012", "tst2013"]
        all_stats = {}

        for p in prefixes:
            print(f"\nProcessing dataset: {p}")
            stats = self.process_dataset(p)
            if stats:
                all_stats[p] = stats

        return all_stats


if __name__ == "__main__":
    RAW_DIR = "src/data/raw"
    SAVE_DIR = "src/data/processed/statistics"

    analyzer = DataStatistics(raw_data_dir=RAW_DIR, save_report_dir=SAVE_DIR)
    report = analyzer.run()

    print("\n===== Hoàn thành thống kê dữ liệu =====")
