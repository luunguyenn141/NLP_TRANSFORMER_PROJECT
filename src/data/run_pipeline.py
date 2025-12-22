#!/usr/bin/env python3
"""
File điều phối toàn bộ pipeline xử lý dữ liệu cho bài toán dịch máy Transformer.
Cập nhật để hỗ trợ cả word-level và BPE tokenization.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from configs.config import cfg
from src.data.data_processing.preprocessing import Preprocessor
from src.data.data_processing.tokenizer import BPETokenizer, BPETokenizerManager, SimpleTokenizer
from src.data.data_processing.vocabulary import Vocabulary
from src.data.data_processing.statistics import DataStatistics
from src.data.data_processing.dataset import TranslationDataset
from src.data.data_processing.dataloader import create_dataloader

class DataPipeline:
    def __init__(self, config, use_bpe=True):
        """
        Args:
            config: Configuration object
            use_bpe: Use BPE tokenization if True, else word-level
        """
        self.config = config
        self.use_bpe = use_bpe
        self.preprocessor = Preprocessor(lowercase=True)
        
        # Paths
        self.raw_dir = Path("src/data/raw")
        self.processed_dir = Path("src/data/processed")
        self.tokenized_dir = Path("src/data/processed/tokenized")
        self.vocab_dir = Path("src/data/vocab")
        self.statistics_dir = Path("src/data/processed/statistics")
        
        # Create directories
        for dir_path in [self.processed_dir, self.tokenized_dir, self.vocab_dir, self.statistics_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Tokenizers
        self.src_tokenizer = None
        self.tgt_tokenizer = None
        
    def preprocess(self):
        """Step 1: Clean raw data"""
        print("\n" + "="*50)
        print("BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU THÔ")
        print("="*50)
        
        datasets = ["train", "tst2012", "tst2013"]
        
        for dataset in datasets:
            src_path = self.raw_dir / f"{dataset}.{self.config.src_lang}"
            tgt_path = self.raw_dir / f"{dataset}.{self.config.tgt_lang}"
            
            if not src_path.exists() or not tgt_path.exists():
                print(f"⚠️  File không tồn tại: {src_path} hoặc {tgt_path}")
                continue
                
            print(f"📄 Đang xử lý {dataset}...")
            src_cleaned, tgt_cleaned = self.preprocessor.load_parallel_data(
                str(src_path), str(tgt_path)
            )
            
            # Save cleaned data
            src_out = self.processed_dir / f"{dataset}.clean.{self.config.src_lang}"
            tgt_out = self.processed_dir / f"{dataset}.clean.{self.config.tgt_lang}"
            self.preprocessor.save_processed(src_cleaned, tgt_cleaned, str(src_out), str(tgt_out))
            
            print(f"✅ Đã xử lý {dataset}: {len(src_cleaned)} cặp câu")
            
    def tokenize(self):
        """Step 2: Tokenize data"""
        print("\n" + "="*50)
        print("BƯỚC 2: TOKENIZATION")
        print(f"Phương pháp: {'BPE' if self.use_bpe else 'Word-level'}")
        print("="*50)
        
        if self.use_bpe:
            self._tokenize_bpe()
        else:
            self._tokenize_word_level()
    
    def _tokenize_bpe(self):
        """Tokenize using BPE"""
        print("🔄 Tokenizing với BPE...")
        
        # Check if BPE models already exist
        bpe_models_dir = self.vocab_dir / "bpe_models"
        bpe_models_dir.mkdir(exist_ok=True)
        
        vi_model_path = bpe_models_dir / "bpe_vi.model"
        en_model_path = bpe_models_dir / "bpe_en.model"
        
        # Train BPE models if they don't exist
        if not vi_model_path.exists() or not en_model_path.exists():
            print("🔄 Training BPE models...")
            
            # Prepare text files for BPE training
            all_vi_text = []
            all_en_text = []
            
            datasets = ["train", "tst2012", "tst2013"]
            for dataset in datasets:
                src_path = self.processed_dir / f"{dataset}.clean.{self.config.src_lang}"
                tgt_path = self.processed_dir / f"{dataset}.clean.{self.config.tgt_lang}"
                
                if src_path.exists():
                    with open(src_path, 'r', encoding='utf-8') as f:
                        all_vi_text.extend(f.readlines())
                
                if tgt_path.exists():
                    with open(tgt_path, 'r', encoding='utf-8') as f:
                        all_en_text.extend(f.readlines())
            
            # Write combined text files
            vi_text_file = bpe_models_dir / "vi_all_text.txt"
            en_text_file = bpe_models_dir / "en_all_text.txt"
            
            with open(vi_text_file, 'w', encoding='utf-8') as f:
                f.writelines(all_vi_text)
            
            with open(en_text_file, 'w', encoding='utf-8') as f:
                f.writelines(all_en_text)
            
            # Train BPE
            tokenizer_manager = BPETokenizerManager(self.config)
            vi_model_path_str, en_model_path_str = tokenizer_manager.train(
                str(vi_text_file),
                str(en_text_file),
                vocab_size=self.config.vocab_size
            )
            
            # Load tokenizers
            self.src_tokenizer, self.tgt_tokenizer = tokenizer_manager.load(
                vi_model_path_str,
                en_model_path_str
            )
        else:
            print("✅ Loading existing BPE models...")
            self.src_tokenizer = BPETokenizer(str(vi_model_path), lowercase=True)
            self.tgt_tokenizer = BPETokenizer(str(en_model_path), lowercase=True)
        
        # Tokenize all datasets
        datasets = ["train", "tst2012", "tst2013"]
        for dataset in datasets:
            src_path = self.processed_dir / f"{dataset}.clean.{self.config.src_lang}"
            tgt_path = self.processed_dir / f"{dataset}.clean.{self.config.tgt_lang}"
            
            if not src_path.exists() or not tgt_path.exists():
                print(f"⚠️  File không tồn tại: {src_path} hoặc {tgt_path}")
                continue
            
            print(f"📄 Tokenizing {dataset}...")
            
            # Read cleaned data
            with open(src_path, 'r', encoding='utf-8') as f:
                src_sentences = [line.strip() for line in f]
            
            with open(tgt_path, 'r', encoding='utf-8') as f:
                tgt_sentences = [line.strip() for line in f]
            
            # Tokenize
            src_tokenized = []
            tgt_tokenized = []
            
            for src_sent, tgt_sent in zip(src_sentences, tgt_sentences):
                src_tokens = self.src_tokenizer.tokenize(src_sent)
                tgt_tokens = self.tgt_tokenizer.tokenize(tgt_sent)
                
                # Filter too long sentences
                if len(src_tokens) <= self.config.max_seq_len and len(tgt_tokens) <= self.config.max_seq_len:
                    src_tokenized.append(src_tokens)
                    tgt_tokenized.append(tgt_tokens)
            
            # Save tokenized data
            src_out = self.tokenized_dir / f"{dataset}.bpe.{self.config.src_lang}"
            tgt_out = self.tokenized_dir / f"{dataset}.bpe.{self.config.tgt_lang}"
            
            with open(src_out, 'w', encoding='utf-8') as f:
                for tokens in src_tokenized:
                    f.write(' '.join(tokens) + '\n')
            
            with open(tgt_out, 'w', encoding='utf-8') as f:
                for tokens in tgt_tokenized:
                    f.write(' '.join(tokens) + '\n')
            
            print(f"✅ {dataset}: {len(src_tokenized)} cặp câu đã tokenized")
    
    def _tokenize_word_level(self):
        """Tokenize using word-level tokenizer"""
        print("🔄 Tokenizing với word-level...")
        
        tokenizer = SimpleTokenizer(lowercase=True)
        
        datasets = ["train", "tst2012", "tst2013"]
        for dataset in datasets:
            src_path = self.processed_dir / f"{dataset}.clean.{self.config.src_lang}"
            tgt_path = self.processed_dir / f"{dataset}.clean.{self.config.tgt_lang}"
            
            if not src_path.exists() or not tgt_path.exists():
                print(f"⚠️  File không tồn tại: {src_path} hoặc {tgt_path}")
                continue
            
            print(f"📄 Tokenizing {dataset}...")
            
            # Read and tokenize
            src_tokenized = []
            tgt_tokenized = []
            
            with open(src_path, 'r', encoding='utf-8') as f_src, \
                 open(tgt_path, 'r', encoding='utf-8') as f_tgt:
                
                for src_line, tgt_line in zip(f_src, f_tgt):
                    src_tokens = tokenizer.tokenize(src_line.strip())
                    tgt_tokens = tokenizer.tokenize(tgt_line.strip())
                    
                    # Filter too long sentences
                    if len(src_tokens) <= self.config.max_seq_len and len(tgt_tokens) <= self.config.max_seq_len:
                        src_tokenized.append(src_tokens)
                        tgt_tokenized.append(tgt_tokens)
            
            # Save tokenized data
            src_out = self.tokenized_dir / f"{dataset}.tok.{self.config.src_lang}"
            tgt_out = self.tokenized_dir / f"{dataset}.tok.{self.config.tgt_lang}"
            
            with open(src_out, 'w', encoding='utf-8') as f:
                for tokens in src_tokenized:
                    f.write(' '.join(tokens) + '\n')
            
            with open(tgt_out, 'w', encoding='utf-8') as f:
                for tokens in tgt_tokenized:
                    f.write(' '.join(tokens) + '\n')
            
            print(f"✅ {dataset}: {len(src_tokenized)} cặp câu đã tokenized")
    
    def build_vocab(self):
        """Step 3: Build vocabulary"""
        print("\n" + "="*50)
        print("BƯỚC 3: XÂY DỰNG VOCABULARY")
        print("="*50)
        
        if self.use_bpe and self.src_tokenizer and self.tgt_tokenizer:
            # Build vocabulary from BPE tokenizers
            print("🔄 Building vocabulary từ BPE tokenizers...")
            
            # Source vocabulary
            src_vocab = Vocabulary(bpe_tokenizer=self.src_tokenizer)
            src_vocab.build_vocab_from_bpe()
            src_vocab.save(str(self.vocab_dir / "src_vocab_bpe.json"))
            
            # Target vocabulary
            tgt_vocab = Vocabulary(bpe_tokenizer=self.tgt_tokenizer)
            tgt_vocab.build_vocab_from_bpe()
            tgt_vocab.save(str(self.vocab_dir / "tgt_vocab_bpe.json"))
            
            print(f"✅ Source vocab size: {len(src_vocab)}")
            print(f"✅ Target vocab size: {len(tgt_vocab)}")
            
        else:
            # Build vocabulary from tokenized files (word-level)
            print("🔄 Building vocabulary từ tokenized files...")
            
            # Load tokenized training data
            train_src_path = self.tokenized_dir / f"train.tok.{self.config.src_lang}"
            train_tgt_path = self.tokenized_dir / f"train.tok.{self.config.tgt_lang}"
            
            if not train_src_path.exists() or not train_tgt_path.exists():
                print(f"⚠️  Tokenized files not found: {train_src_path} hoặc {train_tgt_path}")
                return
            
            # Read tokenized data
            src_tokenized = []
            with open(train_src_path, 'r', encoding='utf-8') as f:
                for line in f:
                    src_tokenized.append(line.strip().split())
            
            tgt_tokenized = []
            with open(train_tgt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    tgt_tokenized.append(line.strip().split())
            
            # Build source vocabulary
            src_vocab = Vocabulary(min_freq=2, max_size=self.config.vocab_size)
            src_vocab.build_vocab(src_tokenized)
            src_vocab.save(str(self.vocab_dir / "src_vocab.json"))
            
            # Build target vocabulary
            tgt_vocab = Vocabulary(min_freq=2, max_size=self.config.vocab_size)
            tgt_vocab.build_vocab(tgt_tokenized)
            tgt_vocab.save(str(self.vocab_dir / "tgt_vocab.json"))
            
            print(f"✅ Source vocab size: {len(src_vocab)}")
            print(f"✅ Target vocab size: {len(tgt_vocab)}")
    
    def calculate_statistics(self):
        """Step 4: Calculate statistics"""
        print("\n" + "="*50)
        print("BƯỚC 4: THỐNG KÊ DỮ LIỆU")
        print("="*50)
        
        analyzer = DataStatistics(
            raw_data_dir=str(self.raw_dir),
            save_report_dir=str(self.statistics_dir)
        )
        analyzer.run()
        
        print("✅ Statistics calculated and saved")
    
    def create_datasets(self):
        """Step 5: Create datasets and dataloaders (for testing)"""
        print("\n" + "="*50)
        print("BƯỚC 5: TẠO DATASETS (TEST)")
        print("="*50)
        
        # Load vocabularies
        vocab_files = {
            'bpe': ("src_vocab_bpe.json", "tgt_vocab_bpe.json"),
            'word': ("src_vocab.json", "tgt_vocab.json")
        }
        
        vocab_key = 'bpe' if self.use_bpe else 'word'
        src_vocab_file, tgt_vocab_file = vocab_files[vocab_key]
        
        src_vocab = Vocabulary()
        src_vocab.load(str(self.vocab_dir / src_vocab_file))
        
        tgt_vocab = Vocabulary()
        tgt_vocab.load(str(self.vocab_dir / tgt_vocab_file))
        
        # Load tokenized data
        token_ext = 'bpe' if self.use_bpe else 'tok'
        
        train_src_path = self.tokenized_dir / f"train.{token_ext}.{self.config.src_lang}"
        train_tgt_path = self.tokenized_dir / f"train.{token_ext}.{self.config.tgt_lang}"
        
        # Read tokenized sentences
        src_sentences = []
        with open(train_src_path, 'r', encoding='utf-8') as f:
            for line in f:
                src_sentences.append(line.strip().split())
        
        tgt_sentences = []
        with open(train_tgt_path, 'r', encoding='utf-8') as f:
            for line in f:
                tgt_sentences.append(line.strip().split())
        
        # Create dataset
        dataset = TranslationDataset(
            source_sentences=src_sentences[:10],  # Test với 10 câu
            target_sentences=tgt_sentences[:10],
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            max_len=self.config.max_seq_len
        )
        
        # Create dataloader
        dataloader = create_dataloader(dataset, batch_size=4, shuffle=True)
        
        # Test one batch
        print("🧪 Testing dataset and dataloader...")
        for batch in dataloader:
            print(f"✅ Batch shapes:")
            print(f"   src: {batch['src'].shape}")
            print(f"   trg_input: {batch['trg_input'].shape}")
            print(f"   trg_label: {batch['trg_label'].shape}")
            break
        
        print("✅ Datasets created successfully!")
    
    def run(self, steps=None):
        """
        Run the complete pipeline or specific steps
        
        Args:
            steps: List of steps to run (1-5). If None, run all.
        """
        if steps is None:
            steps = [1, 2, 3, 4, 5]
        
        step_functions = {
            1: self.preprocess,
            2: self.tokenize,
            3: self.build_vocab,
            4: self.calculate_statistics,
            5: self.create_datasets
        }
        
        for step in steps:
            if step in step_functions:
                print(f"\n🎯 Running step {step}...")
                try:
                    step_functions[step]()
                except Exception as e:
                    print(f"❌ Error in step {step}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️  Step {step} không tồn tại")
        
        print("\n" + "="*50)
        print("🎉 PIPELINE HOÀN TẤT!")
        print("="*50)

def main():
    """Main function to run the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run data processing pipeline for Transformer')
    parser.add_argument('--use-bpe', action='store_true', 
                       help='Use BPE tokenization (default: True)')
    parser.add_argument('--steps', type=str, default='1,2,3,4,5',
                       help='Comma-separated steps to run (1-5)')
    
    args = parser.parse_args()
    
    # Parse steps
    steps = [int(s.strip()) for s in args.steps.split(',')]
    
    # Create and run pipeline
    pipeline = DataPipeline(cfg, use_bpe=args.use_bpe)
    pipeline.run(steps=steps)

if __name__ == "__main__":
    main()