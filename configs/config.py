import torch

class Config:
    # 1. Tham số Dữ liệu
    src_lang = 'vi'
    tgt_lang = 'en'
    max_seq_len = 100    
    vocab_size = 30000   # Sẽ được B cập nhật sau
    
    # 2. Tham số Model
    d_model = 512        # Kích thước Embedding và Model
    n_head = 8           # Số lượng đầu Attention
    n_layer = 6          # Số lớp Encoder/Decoder
    d_ff = 2048          # Kích thước lớp ẩn trong FFN
    dropout = 0.1        
    
    # 3. Tham số Huấn luyện
    batch_size = 32
    lr = 0.0001
    num_epochs = 20
    
    # Thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()