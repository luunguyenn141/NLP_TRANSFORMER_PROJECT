import torch

class Config:
    # 1. Tham số Dữ liệu
    src_lang = 'en'
    tgt_lang = 'vi'
    max_seq_len = 150   
    vocab_size = 30000   
    
    # 2. Tham số Model
    d_model = 512        
    n_head = 8           
    n_layer = 3          
    d_ff = 2048          
    dropout = 0.1        
    
    # 3. Tham số Huấn luyện
    batch_size = 64      
    lr = 0.0001          # Nếu dùng NoamOpt, tham số này có thể không quan trọng lắm
    num_epochs = 20      # <-- Tăng lên để đảm bảo mô hình học đủ sâu
    
    # Thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()