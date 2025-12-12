import torch

class Config:
    # 1. Tham số Dữ liệu
    src_lang = 'vi'
    tgt_lang = 'en'
    max_seq_len = 100    
    vocab_size = 30000   
    
    # 2. Tham số Model
    d_model = 512        
    n_head = 8           
    n_layer = 3          # Giữ nguyên 3 là tốt cho 130k data
    d_ff = 2048          
    dropout = 0.1        
    
    # 3. Tham số Huấn luyện
    batch_size = 64      # <-- TĂNG LÊN (Quan trọng nhất)
    lr = 0.0001          # Nếu dùng NoamOpt, tham số này có thể không quan trọng lắm
    num_epochs = 20      # <-- Tăng lên để đảm bảo mô hình học đủ sâu
    
    # Thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()