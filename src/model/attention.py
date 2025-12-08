import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        # d_model: Kích thước vector của từ (ví dụ 512)
        # n_head: Số lượng đầu (ví dụ 8)
        
        # Kiểm tra xem d_model có chia hết cho số head không
        assert d_model % n_head == 0, "d_model phải chia hết cho n_head"
        
        self.d_head = d_model // n_head # Kích thước của mỗi head
        self.d_model = d_model
        self.n_head = n_head
        
        # Các lớp Linear để tạo ra Q, K, V
        # Công thức: Linear(input_dim, output_dim)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Lớp Linear cuối cùng để tổng hợp kết quả
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # q, k, v có kích thước: (batch_size, seq_len, d_model)
        batch_size = q.size(0)

        # --- BƯỚC 1: TÍNH Q, K, V VÀ TÁCH HEAD ---
        # 1. Qua lớp Linear
        # 2. Reshape để tách thành n_head
        # 3. Transpose để đảo n_head lên trước seq_len để tính toán song song
        # Kích thước sau cùng: (batch_size, n_head, seq_len, d_head)
        
        Q = self.w_q(q).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

        # --- BƯỚC 2: TÍNH SCALED DOT-PRODUCT ATTENTION ---
        # Công thức: softmax(Q * K^T / sqrt(d_k))
        
        # Nhân ma trận Q với K chuyển vị (transpose 2 chiều cuối)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Chia cho căn bậc 2 của d_head để ổn định gradient
        scores = scores / math.sqrt(self.d_head)

        # --- BƯỚC 3: ÁP DỤNG MASK (QUAN TRỌNG) ---
        # Nếu có mask, ta gán giá trị tại vị trí mask=0 thành âm vô cùng (-1e9)
        # Để khi qua Softmax nó sẽ bằng 0 (nghĩa là không thèm nhìn vào từ đó)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # --- BƯỚC 4: SOFTMAX VÀ NHÂN VỚI V ---
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Output shape: (batch_size, n_head, seq_len, d_head)
        output = torch.matmul(attn_weights, V)

        # --- BƯỚC 5: GHÉP CÁC HEAD LẠI (CONCAT) ---
        # Đảo lại vị trí dimensions: (batch_size, seq_len, n_head, d_head)
        # Sau đó ép phẳng (flatten) 2 chiều cuối lại thành d_model
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # --- BƯỚC 6: QUA LỚP LINEAR CUỐI ---
        # Trả về cả output và attention weights để các lớp Encoder/Decoder
        # có thể (tuỳ chọn) truy cập attention maps.
        return self.w_o(output), attn_weights

# ==========================================
# (UNIT TEST)
# ==========================================
if __name__ == "__main__":
    # Giả lập tham số
    d_model = 512
    n_head = 8
    seq_len = 10
    batch_size = 2

    print("--- TESTING MODULE ATTENTION ---")
    try:
        # Tạo dữ liệu giả ngẫu nhiên
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Khởi tạo mô hình
        attention = MultiHeadAttention(d_model, n_head)
        
        # Chạy thử
        output, attn = attention(x, x, x) # Self-attention
        
        print(f"✅ Input size: {x.shape}")
        print(f"✅ Output size: {output.shape}")
        print(f"✅ Attn size: {attn.shape}")
        
        if output.shape == (batch_size, seq_len, d_model):
            print("DONE")
        else:
            print("ERROR")
            
    except Exception as e:
        print(f"Error: {e}")