import os
import sys
import torch

# Ensure `src/` is on sys.path so imports like `from model.transformer import Transformer` work
# when running the test from project root or other working directories.
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from model.transformer import Transformer

def test_transformer_architecture():
    print("=== BẮT ĐẦU KIỂM TRA MÔ HÌNH TRANSFORMER ===")
    
    # 1. Giả lập siêu tham số (Hyperparameters)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang chạy trên thiết bị: {device}")

    src_vocab_size = 1000  # Giả sử vocab nguồn có 1000 từ
    trg_vocab_size = 1000  # Giả sử vocab đích có 1000 từ
    src_pad_idx = 0        # Index của token padding
    trg_pad_idx = 0
    
    # Các tham số kiến trúc
    d_model = 512
    n_layers = 3     # Test thử 3 lớp cho nhẹ
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    max_len = 100

    # 2. Khởi tạo mô hình
    try:
        model = Transformer(
            src_vocab_size, trg_vocab_size, 
            src_pad_idx, trg_pad_idx,
            d_model, n_layers, n_heads, 
            d_ff, dropout, max_len
        ).to(device)
        print("✅ Khởi tạo mô hình thành công!")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo mô hình: {e}")
        return

    # 3. Tạo dữ liệu giả (Dummy Data)
    batch_size = 2
    src_len = 10
    trg_len = 12 # Lưu ý: trg_len thường khác src_len

    # Tạo tensor ngẫu nhiên (giả lập index của các từ)
    # Range từ 1 đến vocab_size (tránh số 0 vì là padding)
    src = torch.randint(1, src_vocab_size, (batch_size, src_len)).to(device)
    trg = torch.randint(1, trg_vocab_size, (batch_size, trg_len)).to(device)

    # Thử gán vài vị trí là padding để xem mask có lỗi không
    src[0, -2:] = 0  # Câu 1 trong batch bị pad 2 từ cuối
    trg[0, -1:] = 0  # Câu 1 trong batch đích bị pad 1 từ cuối

    print(f"\nInput shape (Source): {src.shape}")
    print(f"Input shape (Target): {trg.shape}")

    # 4. Chạy Forward Pass
    try:
        output = model(src, trg)
        print("✅ Forward pass chạy thành công!")
    except RuntimeError as e:
        print(f"❌ Lỗi trong quá trình Forward: {e}")
        print("Gợi ý: Kiểm tra kỹ dimension trong file encoder.py hoặc decoder.py")
        return

    # 5. Kiểm tra Output Shape
    # Output mong đợi: [batch_size, trg_len, trg_vocab_size]
    expected_shape = torch.Size([batch_size, trg_len, trg_vocab_size])
    
    print(f"\nOutput shape thực tế: {output.shape}")
    print(f"Output shape mong đợi: {expected_shape}")

    if output.shape == expected_shape:
        print("\n🎉 CHÚC MỪNG! MÔ HÌNH CỦA BẠN ĐÃ CHẠY CHUẨN VỀ MẶT KIẾN TRÚC.")
        print("Sẵn sàng để chuyển sang bước Train.")
    else:
        print("\n⚠️ CẢNH BÁO: Kích thước đầu ra không đúng!")

if __name__ == "__main__":
    test_transformer_architecture()