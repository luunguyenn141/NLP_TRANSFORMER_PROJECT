import torch
import math

def beam_search_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device, beam_width=3):
    """
    Thực hiện Beam Search Decoding.
    """
    model.eval()
    
    # 1. Encode source sentence
    with torch.no_grad():
        enc_src = model.encoder(src, src_mask)
    
    # 2. Khởi tạo beam: mỗi phần tử là (score, sequence)
    # sequence bắt đầu bằng start_symbol (SOS)
    beam = [(0.0, [start_symbol])] 
    
    for _ in range(max_len):
        candidates = []
        
        # Duyệt qua các beam hiện tại
        for score, seq in beam:
            # Nếu câu đã kết thúc bằng EOS, giữ nguyên
            if seq[-1] == end_symbol:
                candidates.append((score, seq))
                continue
            
            # Chuẩn bị input cho decoder
            trg_tensor = torch.LongTensor(seq).unsqueeze(0).to(device) # [1, seq_len]
            trg_mask = model.make_trg_mask(trg_tensor)
            
            with torch.no_grad():
                # Decoder output: [1, seq_len, vocab_size]
                output = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
                
                # Lấy output của token cuối cùng
                prob = output[:, -1, :] # [1, vocab_size]
                log_prob = torch.log_softmax(prob, dim=-1) # Chuyển về log probability
            
            # Lấy top k tokens có xác suất cao nhất cho nhánh này
            topk_log_probs, topk_indices = torch.topk(log_prob, beam_width)
            
            for i in range(beam_width):
                sym = topk_indices[0][i].item()
                added_score = topk_log_probs[0][i].item()
                
                # Cập nhật score và sequence (Score trong Beam Search thường là tổng log prob)
                candidates.append((score + added_score, seq + [sym]))
        
        # Sắp xếp các candidates theo score giảm dần và lấy top k (beam_width)
        beam = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
        
        # Kiểm tra nếu tất cả các beam đều đã kết thúc
        if all(seq[-1] == end_symbol for _, seq in beam):
            break
            
    # Trả về sequence có score cao nhất
    best_seq = beam[0][1]
    return best_seq