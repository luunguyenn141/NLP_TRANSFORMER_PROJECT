import torch
import math

def beam_search_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device, beam_width=3, no_repeat_ngram_size=3):
    """
    Thực hiện Beam Search Decoding.
    Args:   
        model: Mô hình Transformer đã huấn luyện.
        src: Tensor đầu vào của câu nguồn [1, src_len].
        src_mask: Mask cho câu nguồn.
        max_len: Độ dài tối đa của câu đích.
        start_symbol: Chỉ số của token <sos>.
        end_symbol: Chỉ số của token <eos>.
        device: Thiết bị (CPU/GPU).
        beam_width: Số lượng nhánh trong Beam Search.
        no_repeat_ngram_size: Kích thước n-gram để chặn lặp lại.
    """
    model.eval()
    with torch.no_grad():
        enc_src = model.encoder(src, src_mask)
    
    # Beam: (score, sequence)
    beam = [(0.0, [start_symbol])] 
    
    for _ in range(max_len):
        candidates = []
        
        # Duyệt qua từng nhánh trong beam hiện tại
        for score, seq in beam:
            # 1. Nếu nhánh này đã xong (gặp EOS), giữ nguyên và đưa vào candidates
            if seq[-1] == end_symbol:
                candidates.append((score, seq))
                continue
            
            # 2. Dự đoán từ tiếp theo
            trg_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor)
            
            with torch.no_grad():
                output = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
                prob = output[:, -1, :] 
                log_prob = torch.log_softmax(prob, dim=-1) # [1, vocab_size]
            
            # --- [QUAN TRỌNG] LOGIC CHẶN LẶP TỪ ---
            # Nếu cụm từ sắp sinh ra đã từng xuất hiện trong quá khứ của câu này -> Gán điểm âm vô cùng
            if no_repeat_ngram_size > 0:
                banned_tokens = _get_banned_tokens(seq, no_repeat_ngram_size)
                # Gán log_prob của các từ bị cấm thành -inf
                if banned_tokens:
                     log_prob[0, list(banned_tokens)] = -float('inf')
            # --------------------------------------

            # 3. Lấy Top-K ứng viên tốt nhất cho nhánh này
            topk_log_probs, topk_indices = torch.topk(log_prob, beam_width)
            
            for i in range(beam_width):
                sym = topk_indices[0][i].item()
                added_score = topk_log_probs[0][i].item()
                
                # Length Penalty nhẹ (tùy chọn): Chia điểm cho độ dài để không thiên vị câu quá ngắn
                # new_score = (score + added_score) # Có thể để đơn giản như cũ
                
                candidates.append((score + added_score, seq + [sym]))
        
        # 4. Chọn lọc lại Top-K nhánh tốt nhất từ tất cả candidates
        beam = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
        
        # Kiểm tra nếu tất cả các nhánh đều đã xong
        if all(seq[-1] == end_symbol for _, seq in beam):
            break
            
    return beam[0][1]

def _get_banned_tokens(generated_seq, n):
    """
    Hàm phụ trợ: Tìm các token sẽ tạo thành n-gram lặp lại.
    Ví dụ: seq = [A, B, C, A, B], n=3. 
    Prefix hiện tại là [A, B]. Ta thấy trong quá khứ đã có [A, B] đi với C.
    => Cấm sinh ra C tiếp theo.
    """
    if len(generated_seq) < n - 1:
        return set()
    
    banned_tokens = set()
    # Lấy (n-1) từ cuối cùng làm "đuôi" để soi lại quá khứ
    ngram_prefix = tuple(generated_seq[-(n-1):])
    
    # Quét qua toàn bộ chuỗi đã sinh để xem cái "đuôi" này từng xuất hiện ở đâu
    for i in range(len(generated_seq) - n + 1):
        # Lấy thử cụm n từ cũ
        previous_ngram = tuple(generated_seq[i : i+n])
        
        # Nếu phần đầu của cụm cũ trùng với prefix hiện tại
        if previous_ngram[:-1] == ngram_prefix:
            # Thì từ cuối cùng của cụm cũ chính là từ CẤM
            banned_tokens.add(previous_ngram[-1])
            
    return banned_tokens