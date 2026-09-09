import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    torch.manual_seed(42)
    # ---------------------------------------------------------
    # B = batch_size
    # H = number of attention heads
    # L = sequence length
    #
    # Attention Scores 的 shape：
    #
    # [B, H, L, L]
    #
    # 最后两个 L：
    #
    # 第一个 L -> Query positions
    # 第二个 L -> Key positions
    # ---------------------------------------------------------
    
    B = 3
    H = 4
    L = 5

    # ---------------------------------------------------------
    # attention_mask:
    #
    # 1 -> 真实 token
    # 0 -> padding token
    #
    # 第 0 个样本只有 4 个真实 token，
    # 所以最后一个位置是 PAD。
    #
    # shape:
    #
    # [B, L]
    # ---------------------------------------------------------
    
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0], 
            [1, 1, 1, 1, 1], 
            [1, 1, 1, 1, 1]
        ],
        device = device
    )

    print("Attention Mask:")
    print(attention_mask)

    print("\nAttention Mask shape:")
    print(attention_mask.shape)
    
    # ---------------------------------------------------------
    # Padding Mask
    #
    # [B, L]
    #    ↓
    # [B, 1, 1, L]
    #
    # 中间两个维度设置为 1，
    # 是为了之后能够 broadcasting 到：
    #
    # [B, H, L, L]
    #
    # 第一个 1：
    #   对所有 attention heads 使用相同的 padding mask
    #
    # 第二个 1：
    #   对所有 query positions 使用相同的 padding mask
    #
    # 最后一个 L：
    #   表示哪些 Key positions 是 PAD
    # ---------------------------------------------------------

    padding_mask = attention_mask[:, None, None, :]

    print("\nPadding Mask:")
    print(padding_mask)

    print("\nPadding Mask shape:")
    print(padding_mask.shape)

    # ---------------------------------------------------------
    # 模拟 Attention Scores
    #
    # 实际 Attention 中：
    #
    # Q:
    # [B, H, L, D_h]
    #
    # K^T:
    # [B, H, D_h, L]
    #
    # Q @ K^T
    #      ↓
    #
    # [B, H, L, L]
    #
    # 这里直接生成随机 attention scores。
    # ---------------------------------------------------------

    attention_scores = torch.randn(B, H, L, L, device = device)

    print("\nAttention Scores shape:")
    print(attention_scores.shape)

    print("\nAttention Scores - Sample 0, Head 0:")
    print(attention_scores[0, 0])
    
    # =========================================================
    # Part 1
    # Padding Mask
    # =========================================================

    # ---------------------------------------------------------
    # Padding Mask 的作用：
    #
    # 不允许 Attention 读取 PAD Key。
    #
    # padding_mask:
    #
    # [B, 1, 1, L]
    #
    # attention_scores:
    #
    # [B, H, L, L]
    #
    # PyTorch 会自动 broadcasting。
    #
    # PAD 对应的位置填成 -inf。
    #
    # 注意：
    #
    # 不能简单乘以 0。
    #
    # 因为：
    #
    # exp(0) = 1
    #
    # softmax 后仍然会得到非零概率。
    #
    # 而：
    #
    # exp(-inf) = 0
    #
    # 所以 softmax 后 PAD 的 attention weight
    # 才会真正变成 0。
    # ---------------------------------------------------------

    padding_scores = attention_scores.masked_fill(
        padding_mask == 0,
        float("-inf")
    )

    print("\nAfter Padding Mask - Sample 0, Head 0:")
    print(padding_mask[0, 0])

    # ---------------------------------------------------------
    # 对最后一个维度进行 softmax。
    #
    # 最后一个维度就是 Key positions。
    #
    # 对于每一个 Query：
    #
    # Attention 都会在所有 Key 之间形成一个概率分布。
    #
    # ---------------------------------------------------------

    padding_attention = torch.softmax(padding_scores, dim = -1)
    print("\nAfter Softmax - Sample 0, Head 0:")
    print(padding_attention[0, 0])

    # ---------------------------------------------------------
    # 第 0 个样本最后一个 Key 是 PAD。
    #
    # padding_attention:
    #
    # [B, H, Query, Key]
    #
    # [0, 0, :, -1]
    #
    # 0  -> 第 0 个样本
    # 0  -> 第 0 个 Head
    # :  -> 所有 Query positions
    # -1 -> 最后一个 Key position
    #
    # 所以这里应该全部为 0。
    # ---------------------------------------------------------

    print("\nPAD Attention Weights - Sample 0, Head 0:")
    print(padding_attention[0, 0, :, -1])

    # =========================================================
    # Part 2
    # Causal Mask
    # =========================================================

    # ---------------------------------------------------------
    # Decoder-only LLM 不能看到未来 token。
    #
    # 假设 L = 5：
    #
    #            Key
    #
    # Query 0:  ✓  X  X  X  X
    # Query 1:  ✓  ✓  X  X  X
    # Query 2:  ✓  ✓  ✓  X  X
    # Query 3:  ✓  ✓  ✓  ✓  X
    # Query 4:  ✓  ✓  ✓  ✓  ✓
    #
    # torch.triu(..., diagonal = 1)
    #
    # 生成严格上三角区域。
    #
    # True 表示：
    #
    # 这个位置需要被 Mask。
    # ---------------------------------------------------------
    
    causal_mask = torch.triu(
        torch.ones(L, L, dtype = torch.bool, device = device),
        diagonal = 1
    )

    # ---------------------------------------------------------
    # [L, L]
    #    ↓
    # [1, 1, L, L]
    #
    # 第一个 1：
    #   对所有 batch samples 共用
    #
    # 第二个 1：
    #   对所有 attention heads 共用
    #
    # broadcasting 后：
    #
    # [1, 1, L, L]
    #       ↓
    # [B, H, L, L]
    # ---------------------------------------------------------
    
    causal_mask = causal_mask[None, None, :, :]

    print("\nCausal Mask:")
    print(causal_mask[0, 0].int())

    print("\nCausal Mask shape:")
    print(causal_mask.shape)

    # =========================================================
    # Part 3
    # Padding Mask + Causal Mask
    # =========================================================

    # ---------------------------------------------------------
    # 前面的 padding_scores 已经完成：
    #
    # attention_scores
    #       ↓
    # Padding Mask
    #       ↓
    # padding_scores
    #
    # 所以这里不需要重新做一次 Padding Mask。
    #
    # 直接在 padding_scores 的基础上，
    # 再加入 Causal Mask。
    #
    # ---------------------------------------------------------
    
    masked_scores = padding_scores.masked_fill(
        causal_mask,
        float('-inf')
    )

    print("\nPadding + Causal Mask - Sample 0, Head 0:")
    print(masked_scores[0, 0])
    
    # ---------------------------------------------------------
    # 最后进行 softmax。
    #
    # 所有被 Padding Mask 或 Causal Mask
    # 屏蔽的位置：
    #
    # score = -inf
    #
    # 因此：
    #
    # exp(-inf) = 0
    #
    # 最终：
    #
    # attention weight = 0
    # ---------------------------------------------------------

    attention = torch.softmax(
        masked_scores,
        dim = -1
    )

    print("\nFinal Attention Weights - Sample 0, Head 0:")
    print(attention[0, 0])
