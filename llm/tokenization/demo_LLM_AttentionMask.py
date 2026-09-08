import torch

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 假设：
    # B = 3：batch 中有 3 个样本
    # H = 4：有 4 个 attention heads
    # L = 5：padding 后的统一 sequence length
    # ---------------------------------------------------------

    B = 3
    H = 4
    L = 5

    # ---------------------------------------------------------
    # attention_mask:
    #
    # 1 表示真实 token
    # 0 表示 padding token
    #
    # 第一句只有 4 个真实 token，所以最后一个位置是 PAD。
    # ---------------------------------------------------------

    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

    print("Attention Mask:")
    print(attention_mask)

    print("\nAttention Mask shape:")
    print(attention_mask.shape)

    # ---------------------------------------------------------
    # [B, L]
    #     ↓
    # [B, 1, 1, L]
    #
    # 中间两个维度设置为 1，
    # 是为了之后能够广播到：
    #
    # [B, H, L, L]
    #
    # 第一个 1  -> 广播到所有 attention heads
    # 第二个 1  -> 广播到所有 query positions
    # ---------------------------------------------------------

    padding_mask = attention_mask[:, None, None, :]

    print("\nPadding Mask:")
    print(padding_mask)

    print("\nPadding Mask shape:")
    print(padding_mask.shape)

    # ---------------------------------------------------------
    # 模拟 Attention 中的 attention scores。
    #
    # 实际模型中：
    #
    # attention_scores = Q @ K^T
    #
    # shape:
    #
    # [B, H, L, head_dim]
    #        @
    # [B, H, head_dim, L]
    #
    #        ↓
    #
    # [B, H, L, L]
    #
    # 这里暂时直接构造一个随机张量。
    # ---------------------------------------------------------

    attention_scores = torch.randn(B, H, L, L)

    print("\nAttention Scores shape:")
    print(attention_scores.shape)

    # ---------------------------------------------------------
    # padding_mask:
    #
    # [B, 1, 1, L]
    #
    # attention_scores:
    #
    # [B, H, L, L]
    #
    # PyTorch 会自动 broadcasting：
    #
    # [B, 1, 1, L]
    # [B, H, L, L]
    #
    #       ↓
    #
    # [B, H, L, L]
    #
    # padding 位置被填成 -inf。
    # ---------------------------------------------------------

    masked_scores = attention_scores.masked_fill(padding_mask == 0, float("-inf"))

    print("\nMasked Attention Scores shape:")
    print(masked_scores.shape)

    # ---------------------------------------------------------
    # 只看：
    #
    # 第 0 个样本
    # 第 0 个 attention head
    #
    # 它的最后一列应该全部变成 -inf，
    # 因为第 0 个样本的最后一个 key 是 PAD。
    # ---------------------------------------------------------

    print("\nSample 0, Head 0:")
    for b_index in range(B):
        for head_index in range(H):
            print(masked_scores[b_index, head_index])
