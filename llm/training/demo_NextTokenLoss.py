import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from llm.modeling.demo_MiniLLM import MiniLLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def next_token_loss(logits, input_ids):
    # Logits 去掉最后一个位置:
    # [B, L, V] -> [B, L - 1, V]
    # [B, L] -> [B, L - 1]
    shift_logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]

    vocab_size = logits.shape[-1]

    # ------------------------------------------------------------
    # 5. CrossEntropy Loss
    #
    # 当前：
    #
    # shift_logits: [B, L-1, V]
    # targets:      [B, L-1]
    #
    # CrossEntropy 可以把“每一个 token 位置”
    # 看成一个独立的 V 类分类任务。
    #
    # 因此把 B 和 L-1 合并成样本维 N：
    #
    # [B, L-1, V]
    #      ↓ reshape
    # [B*(L-1), V]
    #
    # [B, L-1]
    #      ↓ reshape
    # [B*(L-1)]
    #
    # 即：
    # prediction: [N, V]
    # target:     [N]
    # ------------------------------------------------------------

    # 1. 对每行 logits 做 LogSoftmax
    # 2. 根据 target[i] 找到正确类别的 log probability
    # 3. 取负值：
    #
    #       loss_i = -log P(correct_class)
    #
    # 4. 对 N 个样本的 loss 默认取平均

    loss = F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        targets.reshape(-1)
    )

    return loss

def next_token_loss_with_mask(logits, input_ids, attention_mask):
    # =========================================================
    # Part 1
    # Next-token Prediction Shift
    # =========================================================
    #
    # 假设原始序列：
    #
    # input_ids:
    #
    # [A, B, C, D, PAD]
    #
    # Next-token Prediction 实际上是：
    #
    # A -> B
    # B -> C
    # C -> D
    # D -> PAD
    #
    # 所以：
    #
    # logits 去掉最后一个位置：
    #
    # [B, L, V]
    #      ↓
    # [B, L-1, V]
    #
    # targets 去掉第一个位置：
    #
    # [B, L]
    #      ↓
    # [B, L-1]
    # =========================================================
    
    shift_logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]

    vocab_size = logits.shape[-1]

    # =========================================================
    # Part 2
    # Loss Mask
    # =========================================================
    #
    # attention_mask:
    #
    # [1, 1, 1, 1, 0]
    #
    # targets:
    #
    # [B, C, D, PAD]
    #
    # 因此 Loss Mask 必须和 targets 对齐：
    #
    # attention_mask[:, 1:]
    #
    #      ↓
    #
    # [1, 1, 1, 0]
    #
    # 表示：
    #
    # A -> B      计算 Loss
    # B -> C      计算 Loss
    # C -> D      计算 Loss
    # D -> PAD    不计算 Loss
    #
    # 注意：
    #
    # 这里是 [:, 1:]
    #
    # 不是 [:, :-1]
    #
    # 因为我们要判断的是：
    #
    # target 是不是 PAD。
    # =========================================================

    loss_mask = attention_mask[:, 1:]

    # =========================================================
    # Part 3
    # Per-token CrossEntropy Loss
    # =========================================================
    #
    # reduction="none"
    #
    # 非常重要。
    #
    # 原来的 CrossEntropy 默认：
    #
    # reduction="mean"
    #
    # 会直接把所有位置的 loss 平均，
    # 这样我们就没机会把 PAD loss 去掉。
    #
    # 所以这里先保留每一个 token 的 loss。
    # =========================================================
    
    token_losses = F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction = "none"
    )

    # ---------------------------------------------------------
    # 当前：
    #
    # token_losses:
    #
    # [B * (L - 1)]
    #
    # 恢复成：
    #
    # [B, L - 1]
    #
    # 这样就和 loss_mask 的 shape 一样了。
    # ---------------------------------------------------------

    token_losses = token_losses.view(targets.shape)

    # =========================================================
    # Part 4
    # Apply Loss Mask
    # =========================================================
    #
    # token_losses:
    #
    # [loss, loss, loss, loss]
    #
    # loss_mask:
    #
    # [1, 1, 1, 0]
    #
    # 相乘：
    #
    # [loss, loss, loss, 0]
    #
    # PAD 对应的 loss 被清零。
    # =========================================================

    masked_token_losses = token_losses * loss_mask

    # =========================================================
    # Part 5
    # Average Only Valid Tokens
    # =========================================================
    #
    # 不写 masked_token_losses.mean()，
    #
    # 因为这样 PAD 的 0 仍然会进入分母。
    #
    # 正确做法：
    #
    # 有效 token 的 loss 总和
    # -----------------------
    # 有效 token 的数量
    # =========================================================

    loss = (masked_token_losses.sum() / loss_mask.sum())
    return loss

if __name__ == "__main__":
    text = "I love Yuri Nakamura and Computer Science!"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # input_ids:
    # [B, L]

    input_ids = tokenizer.encode(
        text, add_special_tokens=False, return_tensors="pt"
    ).to(device)

    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        num_layers=4,
        d_model=64,
        num_heads=4,
        num_kv_heads=2,
        d_ff=256,
    ).to(device)

    # Logits
    # [B, L, V] V = vocab_size

    logits = model(input_ids)

    loss = next_token_loss(logits, input_ids)

    print("Text:")
    print(text)

    print("\nInput IDs shape:")
    print(input_ids.shape)

    print("\nLogits shape:")
    print(logits.shape)

    print("\nloss:")
    print(loss.item())

    # =========================================================
    # Test Loss Mask
    # =========================================================

    print("\n" + "=" * 60)
    print("Test Loss Mask")
    print("=" * 60)

    texts = [
        "I love deep learning",
        "Artificial intelligence is amazing"
    ]

    tokenizer.pad_token = tokenizer.eos_token

    batch = tokenizer(
        texts,
        padding = True,
        add_special_tokens = False,
        return_tensors = "pt"
    )

    batch_input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    print("\nInput IDs:")
    print(batch_input_ids)

    print("\nInput IDs shape:")
    print(batch_input_ids.shape)

    print("\nAttention Mask:")
    print(attention_mask)

    print("\nAttention Mask shape:")
    print(attention_mask.shape)

    # ---------------------------------------------------------
    # MiniLLM 得到 logits
    #
    # [B, L]
    #
    #   ↓
    #
    # [B, L, V]
    # ---------------------------------------------------------

    batch_logits = model(batch_input_ids)
    print("\nLogits shape:")
    print(batch_logits.shape)

    # =========================================================
    # 构造 Loss Mask
    # =========================================================

    loss_mask = attention_mask[:, 1:]

    print("\nLoss Mask:")
    print(loss_mask)

    print("\nLoss Mask shape:")
    print(loss_mask.shape)

    # ---------------------------------------------------------
    # 使用新的 Loss Mask 版本
    # ---------------------------------------------------------
    masked_loss = next_token_loss_with_mask(
        batch_logits,
        batch_input_ids,
        attention_mask
    )

    print("\nLoss with Loss Mask:")
    print(masked_loss.item())


