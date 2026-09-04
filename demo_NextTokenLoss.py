import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from demo_MiniLLM import MiniLLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    text = "I love Yuri Nakamura and Computer Science!"
    # 本机访问不了 huggingface.co，但 gpt2 已缓存在本地，
    # 用 local_files_only=True 强制只用本地缓存，避免联网卡死
    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)

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

    # Logits 去掉最后一个位置:
    # [B, L, V] -> [B, L - 1, V]
    # [B, L] -> [B, L - 1]

    shift_logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]

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
        shift_logits.reshape(-1, tokenizer.vocab_size), targets.reshape(-1)
    )

    print("Text:")
    print(text)

    print("\nInput IDs shape:")
    print(input_ids.shape)

    print("\nLogits shape:")
    print(logits.shape)

    print("\nShift logits shape:")
    print(shift_logits.shape)

    print("\nTargets shape:")
    print(targets.shape)

    print("\nFlattened logits shape:")
    print(shift_logits.reshape(-1, tokenizer.vocab_size).shape)

    print("\nLoss:")
    print(loss.item())
