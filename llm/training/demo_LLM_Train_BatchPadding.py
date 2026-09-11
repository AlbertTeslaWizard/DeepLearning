import torch
import torch.optim as optim 

from transformers import AutoTokenizer

from llm.modeling.demo_MiniLLM_AttentionMask import MiniLLM
from llm.training.demo_NextTokenLoss import next_token_loss_with_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    torch.manual_seed(42)
    texts = [
        "I love deep learning.\n",
        "I love Yuri Nakamura.\n",
        "Deep learning is magical."
    ]

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    batch = tokenizer(
        texts,
        padding = True,
        add_special_tokens = False,
        return_tensors = "pt"
    )

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    print("Texts:")
    
    for i, text in enumerate(texts):
        print(f"{i}: {text}")

    print("\nInput IDs:")
    print(input_ids)

    print("\nInput IDs shape:")
    print(input_ids.shape)

    print("\nAttention Mask:")
    print(attention_mask)

    print("\nAttention Mask shape:")
    print(attention_mask.shape)

    # ---------------------------------------------------------
    # Loss Mask 与 targets 对齐。
    #
    # input_ids:
    #
    # [A, B, C, D, PAD]
    #
    # targets:
    #
    # [B, C, D, PAD]
    #
    # 所以：
    #
    # attention_mask:
    #
    # [1, 1, 1, 1, 0]
    #
    # loss_mask:
    #
    # [1, 1, 1, 0]
    # ---------------------------------------------------------
    
    loss_mask = attention_mask[:, 1:]
    print("\nLoss Mask:")
    print(loss_mask)

    print("\nLoss Mask shape:")
    print(loss_mask.shape)

    # =========================================================
    # Part 4
    # MiniLLM
    # =========================================================

    # ---------------------------------------------------------
    # 使用支持 attention_mask 的 MiniLLM。
    #
    # attention_mask 会一路经过：
    #
    # MiniLLM
    #    ↓
    # LLMBackbone
    #    ↓
    # TransformerStack
    #    ↓
    # TransformerBlock
    #    ↓
    # GroupedQueryAttention
    #
    # 最终用于 Padding Mask。
    # ---------------------------------------------------------

    model = MiniLLM(
        vocab_size = tokenizer.vocab_size,
        num_layers = 4,
        d_model = 64,
        num_heads = 4,
        num_kv_heads = 2,
        d_ff = 256
    ).to(device)

    model.train()
    
    # =========================================================
    # Part 5
    # Optimizer
    # =========================================================

    optimizer = optim.AdamW(model.parameters(), lr = 1e-3)
    epochs = 500

    # =========================================================
    # Part 6
    # Training Loop
    # =========================================================

    for epoch in range(epochs):
        # =====================================================
        # Forward
        # =====================================================

        # -----------------------------------------------------
        # input_ids:
        #
        # [B, L]
        #
        # attention_mask:
        #
        # [B, L]
        #
        # ↓
        #
        # logits:
        #
        # [B, L, vocab_size]
        #
        #
        # 模型内部的 attention_mask 负责：
        #
        # PAD 不能作为有效 Key 被 Attention 读取。
        # -----------------------------------------------------

        logits = model(input_ids, attention_mask = attention_mask)

        # =====================================================
        # Loss
        # =====================================================

        # -----------------------------------------------------
        # next_token_loss_with_mask 负责两件事：
        #
        # 1.
        # Next-token shift
        #
        # logits:
        # [B, L, V]
        #
        #   ↓
        #
        # shift_logits:
        # [B, L-1, V]
        #
        #
        # input_ids:
        # [B, L]
        #
        #   ↓
        #
        # targets:
        # [B, L-1]
        #
        #
        # 2.
        # Loss Mask
        #
        # attention_mask[:, 1:]
        #
        # 让 target = PAD 的位置
        # 不参与 CrossEntropy Loss。
        # -----------------------------------------------------
        
        loss = next_token_loss_with_mask(logits, input_ids, attention_mask = attention_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Loss: {loss.item():.4f}"
            )
    
    torch.save(model.state_dict(), "mini_llm.pt")
    print("\nModel saved to mini_llm.pt")
