import torch 
from torch.cuda import device_count
from transformers import AutoTokenizer

from llm.modeling.demo_MiniLLM_AttentionMask import MiniLLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    torch.manual_seed(42)

    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only = True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    prompts = [
        "I love",
        "I love Yuri Nakamura",
        "Deep learning is"
    ]

    batch = tokenizer(
        prompts,
        padding = True,
        add_special_tokens = False,
        return_tensors = "pt"
    )

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    print("Prompts:")

    for i, prompt in enumerate(prompts):
        print(f"{i}: {prompt}")

    print("\nInput IDs:")
    print(input_ids)

    print("\nAttention Mask:")
    print(attention_mask)

    model = MiniLLM(
        vocab_size = tokenizer.vocab_size,
        num_layers = 4,
        d_model = 64,
        num_heads = 4,
        num_kv_heads = 2,
        d_ff = 256
    ).to(device)

    state_dict = torch.load("mini_llm.pt", map_location = device, weights_only = True)

    model.load_state_dict(state_dict)
    model.eval()

    max_new_tokens = 10

    # =========================================================
    # Batch Autoregressive Generation
    # =========================================================
    with torch.no_grad():
        for step in range(max_new_tokens):
            # -------------------------------------------------
            # Forward
            #
            # input_ids:
            #
            # [B, L]
            #
            # attention_mask:
            #
            # [B, L]
            #
            #       ↓
            #
            # logits:
            #
            # [B, L, V]
            # -------------------------------------------------
            
            logits = model(input_ids, attention_mask = attention_mask)
            
            # =================================================
            # Step 1
            # 找到每条样本最后一个有效 Token
            # =================================================

            # -------------------------------------------------
            # attention_mask:
            #
            # [1, 1, 0, 0]
            # [1, 1, 1, 1]
            #
            # sum(dim=1):
            #
            # [2, 4]
            #
            # 因为 Tensor index 从 0 开始：
            #
            # last_valid_positions:
            #
            # [1, 3]
            # -------------------------------------------------

            sequence_lengths = attention_mask.sum(dim = 1)
            last_valid_positions = sequence_lengths - 1

            batch_indices = torch.arange(
                input_ids.size(0),
                device = device
            )

            # -------------------------------------------------
            # 对每条样本分别取：
            #0
            # logits[
            #     batch_index,
            #     last_valid_position,
            #     :
            # ]
            #
            # [B, L, V]
            #
            #      ↓
            #
            # [B, V]
            # -------------------------------------------------

            next_token_logits = logits[batch_indices, last_valid_positions, :]

            # =================================================
            # Step 2
            # Greedy Decoding
            # =================================================

            # [B, V]
            #
            #   ↓
            #
            # [B]
            
            next_token = torch.argmax(next_token_logits, dim = -1)

            # =================================================
            # Step 3
            # 给 Batch 再增加一列 Padding
            # =================================================

            # -------------------------------------------------
            # 为什么不是直接：
            #
            # input_ids = torch.cat(
            #     [input_ids, next_token],
            #     dim=1
            # )
            #
            # 因为 Right Padding 下，
            # 短样本可能会变成：
            #
            # [A, B, PAD, PAD, X]
            #
            # 这样真实 Token 就不连续了。
            #
            # 我们希望始终保持：
            #
            # [A, B, X, PAD, PAD]
            #
            # 所以：
            #
            # 先在所有样本最右边增加一个 PAD，
            # 再把新 Token 写到每条样本自己的
            # 第一个 PAD 位置。
            # -------------------------------------------------

            # ---------------------------------------------------------
            # 先给整个 Batch 在最右边增加一列 PAD。
            #
            # 例如原来：
            #
            # input_ids:
            # [A, B, PAD, PAD]
            # [C, D, E,   F]
            #
            # attention_mask:
            # [1, 1, 0, 0]
            # [1, 1, 1, 1]
            #
            # 增加一列后：
            #
            # input_ids:
            # [A, B, PAD, PAD, PAD]
            # [C, D, E, F, PAD]
            #
            # attention_mask:
            # [1, 1, 0, 0, 0]
            # [1, 1, 1, 1, 0]
            # ---------------------------------------------------------

            pad_column = torch.full(
                (input_ids.size(0), 1),
                tokenizer.pad_token_id,
                dtype = input_ids.dtype,
                device = device
            )

            input_ids = torch.cat(
                [input_ids, pad_column],
                dim = 1
            )

            mask_column = torch.zeros(
                (attention_mask.size(0), 1),
                dtype = attention_mask.dtype,
                device = device
            )
            
            attention_mask = torch.cat(
                [attention_mask, mask_column],
                dim = 1
            )
            
            # =========================================================
            # Step 4
            # 把新 Token 写到每条样本的第一个 PAD 位置
            # =========================================================

            # ---------------------------------------------------------
            # 假设：
            #
            # sequence_lengths = [2, 4]
            # next_token       = [X, Y]
            #
            # 那么：
            #
            # Sample 0 的新 Token 写到 index 2
            # Sample 1 的新 Token 写到 index 4
            #
            # [A, B, PAD, PAD, PAD] -> [A, B, X, PAD, PAD]
            # [C, D, E, F, PAD]   -> [C, D, E, F, Y]
            #
            # 因为真实 Token 从 index 0 开始连续排列，
            # 所以 sequence_length 正好就是第一个 PAD 的位置。
            # ---------------------------------------------------------

            write_positions = sequence_lengths
            input_ids[batch_indices, write_positions] = next_token
            
            # 新写入的位置已经是真实 Token，
            # 所以对应的 attention_mask 从 0 改成 1。
            #
            # [1, 1, 0, 0, 0] -> [1, 1, 1, 0, 0]
            # [1, 1, 1, 1, 0] -> [1, 1, 1, 1, 1]
            
            attention_mask[batch_indices, write_positions] = 1

            print(f"\nStep {step + 1}")
            print("Sequence lengths:")
            print(sequence_lengths)

            print("Last valid positions:")
            print(last_valid_positions)

            print("Next token:")
            print(next_token)

            print("Input IDs:")
            print(input_ids)

            print("Attention Mask:")
            print(attention_mask)

    # =========================================================
    # Decode
    # =========================================================
    
    print("\nGenerated:")

    for i in range(input_ids.size(0)):
        # -----------------------------------------------------
        # 只取 attention_mask == 1 的真实 Token。
        #
        # 不把尾部 Padding 一起拿去 decode。
        # -----------------------------------------------------

        valid_input_ids = input_ids[i][attention_mask[i].bool()]
        generated_text = tokenizer.decode(valid_input_ids)

        print(f"\n{i}:")
        print(generated_text)
