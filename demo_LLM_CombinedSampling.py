import torch 
import torch.nn.functional as F
from transformers import AutoTokenizer

from demo_MiniLLM import MiniLLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    torch.manual_seed(42)

    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only = True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

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

    prompt = "I love"
    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens = True,
        return_tensors = "pt"
    ).to(device)

    max_tokens = 10

    temperature = 0.8
    top_k = 20
    top_p = 0.9

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            
            # --------------------------------------------------
            # Step 1: Temperature
            # --------------------------------------------------
            #
            # 调整概率分布的尖锐程度。
            #
            # T < 1:
            #   高概率 token 更突出，生成更保守
            #
            # T > 1:
            #   概率分布更平坦，生成更多样
            #
            # shape 不变：
            #
            # [B, V] -> [B, V]
            scaled_logits = next_token_logits / temperature

            # --------------------------------------------------
            # Step 2: Top-k
            # --------------------------------------------------
            #
            # 从整个词表中，
            # 只保留 logits 最大的 k 个 token。
            #
            # [B, V]
            #    ↓
            # [B, k]
            #
            # top_k_indices 保存这些候选
            # 在原始词表中的 token id。
            top_k_logits, top_k_indices = torch.topk(
                scaled_logits,
                k = top_k,
                dim = -1
            )

            # --------------------------------------------------
            # Step 3: Softmax
            # --------------------------------------------------
            #
            # 将 Top-k logits 转换成概率。
            #
            # torch.topk 默认按从大到小返回，
            # 所以 top_k_probs 也按概率从大到小排列。
            #
            # [B, k] -> [B, k]
            top_k_probs = F.softmax(top_k_logits, dim = -1)
            
            # --------------------------------------------------
            # Step 4: Top-p
            # --------------------------------------------------
            #
            # 在 Top-k 候选内部计算累计概率。
            #
            # 例如：
            #
            # top_k_probs:
            #
            # [0.40, 0.30, 0.15, 0.10, 0.05]
            #
            # cumulative_probs:
            #
            # [0.40, 0.70, 0.85, 0.95, 1.00]
            cumulative_probs = torch.cumsum(
                top_k_probs,
                dim = -1
            )

            # 找出累计概率超过 top_p 的位置。
            #
            # top_p = 0.8:
            #
            # [0.40, 0.70, 0.85, 0.95, 1.00]
            #
            #               ↓
            #
            # [False, False, True, True, True]

            remove_mask = cumulative_probs > top_p
            remove_mask[:, 1:] = remove_mask[:, :-1].clone()

            remove_mask[:, 0] = False
            
            filtered_probs = top_k_probs.masked_fill(
                remove_mask,
                0.0
            )

            filtered_probs = filtered_probs / filtered_probs.sum(
                dim = -1,
                keepdim = True
            )

            # --------------------------------------------------
            # Step 5: Multinomial Sampling
            # --------------------------------------------------
            #
            # 从最终候选概率分布中随机采样。
            #
            # filtered_probs:
            # [B, k]
            #
            # sampled_index:
            # [B, 1]
            #
            # sampled_index 只是 Top-k 列表中的位置，
            # 还不是原词表中的 token id。

            sampled_index = torch.multinomial(
                filtered_probs,
                num_samples = 1
            )

            # 根据 sampled_index，
            # 从 top_k_indices 中取回
            # 真正的词表 token id。
            #
            # top_k_indices:
            # [B, k]
            #
            # sampled_index:
            # [B, 1]
            #
            # ↓
            #
            # next_token:
            # [B, 1]

            next_token = torch.gather(
                top_k_indices,
                dim = -1,
                index = sampled_index
            )

            input_ids = torch.cat((input_ids, next_token), dim = -1)

        generated_text = tokenizer.decode(input_ids[0])
        
        print("Prompt:")
        print(prompt)

        print("\nTempature:")
        print(temperature)

        print("\nTop-k:")
        print(top_k)

        print("\nTop-p:")
        print(top_p)

        print("\nGenerated:")
        print(generated_text)
