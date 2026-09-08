import torch 
import torch.nn.functional as F
from transformers import AutoTokenizer 

from llm.modeling.demo_MiniLLM import MiniLLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    torch.manual_seed(42)

    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only = True)
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = MiniLLM(
        vocab_size = tokenizer.vocab_size,
        num_layers = 4,
        d_model = 64,
        num_heads = 4,
        num_kv_heads = 2,
        d_ff = 256
    ).to(device)

    state_dict = torch.load(
        "mini_llm.pt",
        map_location = device,
        weights_only = True
    )
    
    model.load_state_dict(state_dict)

    model.eval()

    prompt = "I love"
    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens = False,
        return_tensors = "pt"
    ).to(device)

    max_new_tokens = 10
    top_k = 5

    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]

        # Top-k:
        #
        # 从整个词表 V 中，
        # 只取 logits 最大的 k 个 token
        #
        # [B, V]
        #    ↓
        # [B, k]
        #
        # top_k_logits:
        # 前 k 大的 logits
        #
        # top_k_indices:
        # 这些 token 在原词表中的 token id

        top_k_logits, top_k_indices = torch.topk(
            next_token_logits,
            k = top_k,
            dim = -1            
        )
        
        top_k_probs = F.softmax(
            top_k_logits,
            dim = -1
        )

        sampled_index = torch.multinomial(
            top_k_probs,
            num_samples = 1
        )
        
        # 根据 sampled_index，
        # 从 top_k_indices 中取出真正的 token id
        #
        # 注意：
        # sampled_index 不是原词表中的 token id，
        # 它只是表示：
        #
        #     “选中了 top-k 候选中的第几个”
        #
        # 例如：
        #
        # top_k_indices =
        # [[1543, 318, 257, 1842, 4673]]
        #
        # sampled_index =
        # [[2]]
        #
        # 表示选中了 top-k 列表中的第 2 个位置，
        # 所以真正的 token id 是：
        #
        # top_k_indices[0][2] = 257
        #
        # top_k_indices: [B, k]
        # sampled_index: [B, 1]
        #
        # sampled_index 为每个 batch 指定
        # 要从 top_k_indices 的哪个位置取值
        #
        # torch.gather(...)
        #        ↓
        # next_token: [B, 1]
        #
        # 输出 shape 与 sampled_index 相同
        
        next_token = torch.gather(
            top_k_indices,
            dim = -1,
            index = sampled_index
        )
        
        input_ids = torch.cat((input_ids, next_token), dim = 1)

    generate_text = tokenizer.decode(input_ids[0])
    
    print("Prompt:")
    print(prompt)

    print("\nTop-k:")
    print(top_k)

    print("\nGenerated:")
    print(generate_text)
