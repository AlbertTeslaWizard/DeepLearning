import torch 
import torch.nn.functional as F 
from transformers import AutoTokenizer 

from demo_MiniLLM import MiniLLM

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

    state_dict = torch.load("mini_llm.pt", map_location = device, weights_only = True)
    model.load_state_dict(state_dict)

    model.eval()
    prompt = "I love"
    
    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens = False,
        return_tensors = "pt"
    ).to(device)

    max_new_tokens = 10
    top_p = 0.9

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]

            probs = F.softmax(next_token_logits, dim = -1)
            # Top-p Sampling:
            #
            # 先按照概率从大到小排序
            #
            # probs:          [B, V]
            # sorted_probs:   [B, V]
            # sorted_indices: [B, V]
            #
            # sorted_indices 保存这些概率
            # 在原词表中的 token id
            
            sorted_probs, sorted_indices = torch.sort(
                probs,
                descending = True,
                dim = -1
            )

            # 计算累计概率：
            #
            # 例如：
            #
            # sorted_probs:
            # [0.40, 0.30, 0.15, 0.10, 0.05]
            #
            # cumulative_probs:
            # [0.40, 0.70, 0.85, 0.95, 1.00]

            cumulative_probs = torch.cumsum(
                sorted_probs,
                dim = -1
            )
            
            # 找出累计概率超过 top_p 的位置
            #
            # top_p = 0.8:
            #
            # cumulative_probs:
            # [0.40, 0.70, 0.85, 0.95, 1.00]
            #
            # remove_mask:
            # [False, False, True, True, True]

            remove_mask = cumulative_probs > top_p

            # 不能直接删除第一个超过 top_p 的 token，
            # 因为它正是使累计概率达到 top_p 的 token。
            #
            # 所以将 mask 向右移动一位：
            #
            # [False, False, True,  True, True]
            #                 ↓
            # [False, False, False, True, True]

            remove_mask[:, 1:] = remove_mask[:, :-1].clone()
            remove_mask[:, 0] = False

            filtered_probs = sorted_probs.masked_fill(
                remove_mask,
                0.0
            )

            filtered_probs = filtered_probs / filtered_probs.sum(
                dim = -1,
                keepdim = True
            )

            # 在 Top-p 候选集合中进行随机采样
            #
            # sampled_index:
            # [B, 1]
            #
            # 注意：
            # sampled_index 只是 sorted_indices 中的位置，
            # 还不是原词表中的 token id

            sampled_index = torch.multinomial(
                filtered_probs,
                num_samples = 1
            )
            
            # 从 sorted_indices 中取回真正的 token id
            #
            # sorted_indices: [B, V]
            # sampled_index:  [B, 1]
            #                   ↓
            # next_token:     [B, 1]

            next_token = torch.gather(
                sorted_indices,
                dim = -1,
                index = sampled_index
            )

            input_ids = torch.cat((input_ids, next_token), dim = -1)
        
        generated_text = tokenizer.decode(input_ids[0])
        
        print("Prompt:")
        print(prompt)
        
        print("\nTop-p:")
        print(top_p)

        print("\nGenerated:")
        print(generated_text)
