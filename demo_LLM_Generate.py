import torch
from transformers import AutoTokenizer

from demo_MiniLLM import MiniLLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    torch.manual_seed(42)
    
       
    try:
       tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except OSError:
       tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        num_layers=4,
        d_model=64,
        num_heads=4,
        num_kv_heads=2,
        d_ff=256,
    ).to(device)

    model.load_state_dict(
        torch.load(
            "mini_llm.pt",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()
    prompt = "I love"

    input_ids = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(device)

    max_new_tokens = 10

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)

            # 只看目前序列的最后一个位置：
            #
            # [B, L, V]
            #   ↓
            # [B, V]
            #
            # 表示：
            # 根据目前所有 token，
            # 预测“下一个 token”

            next_token_logits = logits[:, -1, :]

            # Greedy Decoding:
            # 直接选择 logits 最大的 token
            #
            # [B, V] -> [B]
            next_token = torch.argmax(next_token_logits, dim=-1)

            # [B] -> [B, 1]
            #
            # 因为 input_ids 是 [B, L]，
            next_token = next_token.unsqueeze(-1)

            # 把新 token 拼回 input_ids：
            #
            # [B, L] + [B, 1]
            #   ↓
            # [B, L+1]

            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    

    generate_text = tokenizer.decode(input_ids[0])

    print("Prompt:")
    print(prompt)

    print("\nGenerated:")
    print(generate_text)
