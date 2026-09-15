import torch
from transformers import AutoTokenizer

from llm.modeling.demo_MiniLLM_KVCache import MiniLLM

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
        # =========================================================
        # Step 1
        # Prefill
        # =========================================================
        #
        # 第一次把整个 Prompt 输入模型。
        #
        # input_ids:
        #
        # [B, L_prompt]
        #
        # 模型同时返回：
        #
        # logits
        # past_kv
        #
        # past_kv 保存每一层 Transformer 的 K / V Cache。
        # =========================================================

        logits, past_kv = model(input_ids, use_cache = True)

        # =========================================================
        # Step 2
        # Autoregressive Decode
        # =========================================================
        
        for step in range(max_new_tokens):
            
            # -----------------------------------------------------
            # logits:
            #
            # Prefill：
            # [B, L_prompt, V]
            #
            # Decode：
            # [B, 1, V]
            #
            # 两种情况下都只需要最后一个位置：
            #
            # [B, V]
            # -----------------------------------------------------

            next_token_logits = logits[:, -1, :]

            # -----------------------------------------------------
            # Greedy Decoding
            #
            # [B, V]
            #   ↓
            # [B, 1]
            # -----------------------------------------------------
            
            next_token = torch.argmax(next_token_logits, dim = -1, keepdim = True)
            
            # -----------------------------------------------------
            # 保存生成出来的 Token。
            #
            # input_ids:
            #
            # [B, L]
            #
            # next_token:
            #
            # [B, 1]
            #
            #      ↓
            #
            # [B, L + 1]
            # -----------------------------------------------------

            input_ids = torch.cat(
                [input_ids, next_token], 
                dim = 1
            )
            
            # 最后一个 Token 已经生成，
            # 不需要再计算下一轮 logits。
            if step == max_new_tokens - 1:
                break

            # =====================================================
            # Decode
            # =====================================================
            #
            # 不再把整个 input_ids 输入模型。
            #
            # 只输入最新生成的：
            #
            # next_token
            #
            # shape:
            #
            # [B, 1]
            #
            # 历史信息已经保存在 past_kv 中。
            # =====================================================

            logits, past_kv = model(
                next_token,
                past_kv = past_kv,
                use_cache = True        
            )
            

    generate_text = tokenizer.decode(input_ids[0])

    print("Prompt:")
    print(prompt)

    print("\nGenerated:")
    print(generate_text)
