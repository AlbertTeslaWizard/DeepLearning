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

    # Temperature 控制概率分布的“尖锐程度”：
    #
    # T = 1.0：保持原来的概率分布
    #
    # T < 1.0：logits 差距被放大
    #          -> 概率分布更尖锐
    #          -> 更倾向选择高概率 token
    #          -> 生成更保守
    #
    # T > 1.0：logits 差距被缩小
    #          -> 概率分布更平坦
    #          -> 低概率 token 更容易被采样
    #          -> 生成更多样

    temperature = 0.8
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # input_ids: [B, L]
            # logits: [B, L, V]
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]

            scaled_logits = next_token_logits / temperature
            probs = F.softmax(scaled_logits, dim = -1)

            # Temperature Sampling:
            #
            #     multinomial(probs)
            #
            # 按照概率分布随机抽取一个 token
            
            # next_token: [B, 1]
            next_token = torch.multinomial(
                probs,
                num_samples = 1
            )
            
            input_ids = torch.cat((input_ids, next_token), dim = 1)

    generated_text = tokenizer.decode(input_ids[0])
    
    print("Prompt:")
    print(prompt)

    print("\nTemperature:")
    print(temperature)

    print("\nGenerated:")
    print(generated_text)
