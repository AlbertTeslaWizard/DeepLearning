import torch 
import torch.nn as nn

from transformer.blocks.demo_TransformerStack_KVCache import TransformerStack
from transformers import AutoTokenizer

class LLMBackbone(nn.Module):
    def __init__(self, vocab_size, num_layers = 4, d_model = 64, num_heads = 4, num_kv_heads = 2, d_ff = 256):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = d_model)
        self.transformer = TransformerStack(
            num_layers = num_layers,
            d_model = d_model,
            num_heads = num_heads,
            num_kv_heads = num_kv_heads,
            d_ff = d_ff 
        )

    def forward(self, input_ids, past_kv = None, use_cache = False):
        if past_kv is not None and not use_cache:
            raise ValueError("past_kv requires use_cache=True")

        # ---------------------------------------------------------
        # Token IDs
        #
        # [B, L]
        #
        #   ↓ Embedding
        #
        # [B, L, D]
        # ---------------------------------------------------------
        x = self.embedding(input_ids)

        # ---------------------------------------------------------
        # 不使用 KV Cache
        
        if not use_cache:
            x = self.transformer(x)
            return x
        
        # ---------------------------------------------------------
        # 使用 KV Cache
        #
        # past_kv：
        #
        # [
        #   (K_0, V_0),
        #   (K_1, V_1),
        #   ...
        # ]
        #
        # TransformerStack 会负责：
        #
        #   1. 把每层 Cache 分给对应 Layer
        #   2. 更新每层 Cache
        #   3. 返回新的 present_kv
        # ---------------------------------------------------------

        x, present_kv = self.transformer(
            x,
            past_kv = past_kv,
            use_cache = True 
        )

        return x, present_kv 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = "I love Yuri Nakamura and Artificial Intelligence!"

    input_ids = tokenizer.encode(
        text,
        add_special_tokens = False,
        return_tensors = "pt"
    ).to(device)

    model = LLMBackbone(
        vocab_size = tokenizer.vocab_size,
        num_layers = 4,
        d_model = 64,
        num_heads = 4,
        num_kv_heads = 2,
        d_ff = 256
    ).to(device)

    output = model(input_ids)

    print("Input IDs shape:")
    print(input_ids.shape)
    
    print("\nOutput shape:")
    print(output.shape)




    


