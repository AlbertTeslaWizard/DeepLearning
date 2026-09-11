import torch
import torch.nn as nn
from transformer.components.demo_GQA_MQA_AttentionMask import GroupedQueryAttention
from transformer.components.demo_SwiGLU_FFN import SwiGLUFFN
from transformer.components.demo_RMSNorm import RMSNorm
from transformer.components.demo_RoPE import RoPE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerBlock(nn.Module):
    def __init__(self, d_model = 64, num_heads = 4, num_kv_heads = 2, d_ff = 256):
        super().__init__()
        self.head_dim = d_model // num_heads
        self.attention = GroupedQueryAttention(d_model = d_model, num_heads = num_heads, num_kv_heads = num_kv_heads, is_causal = True)

        self.ffn = SwiGLUFFN(d_model = d_model, d_ff = d_ff)

        self.norm1 = RMSNorm(dim = d_model)
        self.norm2 = RMSNorm(dim = d_model)
        
        self.rope = RoPE(head_dim = self.head_dim)

    def forward(self, x, attention_mask=None):
        seq_len = x.shape[1]
        cos, sin = self.rope(seq_len, device = x.device)

        x = x + self.attention(self.norm1(x), cos = cos, sin = sin, attention_mask = attention_mask)
        x = x + self.ffn(self.norm2(x)) 

        return x

if __name__ == '__main__':
    torch.manual_seed(42)
    
    x = torch.randn(2, 5, 64, device=device)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1]
        ],
        device = device
    )

    model = TransformerBlock(
        d_model = 64,
        num_heads = 4,
        num_kv_heads = 2,
        d_ff = 256
    ).to(device)

    output = model(x, attention_mask = attention_mask)
    params = sum(p.numel() for p in model.parameters())
    
    print(f"Device: {device}")
    print(f"Params: {params:,}")

    print("\nInput shape:")
    print(x.shape)

    print("\nAttention Mask:")
    print(attention_mask)

    print("\nAttention Mask shape:")
    print(attention_mask.shape)

    print("\nOutput shape:")
    print(output.shape)
