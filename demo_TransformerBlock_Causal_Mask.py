import torch
import torch.nn as nn
from demo_GQA_MQA import GroupedQueryAttention
from demo_SwiGLU_FFN import SwiGLUFFN
from demo_RMSNorm import RMSNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerBlock(nn.Module):
    def __init__(self, d_model = 64, num_heads = 4, num_kv_heads = 2, d_ff = 256):
        super().__init__()
        self.attention = GroupedQueryAttention(d_model = d_model, num_heads = num_heads, num_kv_heads = num_kv_heads, is_causal = True)

        self.ffn = SwiGLUFFN(d_model = d_model, d_ff = d_ff)

        self.norm1 = RMSNorm(dim = d_model)
        self.norm2 = RMSNorm(dim = d_model)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x)) 

        return x

if __name__ == '__main__':
    torch.manual_seed(42)

    x = torch.randn(32, 10, 64, device = device)
    model = TransformerBlock(
        d_model = 64,
        num_heads = 4,
        num_kv_heads = 2,
        d_ff = 256
    ).to(device)

    output = model(x)
    params = sum(p.numel() for p in model.parameters())


    print(f"Device: {device}")
    print(f"Params: {params:,}")
    print(f"Shapes : In {tuple(x.shape)} -> Out {tuple(output.shape)}")

