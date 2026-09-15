import torch
import torch.nn as nn
from transformer.components.demo_GQA_MQA_KVCache import GroupedQueryAttention
from transformer.components.demo_SwiGLU_FFN import SwiGLUFFN
from transformer.components.demo_RMSNorm import RMSNorm
from transformer.components.demo_RoPE_KVCache import RoPE

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

    def forward(self, x, past_kv = None, use_cache = False):
        if past_kv is not None and not use_cache:
            raise ValueError("past_kv requires use_cache=True")

        seq_len = x.shape[1]

        # =========================================================
        # RoPE Position
        # =========================================================

        # Prefill：
        #
        # past_kv = None
        # past_len = 0
        #
        # Decode：
        #
        # past_k:
        # [B, H_kv, L_past, D_head]
        #
        # 新 Token 的位置应该从 L_past 开始。
        
        past_len = 0

        if past_kv is not None:
            past_k, _ = past_kv
            past_len = past_k.shape[2]
        
        cos, sin = self.rope(seq_len, device = x.device, start_pos = past_len)
        if use_cache:
            attn_output, present_kv = self.attention(
                self.norm1(x),
                cos = cos,
                sin = sin,
                past_kv = past_kv,
                use_cache = True
            )
        else:
            attn_output = self.attention(
                self.norm1(x),
                cos = cos,
                sin = sin
            )

        x = x + attn_output
        x = x + self.ffn(self.norm2(x))

        if use_cache:
            return x, present_kv

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

