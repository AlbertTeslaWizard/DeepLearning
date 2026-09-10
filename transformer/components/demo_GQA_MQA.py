import torch 
import torch.nn as nn 
import math
from transformer.components.demo_RoPE import apply_rotary_pos_emb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=4, num_kv_heads=2, is_causal=True):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads 需要能被 num_kv_heads 整除"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_head = d_model // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.is_causal = is_causal 

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.num_kv_heads * self.d_head)
        self.W_v = nn.Linear(d_model, self.num_kv_heads * self.d_head)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, cos=None, sin=None, attention_mask=None, debug=False):
        B, L, _ = x.shape
        
        Q = self.W_q(x).view(B, L, self.num_heads, self.d_head)
        K = self.W_k(x).view(B, L, self.num_kv_heads, self.d_head)
        V = self.W_v(x).view(B, L, self.num_kv_heads, self.d_head)
        
        if (cos is None) != (sin is None):
            raise ValueError(
                "cos and sin must either both be provided or both be None"
            )

        if cos is not None:
            Q = apply_rotary_pos_emb(Q, cos, sin)
            K = apply_rotary_pos_emb(K, cos, sin)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        if self.num_queries_per_kv > 1:
            K = K.repeat_interleave(self.num_queries_per_kv, dim = 1)
            V = V.repeat_interleave(self.num_queries_per_kv, dim = 1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        padding_scores = scores

        if attention_mask is not None:
            # ---------------------------------------------------------
            # attention_mask:
            #
            # [B, L]
            #
            #   ↓
            #
            # padding_mask:
            #
            # [B, 1, 1, L]
            #
            # 最后一个 L 表示 Key positions。
            #
            # 之后会 broadcasting 到：
            #
            # [B, H, L, L]
            # ---------------------------------------------------------
            
            padding_mask = attention_mask[:, None, None, :]
            padding_scores = scores.masked_fill(
                padding_mask == 0,
                float('-inf')
            )
        
        causal_scores = padding_scores

        if self.is_causal:
            causal_mask = torch.triu(torch.ones(L, L, dtype = torch.bool, device = x.device), diagonal = 1)
            causal_mask = causal_mask[None, None, :, :]
            
            causal_scores = padding_scores.masked_fill(causal_mask, float('-inf'))

        attention = torch.softmax(causal_scores, dim = -1)

        if debug:
            print("\n" + "=" * 60)
            print("Debug Attention")
            print("=" * 60)

            print("\nOriginal Attention Scores:")
            print(scores[0, 0])

            if attention_mask is not None:
                print("\nAfter Padding Mask:")
                print(padding_scores[0, 0])

            if self.is_causal:
                print("\nAfter Causal Mask:")
                print(causal_scores[0, 0])

            print("\nFinal Attention Weights:")
            print(attention[0, 0])

        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.W_o(out)
        return out

if __name__ == '__main__':
    torch.manual_seed(42)

    
    x = torch.randn(32, 10, 64).to(device)
    
    MHA = GroupedQueryAttention(d_model = 64, num_heads = 4, num_kv_heads = 4).to(device)
    MQA = GroupedQueryAttention(d_model = 64, num_heads = 4, num_kv_heads = 1).to(device)
    GQA = GroupedQueryAttention(d_model = 64, num_heads = 4, num_kv_heads = 2).to(device)

    print(f"MHA 参数量: {sum(p.numel() for p in MHA.parameters()):,}")
    print(f"MQA 参数量: {sum(p.numel() for p in MQA.parameters()):,}")
    print(f"GQA 参数量: {sum(p.numel() for p in GQA.parameters()):,}")

    print(f"MHA 输出形状: {MHA(x).shape}")
    print(f"MQA 输出形状: {MQA(x).shape}")    
    print(f"GQA 输出形状: {GQA(x).shape}")

    # =========================================================
    # Test Padding Mask
    # =========================================================

    print("\n" + "=" * 60)
    print("Test Padding Mask")
    print("=" * 60)

    x_mask = torch.randn(2, 5, 64).to(device)

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1]
        ],
        device = device
    )

    print("\nInput shape:")
    print(x_mask.shape)

    print("\nAttention Mask:")
    print(attention_mask)

    print("\nAttention Mask shape:")
    print(attention_mask.shape)

    out_without_mask = GQA(x_mask)
    print("\nOutput without Padding Mask:")
    print(out_without_mask.shape)

    out_with_mask = GQA(x_mask, attention_mask = attention_mask)
    print("\nOutput with Padding Mask:")
    print(out_with_mask.shape)

    _ = GQA(x_mask, attention_mask = attention_mask, debug = True)
