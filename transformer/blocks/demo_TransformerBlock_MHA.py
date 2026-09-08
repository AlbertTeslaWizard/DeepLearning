import torch 
import torch.nn as nn
import math 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model = 64, num_heads = 4):
        super().__init__()
        
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        
        # (B, L, D)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # (B, L, D) => (B, L, num_heads, d_head) => (B, num_heads, L, d_head)
        Q = Q.view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        
        # (B, num_heads, L, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head) 
        attention = torch.softmax(scores, dim = -1)

        # 对 V 加权求和
        # (B, num_heads, L, L) => (B, num_heads, L, d_head)
        out = torch.matmul(attention, V)
        
        # (B, num_heads, L, d_head) => (B, L, num_heads, d_head)
        out = out.transpose(1, 2).contiguous()

        # (B, L, num_heads, d_head) => (B, L, D)
        out = out.view(B, L, D)

        out = self.W_o(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, num_heads = 4, d_model = 64, d_ff = 256):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_model = d_model,
            num_heads = num_heads
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        
        ff_out = self.ffn(x)
        x = self.norm2(x + ff_out)
        
        return x

if __name__ == '__main__':
    torch.manual_seed(42)

    x = torch.randn(32, 10, 64, device = device)
    model = TransformerBlock(
        d_model = 64,
        num_heads = 4,
        d_ff = 256
    ).to(device)

    output = model(x)
    params = sum(p.numel() for p in model.parameters())


    print(f"Device: {device}")
    print(f"Params: {params:,}")
    print(f"Shapes : In {tuple(x.shape)} -> Out {tuple(output.shape)}")

