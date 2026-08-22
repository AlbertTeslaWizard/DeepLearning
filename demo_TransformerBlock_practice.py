import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerBlock(nn.Module):
    def __init__(self, num_heads = 4, d_model = 64, d_ff = 256):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads = num_heads,
            batch_first = True 
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attention(
            query = x,
            key = x,
            value = x
        )

        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x

if __name__ == '__main__':
    torch.manual_seed(42)
    x = torch.randn(32, 10, 64, device=device)

    model = TransformerBlock(
        num_heads = 4,
        d_model = 64, 
        d_ff = 256,
    ).to(device)

    output = model(x)
    params = sum([p.numel() for p in model.parameters()])
    
    print(f"Using Device: {device}")
    print(f"Params: {params}")
    print(f"Input: {x.shape} -> Output: {output.shape}")


