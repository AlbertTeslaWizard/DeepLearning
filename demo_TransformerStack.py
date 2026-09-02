import torch 
import torch.nn as nn

from demo_TransformerBlock_RoPE import TransformerBlock

class TransformerStack(nn.Module):
    def __init__(self, num_layers = 4, d_model = 64, num_heads = 4, num_kv_heads = 2, d_ff = 256):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model = d_model,
                    num_heads = num_heads,
                    num_kv_heads = num_kv_heads,
                    d_ff = d_ff
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    torch.manual_seed(42)

    x = torch.randn(1, 4, 64, device = device)
    model = TransformerStack(
        num_layers = 4,
        d_model = 64,
        num_heads = 4,
        num_kv_heads = 2,
        d_ff = 256
    ).to(device)

    output = model(x)

    print("Input shape:")
    print(x.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nNumber of layers:")
    print(len(model.layers))

    print("\nParameters:")
    print(f"{sum(p.numel() for p in model.parameters())}")
