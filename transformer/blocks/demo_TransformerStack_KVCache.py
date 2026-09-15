import torch 
import torch.nn as nn

from transformer.blocks.demo_TransformerBlock_RoPE_KVCache import TransformerBlock

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

    def forward(self, x, past_kv = None, use_cache = False):
        if past_kv is not None and not use_cache:
            raise ValueError("past_kv requires use_cache=True")

        # =========================================================
        # 不使用 KV Cache
        # =========================================================
        if not use_cache:
            for layer in self.layers:
                x = layer(x)
            
            return x 
        
        # =========================================================
        # 使用 KV Cache
        # =========================================================

        # Prefill 时：
        #
        # past_kv = None
        #
        # 每一层暂时都没有历史 Cache。

        if past_kv is None:
            past_kv = [None] * len(self.layers)
        elif len(past_kv) != len(self.layers):
            raise ValueError("past_kv must contain one entry per Transformer layer")
        
        # 保存每一层最新的 KV Cache。
        present_kv = []

        # 每一层使用属于自己的 past_kv。
        for layer, layer_past_kv in zip(self.layers, past_kv):
            x, layer_present_kv = layer(x, past_kv = layer_past_kv, use_cache = True)
            present_kv.append(layer_present_kv)

        return x, present_kv
        
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
