import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        
        # 门控分支 w1 和 升维分支 w3
        self.w1 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_model, d_ff)

        # 降维投影 w2
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # SwiGLU(x) = (SiLU(W1 * x) * (W3 * x)) * W2
        # SiLU(z) = z * sigmoid(z) = z / (1 + exp(-z))
        gate = F.silu(self.w1(x))
        up = self.w3(x)

        return self.w2(gate * up) 

if __name__ == '__main__':
    torch.manual_seed(42)

    batch_size, seq_len, d_model, d_ff = 2, 4, 8, 16
    x = torch.randn(batch_size, seq_len, d_model)

    ffn = SwiGLUFFN(d_model = d_model, d_ff = d_ff)

    output = ffn(x)
    print(f"输入 Tensor 形状: {x.shape}")
    print(f"输出 Tensor 形状: {output.shape}")
