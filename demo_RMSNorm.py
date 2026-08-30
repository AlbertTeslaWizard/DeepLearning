import torch 
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim = True)

        # 1 / RMS(x) RMS(x) = sqrt(variance + eps)
        rrms = torch.rsqrt(variance + self.eps)
        
        # RMSNorm(x) = (x / RMS(x)) * γ
        return x * rrms * self.weight

if __name__ == '__main__':
    torch.manual_seed(42)

    batch_size, seq_len, dim = 2, 4, 8
    x = torch.randn(batch_size, seq_len, dim)

    norm = RMSNorm(dim)
    output = norm(x)
    
    print(f"输入 Tensor 形状: {x.shape}")
    print(f"输出 Tensor 形状: {output.shape}")
    print("\n 第一个token 归一化前的均方根: ", torch.sqrt(x[0, 0].pow(2).mean()).item())
    print("\n 第一个token 归一化后的均方根: ", torch.sqrt(output[0, 0].pow(2).mean()).item())


