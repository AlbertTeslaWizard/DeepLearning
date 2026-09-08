import math

import torch
import torch.nn as nn


class RoPE(nn.Module):
    """
    Rotary Position Embedding（旋转位置编码）

    核心思想：
    ------------------------------------------------------------
    对 Attention 中的 Q、K 向量注入位置信息。

    对于位置 pos 和第 i 组二维分量，对应旋转角度：

        theta(pos, i) = pos * theta_i

    其中：

        theta_i = 1 / base^(2i / d)

    d = head_dim

    通常：

        base = 10000

    所以：

        theta_i
        = 10000^(-2i / d)

    为了方便计算，可以改写为指数形式：

        a^x = e^(x ln a)

    因此：

        10000^(-2i / d)
        =
        e^(-(2i / d) * ln(10000))

    最终：

        theta(pos, i)
        =
        pos * e^(-(2i / d) * ln(base))


    RoPE 最终并不是直接把位置向量加到 x 上，
    而是将 Q / K 的二维分量进行旋转：

        [x]
        [y]

        ->

        [x cos(theta) - y sin(theta)]
        [x sin(theta) + y cos(theta)]

    即：

        R(theta) @ [x, y]^T

    其中旋转矩阵：

                   [ cos(theta)  -sin(theta) ]
        R(theta) = [                        ]
                   [ sin(theta)   cos(theta) ]

    """

    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        self.head_dim = head_dim

        # ------------------------------------------------------------
        # RoPE 要把 head_dim 中的维度两两组成二维平面。
        #
        # 本实现采用“前半 / 后半”配对（Llama / HF 风格）：
        #
        #   维度 i 与 维度 i + head_dim/2 组成一个二维平面
        #
        # 例如 head_dim = 8，共 4 个二维平面：
        #
        #   (0, 4), (1, 5), (2, 6), (3, 7)
        #
        # 这里 arange(0, head_dim, 2) 得到：
        #
        #   0, 2, 4, 6
        #
        # 它们对应公式 theta_i = base^(-2i / d) 中的频率下标：
        #
        #   2i
        #
        # 注意：dim_indices 只决定每个平面的旋转频率（快慢），
        # 平面本身的维度配对由 rotate_half 的 chunk(2)
        # “前半 / 后半”切分决定（维度 i 与 i + head_dim/2 成对）。
        #
        # ------------------------------------------------------------

        dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float)
        inv_freq = torch.exp(-dim_indices * math.log(base) / head_dim)

        # ------------------------------------------------------------
        # inv_freq 不是需要通过反向传播学习的参数，
        # 它只是 RoPE 根据数学公式预先确定的常量。
        #
        # 所以不能写成：
        #
        #   nn.Parameter(inv_freq)
        #
        # 而应该注册为 buffer：
        #
        #   register_buffer(...)
        #
        # 这样它：
        #
        # 1. 不参与梯度更新
        # 2. model.to("cuda") 时会自动移动到 GPU
        # 3. 可以像 self.inv_freq 一样使用
        #
        # persistent=False：
        #
        # 不把它保存进 state_dict。
        #
        # 因为这个值可以随时由公式重新计算出来，
        # 没必要存进模型权重文件。
        # ------------------------------------------------------------

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device):
        """
        根据当前序列长度生成：

            cos(theta)
            sin(theta)

        输出形状：

            [1, seq_len, 1, head_dim]

        用于和：

            Q / K:
            [batch_size, seq_len, num_heads, head_dim]

        进行广播运算。
        """

        # ------------------------------------------------------------
        # Step 1
        # 生成位置索引：
        #
        #   pos = [0, 1, 2, ..., seq_len - 1]
        #
        # 例如：
        #
        # seq_len = 4
        #
        # t =
        # [0, 1, 2, 3]
        #
        # shape:
        #
        # [seq_len]
        #
        # ------------------------------------------------------------

        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        # ------------------------------------------------------------
        # Step 2
        # 计算每个位置、每个二维平面的旋转角度。
        #
        # 数学公式：
        #
        #   theta(pos, i)
        #   =
        #   pos * theta_i
        #
        # 其中：
        #
        #   theta_i = base^(-2i / d)
        #
        #
        # t:
        #
        #   [seq_len]
        #
        # inv_freq:
        #
        #   [head_dim / 2]
        #
        #
        # 使用外积：
        #
        #   freqs = t ⊗ inv_freq
        #
        #
        # 得到：
        #
        #                     第0组     第1组     第2组 ...
        #
        # pos = 0             0θ0       0θ1       0θ2
        # pos = 1             1θ0       1θ1       1θ2
        # pos = 2             2θ0       2θ1       2θ2
        # ...
        #
        #
        # shape：
        #
        #   [seq_len, head_dim / 2]
        #
        # ------------------------------------------------------------

        freqs = torch.outer(t, self.inv_freq)

        # ------------------------------------------------------------
        # Step 3
        # 扩展为 head_dim。
        #
        # freqs 原本：
        #
        #   [seq_len, head_dim / 2]
        #
        # 例如：
        #
        #   [θ0, θ1, θ2, θ3]
        #
        #
        # 复制一次：
        #
        #   [θ0, θ1, θ2, θ3,
        #    θ0, θ1, θ2, θ3]
        #
        #
        # 得到：
        #
        #   [seq_len, head_dim]
        #
        #
        # 这样就能与：
        #
        #   x.shape = [..., head_dim]
        #
        # 对齐。
        #
        # 注意：
        # 这种写法和下面 rotate_half() 的“前半/后半”实现
        # 是配套使用的。
        # ------------------------------------------------------------

        emb = torch.cat((freqs, freqs), dim=-1)

        # ------------------------------------------------------------
        # Step 4
        # 根据旋转公式，需要：
        #
        #   cos(theta)
        #   sin(theta)
        #
        #
        # emb:
        #
        #   [seq_len, head_dim]
        #
        #
        # emb.cos():
        #
        #   [seq_len, head_dim]
        #
        #
        # 为了和 Attention 中：
        #
        #   Q/K:
        #   [B, L, H, D]
        #
        # 广播，需要变成：
        #
        #   [1, L, 1, D]
        #
        #
        # 第一个 1：
        #
        #   对 batch 广播
        #
        # 第二个 1：
        #
        #   对 num_heads 广播
        #
        # ------------------------------------------------------------

        cos = emb.cos().unsqueeze(0).unsqueeze(2)
        sin = emb.sin().unsqueeze(0).unsqueeze(2)

        return cos, sin


def rotate_half(x):
    """
    实现 RoPE 旋转公式中的“90°旋转部分”。

    ------------------------------------------------------------
    假设二维向量：

        v = (x, y)

    将它逆时针旋转 90°：

        (-y, x)

    因为：

                    [ 0  -1 ]
        R(90°) =    [       ]
                    [ 1   0 ]

    所以：

        [ 0 -1 ] [x]   [-y]
        [ 1  0 ] [y] = [ x]


    对高维向量：

        x = (x1, x2)

    其中：

        x1 = 前一半维度
        x2 = 后一半维度

    定义：

        rotate_half(x)
        =
        (-x2, x1)

    ------------------------------------------------------------

    例如：

        x = [a, b, c, d]

    chunk(2)：

        x1 = [a, b]
        x2 = [c, d]

    则：

        rotate_half(x)

        =
        [-c, -d, a, b]

    """

    # 将最后一个维度平均切成两半：
    #
    # x:
    #
    # [..., head_dim]
    #
    # ->
    #
    # x1:
    # [..., head_dim / 2]
    #
    # x2:
    # [..., head_dim / 2]
    x1, x2 = x.chunk(2, dim=-1)

    # (x1, x2)
    #
    # ->
    #
    # (-x2, x1)

    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    """
    真正将 RoPE 应用到 Q 或 K 上。

    x 形状：

        [batch_size, seq_len, num_heads, head_dim]

    说明：

        num_heads 处既可以传 Q 的 num_heads，
        也可以是 GQA 下 KV 共享的 num_kv_heads。

        对应的 cos / sin 形状：

        [1, seq_len, 1, head_dim]

        其中第二个 1 是对 num_heads（或 num_kv_heads）广播，
        因此 Q / K 头数不同时也无需额外处理。


    ============================================================
    为什么公式是：

        x_rot
        =
        x * cos(theta)
        +
        rotate_half(x) * sin(theta)

    ============================================================

    从二维旋转矩阵开始：

                   [ cosθ  -sinθ ]
        R(θ) =     [              ]
                   [ sinθ   cosθ ]


    对向量：

        v = [x, y]^T

    有：

        x' = x cosθ - y sinθ

        y' = x sinθ + y cosθ


    重新整理：

        (x', y')

        =
        (x cosθ, y cosθ)
        +
        (-y sinθ, x sinθ)

    即：

        v_rot
        =
        v * cosθ
        +
        (-y, x) * sinθ


    而：

        rotate_half(v)
        =
        (-y, x)

    因此：

        v_rot
        =
        v * cosθ
        +
        rotate_half(v) * sinθ


    这就是下面这一行代码的数学来源：

        x * cos + rotate_half(x) * sin


    注意：

    这里的 * 是逐元素乘法，

    不是矩阵乘法 @。

    """

    return x * cos + rotate_half(x) * sin
