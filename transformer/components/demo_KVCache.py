import torch 
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_heads(x, num_heads):
    """
        [B, L, D]
            ↓
        [B, H, L, D_head]
    """

    B, L, D = x.shape
    d_head = D // num_heads

    x = x.view(B, L, num_heads, d_head)
    x = x.transpose(1, 2)

    return x

def attention(Q, K, V):
    """
        Q: [B, H, L_q, D_head]
        K: [B, H, L_kv, D_head]
        V: [B, H, L_kv, D_head]
    """

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    attention_weights = torch.softmax(scores, dim = -1)

    output = torch.matmul(attention_weights, V)
    return output, attention_weights

if __name__ == '__main__':
    torch.manual_seed(42)

    d_model = 8
    num_heads = 2

    # =========================================================
    # Q / K / V Projection
    # =========================================================

    W_q = nn.Linear(d_model, d_model, bias = False).to(device)
    W_k = nn.Linear(d_model, d_model, bias = False).to(device)
    W_v = nn.Linear(d_model, d_model, bias = False).to(device)

    # =========================================================
    # Step 1
    # Prefill: Prompt = [A, B, C]
    # =========================================================

    # 用随机向量模拟三个 Token 的 hidden states。
    #
    # [B, L, D] = [1, 3, 8]

    x_prompt = torch.randn(1, 3, d_model).to(device)
    
    Q_prompt = split_heads(W_q(x_prompt), num_heads)
    K_prompt = split_heads(W_k(x_prompt), num_heads)
    V_prompt = split_heads(W_v(x_prompt), num_heads)
    
    # KV Cache 保存 Prompt 的 K 和 V。
    K_cache = K_prompt 
    V_cache = V_prompt

    print("Prompt: A B C")
    print("\nQ Prompt shape:")
    print(Q_prompt.shape)

    print("\nK Cache shape:")
    print(K_cache.shape)

    print("\nV Cache shape:")
    print(V_cache.shape)

    # =========================================================
    # Step 2
    # 假设新 Token D 已经生成
    #
    # 现在需要根据：
    #
    # A B C D
    #
    # 继续预测下一个 Token E。
    # =========================================================

    # 只输入新的 Token D。
    #
    # [B, 1, D]
    
    x_new = torch.randn(1, 1, d_model).to(device)
    Q_new = split_heads(W_q(x_new), num_heads)
    K_new = split_heads(W_k(x_new), num_heads)
    V_new = split_heads(W_v(x_new), num_heads)

    print("\nNew Token: D")

    print("\nQ New shape:")
    print(Q_new.shape)
    
    print("\nK New shape:")
    print(K_new.shape)

    # =========================================================
    # Step 3
    # 更新 KV Cache
    # =========================================================

    # 原来：
    #
    # K_cache = [K_A, K_B, K_C]
    #
    # 加入 D：
    #
    # K_cache = [K_A, K_B, K_C, K_D]

    K_cache = torch.cat([K_cache, K_new], dim = 2)
    V_cache = torch.cat([V_cache, V_new], dim = 2)

    print("\nUpdated K Cache shape:")
    print(K_cache.shape)

    print("\nUpdate V Cache shape:")
    print(V_cache.shape)

    # =========================================================
    # Step 4
    # 当前 Token D 查询整个 KV Cache
    # =========================================================
    
    output_cache, attention_weights = attention(Q_new, K_cache, V_cache)

    print("\nAttention weigths shape:")
    print(attention_weights.shape)

    print("\nOutput with KV Cache shape:")
    print(output_cache.shape)

    # =========================================================
    # Step 5
    # 对照实验：不用 KV Cache
    # =========================================================

    # 不使用 Cache 时，需要重新处理：
    #
    # A B C D

    x_full = torch.cat([x_prompt, x_new], dim = 1)
    Q_full = split_heads(W_q(x_full), num_heads)
    K_full = split_heads(W_k(x_full), num_heads)
    V_full = split_heads(W_v(x_full), num_heads)
    
    # 最后一个 Token D 的 Query
    Q_last = Q_full[:, :, -1:, :]
    output_no_cache, _ = attention(Q_last, K_full, V_full)

    print("\nOutput without KV Cache shape:")
    print(output_no_cache.shape)

    # =========================================================
    # Step 6
    # 验证两种方法结果相同
    # =========================================================

    same = torch.allclose(
        output_cache,
        output_no_cache,
        atol = 1e-6
    )

    print("\nKV Cache result == Full recomputation:")
    print(same)


