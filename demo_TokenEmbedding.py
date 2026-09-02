import torch
import torch.nn as nn

from transformers import AutoTokenizer
from demo_TransformerBlock_RoPE import TransformerBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = "I love deep learning"

    input_ids = tokenizer.encode(
        text,
        add_special_tokens = False,
        return_tensors = "pt" # 把 tokenzier 的输出直接转换成哪一种张量格式
    ).to(device)

    d_model = 64

    embedding = nn.Embedding(num_embeddings = tokenizer.vocab_size, embedding_dim = d_model).to(device)

    x = embedding(input_ids)
    model = TransformerBlock(
        d_model = 64,
        num_heads = 4,
        num_kv_heads = 2,
        d_ff = 256
    ).to(device)

    output = model(x)
   
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print("Text:")
    print(text)

    print("\nTokens:")
    print(tokens)

    print("\nTokens IDs:")
    print(input_ids)

    print("\nToken IDs shape:")
    print(input_ids.shape)

    print("\nEmbedding shape:")
    print(x.shape)

    print("\nTransformer output shape:")
    print(output.shape)



