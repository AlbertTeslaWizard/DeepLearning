import torch 
import torch.nn as nn

from transformer.blocks.demo_TransformerStack_AttentionMask import TransformerStack
from transformers import AutoTokenizer

class LLMBackbone(nn.Module):
    def __init__(self, vocab_size, num_layers = 4, d_model = 64, num_heads = 4, num_kv_heads = 2, d_ff = 256):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = d_model)
        self.transformer = TransformerStack(
            num_layers = num_layers,
            d_model = d_model,
            num_heads = num_heads,
            num_kv_heads = num_kv_heads,
            d_ff = d_ff 
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.transformer(x, attention_mask = attention_mask)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    texts = [
        "I love AI",
        "Artificial Intelligence is very interesting",
        "I love Yuri Nakamura"
    ]

    batch = tokenizer(
        texts,
        padding = True,
        add_special_tokens = False,
        return_tensors = "pt"
    )

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    print("Input IDs:")
    print(input_ids)

    print("\nInput IDs shape:")
    print(input_ids.shape)

    print("\nAttention Mask:")
    print(attention_mask)

    print("\nAttention Mask shape:")
    print(attention_mask.shape)

    model = LLMBackbone(
        vocab_size = tokenizer.vocab_size,
        num_layers = 4,
        d_model = 64,
        num_heads = 4,
        num_kv_heads = 2,
        d_ff = 256
    ).to(device)

    output = model(input_ids, attention_mask = attention_mask)
    
    print("\nOutput shape:")
    print(output.shape)




    


