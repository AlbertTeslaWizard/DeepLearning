import torch 
import torch.nn as nn

from demo_TransformerStack import TransformerStack
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

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = "I love Yuri Nakamura and Artificial Intelligence!"

    input_ids = tokenizer.encode(
        text,
        add_special_tokens = False,
        return_tensors = "pt"
    ).to(device)

    model = LLMBackbone(
        vocab_size = tokenizer.vocab_size,
        num_layers = 4,
        d_model = 64,
        num_heads = 4,
        num_kv_heads = 2,
        d_ff = 256
    ).to(device)

    output = model(input_ids)

    print("Input IDs shape:")
    print(input_ids.shape)
    
    print("\nOutput shape:")
    print(output.shape)




    


