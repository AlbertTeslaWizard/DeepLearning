import torch 
import torch.nn as nn

from demo_LLM_Backbone import LLMBackbone
from demo_RMSNorm import RMSNorm
from transformers import AutoTokenizer

class MiniLLM(nn.Module):
    def __init__(self, vocab_size, num_layers = 4, d_model = 64, num_heads = 4, num_kv_heads = 2, d_ff = 256):
        super().__init__()

        self.backbone = LLMBackbone(
            vocab_size = vocab_size,
            num_layers = num_layers,
            d_model = d_model,
            num_heads = num_heads,
            num_kv_heads = num_kv_heads,
            d_ff = d_ff
        )

        self.norm = RMSNorm(dim = d_model)
        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias = False
        )
    
    def forward(self, input_ids):
        x = self.backbone(input_ids)
        x = self.norm(x)

        logits = self.lm_head(x)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    torch.manual_seed(42)
    
    text = "I love Computer Science and Yuri Nakamura!"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    input_ids = tokenizer.encode(
        text = text,
        add_special_tokens = False,
        return_tensors = "pt"
    ).to(device)
    
    vocab_size = tokenizer.vocab_size 
    model = MiniLLM(
        vocab_size = vocab_size, 
        num_layers = 4, 
        d_model = 64,
        num_heads = 4,
        num_kv_heads = 2,
        d_ff = 256
    ).to(device)

    output = model(input_ids)
    print(output.shape)



