import torch 
import torch.nn as nn

from llm.modeling.demo_LLM_Backbone_AttentionMask import LLMBackbone
from transformer.components.demo_RMSNorm import RMSNorm
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
    
    def forward(self, input_ids, attention_mask=None):
        x = self.backbone(input_ids, attention_mask = attention_mask)
        x = self.norm(x)

        logits = self.lm_head(x)
        return logits

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
    
    vocab_size = tokenizer.vocab_size 
    model = MiniLLM(
        vocab_size = vocab_size, 
        num_layers = 4, 
        d_model = 64,
        num_heads = 4,
        num_kv_heads = 2,
        d_ff = 256
    ).to(device)

    logits = model(input_ids, attention_mask = attention_mask)

    print("\nLogits shape:")
    print(logits.shape)

    print("\nVocabulary size:")
    print(vocab_size)
