import torch
import torch.optim as optim 

from transformers import AutoTokenizer

from demo_MiniLLM import MiniLLM
from demo_NextTokenLoss import next_token_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    torch.manual_seed(42)
    text = (
        "I love deep learning.\n"
        "I love Yuri Nakamura.\n"
        "Deep learning is magical."
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode(
        text,
        add_special_tokens = False,
        return_tensors = "pt"
    ).to(device)

    model = MiniLLM(
        vocab_size = tokenizer.vocab_size,
        num_layers = 4,
        d_model = 64,
        num_heads = 4,
        num_kv_heads = 2,
        d_ff = 256
    ).to(device)

    model.train()

    optimizer = optim.AdamW(model.parameters(), lr = 1e-3)
    epochs = 500

    for epoch in range(epochs):
        logits = model(input_ids)
        loss = next_token_loss(logits, input_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                    f"Loss: {loss.item():.4f}"
            )

