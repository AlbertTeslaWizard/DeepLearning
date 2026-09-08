import torch 
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only = True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    texts = [
        "I love deep learning",
        "Artificial intelligence is amazing",
        "I love Yuri Nakamura"
    ]

    tokenizer.pad_token = tokenizer.eos_token

    batch = tokenizer(
        texts,
        padding = True,
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
    
    for text in texts:
        tokens = tokenizer.tokenize(text)
        print(text)
        print(tokens)
