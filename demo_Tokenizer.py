from transformers import AutoTokenizer

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = "I love deep learning."

    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, add_special_tokens = False)

    decoded_text = tokenizer.decode(token_ids)

    print(f"Original text:{text}\nTokens:{tokens}\nToken IDs:{token_ids}\nToken < - > ID:")
    for token, token_id in zip(tokens, token_ids):
        print(f"{token:15s} -> {token_id}")

    print("\nDecoded text:")
    print(decoded_text)

    print("\nVocabulary size:")
    print(tokenizer.vocab_size)
