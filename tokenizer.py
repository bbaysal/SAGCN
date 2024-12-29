from transformers import BertTokenizer
import torch


def demonstrate_bert_tokenizer():
    # 1. Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 2. Basic tokenization example
    text = "Hello, how are you? I'm doing great!"

    # Simple tokenization
    tokens = tokenizer.tokenize(text)
    print("\n1. Basic tokenization:")
    print(f"Original text: {text}")
    print(f"Tokens: {tokens}")

    # 3. Convert tokens to IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print("\n2. Token IDs:")
    print(f"Token IDs: {token_ids}")

    # 4. Encode the text directly (includes special tokens)
    encoded = tokenizer.encode(text, add_special_tokens=True)
    print("\n3. Full encoding with special tokens:")
    print(f"Encoded: {encoded}")
    print(f"Decoded back: {tokenizer.decode(encoded)}")

    # 5. Handling multiple sentences with padding
    texts = [
        "Hello, how are you?",
        "This is a longer sentence that will need padding.",
        "Short one."
    ]

    # Encode and pad the batch
    encoded_batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors='pt'
    )

    print("\n4. Batch encoding with padding:")
    print(f"Input IDs shape: {encoded_batch['input_ids'].shape}")
    print(f"Attention mask shape: {encoded_batch['attention_mask'].shape}")

    # 6. Demonstrate subword tokenization
    technical_text = "transformers preprocessing tokenization"
    subword_tokens = tokenizer.tokenize(technical_text)
    print("\n5. Subword tokenization example:")
    print(f"Original text: {technical_text}")
    print(f"Subword tokens: {subword_tokens}")

    # 7. Special tokens example
    print("\n6. Special tokens:")
    print(f"CLS token: {tokenizer.cls_token}")
    print(f"SEP token: {tokenizer.sep_token}")
    print(f"PAD token: {tokenizer.pad_token}")

    # 8. Example with special tokens for sentence pairs
    sentence1 = "How are you?"
    sentence2 = "I am fine."
    encoded_pair = tokenizer(
        sentence1,
        sentence2,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors='pt'
    )

    print("\n7. Sentence pair encoding:")
    print(f"Input IDs: {encoded_pair['input_ids']}")
    print(f"Token Type IDs: {encoded_pair['token_type_ids']}")

    # 9. Demonstrate max length truncation
    long_text = "This is a very long text that will be truncated by the tokenizer because it exceeds the maximum length we set."
    encoded_truncated = tokenizer(
        long_text,
        max_length=10,
        truncation=True,
        return_tensors='pt'
    )

    print("\n8. Truncation example:")
    print(f"Original text length: {len(long_text.split())}")
    print(f"Truncated token IDs: {encoded_truncated['input_ids']}")
    print(f"Decoded truncated text: {
          tokenizer.decode(encoded_truncated['input_ids'][0])}")


if __name__ == "__main__":
    demonstrate_bert_tokenizer()
