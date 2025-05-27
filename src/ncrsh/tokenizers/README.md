# Tokenizers

This module provides tokenization utilities for natural language processing tasks.

## Features

- **Word-level tokenization**
- **Subword tokenization** (BPE, WordPiece, etc.)
- **Character-level tokenization**
- **Custom vocabulary support**
- **Special tokens handling**
- **Batch processing**

## Available Tokenizers

- **BasicTokenizer**: Simple whitespace and punctuation-based tokenizer
- **WordTokenizer**: Word-level tokenizer with custom vocabulary
- **BPETokenizer**: Byte Pair Encoding tokenizer
- **WordPieceTokenizer**: WordPiece tokenizer
- **CharacterTokenizer**: Character-level tokenizer

## Usage Example

```python
from ncrsh.tokenizers import WordTokenizer, BPETokenizer

# Word-level tokenization
tokenizer = WordTokenizer()
tokens = tokenizer.tokenize("Hello, world! This is a test.")
# Output: ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'test', '.']

# BPE tokenization
bpe_tokenizer = BPETokenizer()
bpe_tokenizer.train(["sample_text.txt"], vocab_size=1000)
tokens = bpe_tokenizer.tokenize("Hello, world!")
```

## Creating a Custom Tokenizer

```python
from ncrsh.tokenizers import BaseTokenizer

class CustomTokenizer(BaseTokenizer):
    def __init__(self, vocab=None):
        super().__init__(vocab)
        # Initialize your tokenizer
    
    def tokenize(self, text):
        # Implement tokenization logic
        return tokens
    
    def train(self, files, **kwargs):
        # Implement training logic if needed
        pass
```

## Best Practices

- Always implement `tokenize` and `train` methods
- Handle out-of-vocabulary tokens
- Support batch processing
- Include special tokens (e.g., [PAD], [UNK], [BOS], [EOS])
- Provide methods for saving/loading tokenizers

## Saving and Loading

```python
# Save tokenizer
tokenizer.save("path/to/save")

# Load tokenizer
loaded_tokenizer = WordTokenizer.load("path/to/save")
```

## Performance Tips

- Use batch processing for large datasets
- Cache tokenization results when possible
- Consider using multiprocessing for CPU-bound tasks
- Pre-tokenize and cache data when training models
