# 🧠 ncrsh

> The Ultimate Deep Learning Stack — Transformers, Tokenizers, and Torch in One

[![PyPI version](https://img.shields.io/pypi/v/ncrsh)](https://pypi.org/project/ncrsh/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Documentation Status](https://readthedocs.org/projects/ncrsh/badge/?version=latest)](https://ncrsh.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/dinethnethsara/ncrsh/actions/workflows/tests.yml/badge.svg)](https://github.com/dinethnethsara/ncrsh/actions/workflows/tests.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🚀 Overview

ncrsh is a next-generation, all-in-one deep learning library built from the ground up by [Dineth Nethsara](https://github.com/dinethnethsara). Designed to outperform existing solutions, ncrsh fuses powerful model architectures, high-efficiency tokenization, and hardware-accelerated training into one unified toolkit.

## 🔥 Core Features

- **Ultra-Optimized Transformer Architectures**
  - Custom attention mechanisms with superior speed and accuracy
  - Support for various attention patterns (full, sparse, local)
  - Memory-efficient processing for long sequences

- **Integrated Tokenization System**
  - High-performance tokenization engine
  - Support for multiple algorithms (BPE, WordPiece, SentencePiece)
  - Zero-copy tokenization for maximum throughput

- **High-Performance Compute Engine**
  - Torch-compatible tensor operations
  - Native CUDA/Metal acceleration
  - Automatic mixed precision training

- **Developer Experience**
  - Intuitive PyTorch-like API
  - Comprehensive documentation
  - Extensive test coverage

## 🚀 Quick Start

### Installation

```bash
pip install ncrsh
```

### Basic Usage

```python
import ncrsh
import ncrsh.nn as nn

# Create a simple neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)

# Initialize model and process sample input
model = Net()
x = ncrsh.randn(32, 1, 28, 28)
output = model(x)
print(f"Output shape: {output.shape}")
```

## 📚 Documentation

For detailed documentation, tutorials, and API reference, please visit [ncrsh.readthedocs.io](https://ncrsh.readthedocs.io).

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by PyTorch, TensorFlow, and HuggingFace Transformers
- Built with ❤️ by [Dineth Nethsara](https://github.com/dinethnethsara)

---

> "Why choose between PyTorch, TensorFlow, or HuggingFace when you can have something better than all three — in one?"
> 
> — Dineth Nethsara, Creator of ncrsh
