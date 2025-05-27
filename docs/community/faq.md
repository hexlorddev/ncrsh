# Frequently Asked Questions (FAQ)

## General

### What is ncrsh?
ncrsh is a deep learning library focused on providing efficient and flexible tools for training neural networks, with a strong emphasis on data loading and processing capabilities.

### What are the system requirements?
- Python 3.8 or higher
- pip (Python package manager)
- (Optional) CUDA for GPU acceleration

### How do I install ncrsh?
```bash
pip install ncrsh
```

## Data Loading

### What's the difference between Dataset and IterableDataset?
- **Dataset**: Implements `__getitem__` and `__len__`. Good for random access to data.
- **IterableDataset**: Implements `__iter__`. Better for streaming data or when random access is expensive.

### How do I use multiple workers for data loading?
Set `num_workers` in the DataLoader:
```python
dataloader = DataLoader(dataset, num_workers=4)
```

### What's the purpose of `pin_memory`?
`pin_memory=True` enables faster data transfer to CUDA devices. Use it when training on GPU.

## Training

### How do I save and load models?
```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model = ModelClass()
model.load_state_dict(torch.load('model.pth'))
```

### What optimizers are available?
- SGD
- Adam
- AdamW
- RMSprop
- And more in `ncrsh.optim`

### How do I use learning rate scheduling?
```python
from ncrsh.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# In training loop
scheduler.step()
```

## Performance

### How can I speed up data loading?
1. Increase `num_workers`
2. Use `pin_memory=True` with CUDA
3. Use `prefetch_factor` to load next batches in advance
4. Consider using `IterableDataset` for large datasets

### How do I use mixed precision training?
```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Contributing

### How can I contribute to ncrsh?
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### What coding standards should I follow?
- Follow PEP 8
- Use type hints
- Write docstrings for all public functions and classes
- Add tests for new features

## Troubleshooting

### I'm getting CUDA out of memory errors
1. Reduce batch size
2. Use gradient accumulation
3. Clear cache: `torch.cuda.empty_cache()`
4. Check for memory leaks in your code

### Data loading is slow
1. Increase `num_workers`
2. Use SSD instead of HDD
3. Preprocess data in advance
4. Use `pin_memory=True` with CUDA

### My model isn't learning
1. Check your learning rate
2. Normalize your input data
3. Check for vanishing/exploding gradients
4. Try overfitting a small batch first

## Advanced Topics

### How do I implement custom datasets?
Extend the `Dataset` class and implement `__getitem__` and `__len__`:
```python
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        if self.transform:
            item = self.transform(item)
        return item
```

### How do I use custom collate functions?
```python
def custom_collate(batch):
    # Custom collation logic
    return batch

dataloader = DataLoader(dataset, collate_fn=custom_collate)
```

## Getting Help

### Where can I get help?
- Check the [documentation](https://ncrsh.readthedocs.io)
- Open an issue on GitHub
- Join our community forum (coming soon)

### How do I report a bug?
Open an issue on GitHub with:
1. A clear description of the bug
2. Steps to reproduce
3. Expected vs actual behavior
4. Environment details (Python version, OS, etc.)
5. Error messages or logs
