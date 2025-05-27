# Data Loading and Processing

This module provides tools for loading, processing, and managing datasets.

## Core Components

### Dataset Classes
- **Dataset**: Abstract base class for all datasets
- **TensorDataset**: Dataset wrapping tensors
- **ConcatDataset**: For concatenating multiple datasets
- **Subset**: For selecting a subset of a dataset

### Data Loading
- **DataLoader**: Efficient data loading with batching and shuffling
- **Sampler**: Base class for data sampling strategies
- **BatchSampler**: Wraps another sampler to yield a mini-batch of indices

### Data Processing
- **Transforms**: Common data transformations
- **Utils**: Helper functions for data processing

## Usage Example

```python
import ncrsh
from ncrsh.data import DataLoader, TensorDataset
from ncrsh.utils.data_utils import split_dataset

# Create a dataset
data = ncrsh.randn(1000, 10)
targets = ncrsh.randint(0, 2, (1000,))
dataset = TensorDataset(data, targets)

# Split into train/val
train_set, val_set = split_dataset(dataset, [0.8, 0.2])

# Create data loaders
train_loader = DataLoader(
    train_set,
    batch_size=32,
    shuffle=True,
    num_workers=2
)
val_loader = DataLoader(
    val_set,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

# Training loop
for batch_data, batch_targets in train_loader:
    # Training code here
    pass
```

## Creating a Custom Dataset

```python
from ncrsh.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_samples()
    
    def _load_samples(self):
        # Load and return list of samples
        pass
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        # Apply transform if specified
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def __len__(self):
        return len(self.samples)
```

## Best Practices

- Always implement `__len__` and `__getitem__` methods
- Use `num_workers > 0` for faster data loading
- Pin memory when using CUDA (`pin_memory=True`)
- Use transforms for data augmentation
- Handle data loading errors gracefully

## Data Transforms

```python
from ncrsh.data import Compose, RandomCrop, RandomHorizontalFlip, ToTensor

transform = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])
])
```

## Performance Tips

1. **Prefetching**: Use `prefetch_factor` in DataLoader
2. **Memory Pinning**: Use `pin_memory=True` with CUDA
3. **Parallel Loading**: Use `num_workers > 0`
4. **Avoid CPU-GPU Transfers**: Keep data on GPU when possible
5. **Batch Processing**: Use appropriate batch sizes
