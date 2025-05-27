# Data Loading in ncrsh

This guide covers how to load and preprocess data in ncrsh using the `Dataset` and `DataLoader` classes.

## Table of Contents

1. [Dataset Types](#dataset-types)
2. [Creating Custom Datasets](#creating-custom-datasets)
3. [Using DataLoader](#using-dataloader)
4. [Data Transforms](#data-transforms)
5. [Working with Image Data](#working-with-image-data)
6. [Best Practices](#best-practices)

## Dataset Types

ncrsh provides two main dataset classes:

1. **Dataset**: For map-style datasets (implements `__getitem__` and `__len__`)
2. **IterableDataset**: For stream-style datasets (implements `__iter__`)

## Creating Custom Datasets

### Map-style Dataset Example

```python
from ncrsh.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, target
```

### Iterable-style Dataset Example

```python
from ncrsh.data import IterableDataset

class StreamDataset(IterableDataset):
    def __init__(self, data_generator):
        self.data_generator = data_generator
    
    def __iter__(self):
        for data, target in self.data_generator():
            yield data, target
```

## Using DataLoader

The `DataLoader` combines a dataset and a sampler, and provides an iterable over the given dataset.

```python
from ncrsh.data import DataLoader

# Create dataset
dataset = CustomDataset(data, targets)

# Create dataloader
dataloader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Iterate through the dataloader
for batch_idx, (inputs, targets) in enumerate(dataloader):
    # Training code here
    pass
```

## Data Transforms

Transforms are commonly used for data preprocessing and augmentation.

```python
class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, x):
        return (x - self.mean) / self.std

# Usage
transform = Compose([
    RandomHorizontalFlip(),
    RandomCrop(32, padding=4),
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])
])

dataset = CustomDataset(data, targets, transform=transform)
```

## Working with Image Data

For image data, you can use the following pattern:

```python
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Assuming filename format: class_imageid.jpg
        target = int(self.img_names[idx].split('_')[0])
        
        return image, target
```

## Best Practices

1. **Use num_workers > 0**: For CPU-bound tasks, increase `num_workers` to speed up data loading.
2. **Enable pin_memory**: Set `pin_memory=True` when using CUDA for faster data transfer to GPU.
3. **Prefetch**: The DataLoader automatically prefetches data for the next batch.
4. **Memory Efficiency**: For large datasets, use `IterableDataset` or memory-mapped arrays.
5. **Reproducibility**: Set random seeds and use `worker_init_fn` for consistent data loading.

## Common Issues and Solutions

1. **Memory Leaks**:
   - Ensure you're not storing references to batches
   - Use `del` to free up memory when needed

2. **Slow Data Loading**:
   - Increase `num_workers`
   - Use `pin_memory` with CUDA
   - Consider using a faster storage medium (e.g., SSD)

3. **Data Loading Bottlenecks**:
   - Profile your data loading pipeline
   - Consider preprocessing data in advance
   - Use `torch.multiprocessing` for CPU-intensive transforms
