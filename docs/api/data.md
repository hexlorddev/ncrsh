# Data Loading API Reference

This document provides a comprehensive reference for the data loading utilities in ncrsh.

## Table of Contents

1. [Dataset](#dataset)
2. [DataLoader](#dataloader)
3. [Samplers](#samplers)
4. [Transforms](#transforms)
5. [Utilities](#utilities)

## Dataset

### `Dataset`

Base class for creating datasets.

```python
class ncrsh.data.Dataset
```

**Methods:**
- `__getitem__(self, index)`: Retrieve a sample and its target by index
- `__len__(self)`: Return the size of the dataset

**Example:**
```python
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    
    def __len__(self):
        return len(self.data)
```

### `IterableDataset`

Base class for iterable-style datasets.

```python
class ncrsh.data.IterableDataset
```

**Methods:**
- `__iter__(self)`: Returns an iterator over the dataset

**Example:**
```python
class StreamDataset(IterableDataset):
    def __init__(self, data_generator):
        self.data_generator = data_generator
    
    def __iter__(self):
        return iter(self.data_generator())
```

### `TensorDataset`

Dataset wrapping tensors.

```python
class ncrsh.data.TensorDataset(*tensors)
```

**Parameters:**
- `*tensors`: Tensors that have the same size of the first dimension

**Example:**
```python
data = torch.randn(1000, 3, 32, 32)
targets = torch.randint(0, 10, (1000,))
dataset = TensorDataset(data, targets)
```

## DataLoader

### `DataLoader`

```python
class ncrsh.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    multiprocessing_context=None,
    generator=None,
    prefetch_factor=2,
    persistent_workers=False
)
```

**Parameters:**
- `dataset`: Dataset from which to load the data
- `batch_size`: Number of samples per batch
- `shuffle`: Set to `True` to have the data reshuffled at every epoch
- `sampler`: Defines the strategy to draw samples from the dataset
- `batch_sampler`: Like sampler, but returns a batch of indices at a time
- `num_workers`: How many subprocesses to use for data loading
- `collate_fn`: Merges a list of samples to form a mini-batch
- `pin_memory`: If `True`, the data loader will copy Tensors into CUDA pinned memory
- `drop_last`: Set to `True` to drop the last incomplete batch
- `timeout`: Timeout in seconds for collecting a batch from workers
- `worker_init_fn`: Called on each worker subprocess with the worker id
- `prefetch_factor`: Number of batches loaded in advance by each worker
- `persistent_workers`: If `True`, the data loader will not shut down the worker processes

**Example:**
```python
dataloader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

## Samplers

### `Sampler`

Base class for all samplers.

```python
class ncrsh.data.Sampler(data_source)
```

### `SequentialSampler`

Samples elements sequentially.

```python
class ncrsh.data.SequentialSampler(data_source)
```

### `RandomSampler`

Samples elements randomly.

```python
class ncrsh.data.RandomSampler(
    data_source,
    replacement=False,
    num_samples=None,
    generator=None
)
```

### `SubsetRandomSampler`

Samples elements randomly from a given list of indices.

```python
class ncrsh.data.SubsetRandomSampler(indices, generator=None)
```

### `BatchSampler`

Wraps another sampler to yield a mini-batch of indices.

```python
class ncrsh.data.BatchSampler(
    sampler,
    batch_size,
    drop_last
)
```

## Transforms

### `Compose`

Composes several transforms together.

```python
class ncrsh.transforms.Compose(transforms)
```

### `ToTensor`

Convert a PIL Image or numpy.ndarray to tensor.

```python
class ncrsh.transforms.ToTensor
```

### `Normalize`

Normalize a tensor image with mean and standard deviation.

```python
class ncrsh.transforms.Normalize(mean, std, inplace=False)
```

### `Resize`

Resize the input image to the given size.

```python
class ncrsh.transforms.Resize(size, interpolation=2)
```

## Utilities

### `get_worker_info`

Returns information about the current data loading process.

```python
ncrsh.data.get_worker_info()
```

### `default_collate`

Collate function that converts numpy arrays to tensors.

```python
ncrsh.data.default_collate(batch)
```

### `default_convert`

Convert numpy arrays to tensors.

```python
ncrsh.data.default_convert(data)
```

### `random_split`

Randomly split a dataset into non-overlapping new datasets of given lengths.

```python
ncrsh.data.random_split(dataset, lengths, generator=<torch._C.Generator object>)
```

### `Subset`

Subset of a dataset at specified indices.

```python
class ncrsh.data.Subset(dataset, indices)
```

### `ConcatDataset`

Dataset as a concatenation of multiple datasets.

```python
class ncrsh.data.ConcatDataset(datasets)
```

### `ChainDataset`

Dataset for chainning multiple IterableDataset.

```python
class ncrsh.data.ChainDataset(datasets)
```

## Working with Iterable Datasets

### `IterableDataset`

Base class for iterable-style datasets.

```python
class ncrsh.data.IterableDataset(
    *args,
    **kwargs
)
```

### `get_worker_info`

Returns information about the current data loading process.

```python
ncrsh.data.get_worker_info()
```

### `IterableDataset.shard`

Return a sharded version of the dataset.

```python
IterableDataset.shard(
    num_shards,
    index,
    contiguous_batches=False
)
```

### `IterableDataset.shuffle`

Shuffle the dataset.

```python
IterableDataset.shuffle(
    buffer_size,
    seed=None
)
```

## Distributed Data Loading

### `DistributedSampler`

Sampler that restricts data loading to a subset of the dataset.

```python
class ncrsh.data.distributed.DistributedSampler(
    dataset,
    num_replicas=None,
    rank=None,
    shuffle=True,
    seed=0,
    drop_last=False
)
```

### `get_world_size`

Get the number of processes in the distributed group.

```python
ncrsh.data.distributed.get_world_size()
```

### `get_rank`

Get the rank of the current process in the distributed group.

```python
ncrsh.data.distributed.get_rank()
```

## Performance Tuning

### `pin_memory`

Pin the data in page-locked memory for faster transfer to GPU.

```python
torch.utils.data.pin_memory(data, device=None)
```

### `PrefetchGenerator`

Prefetch data using a background thread.

```python
class ncrsh.data.PrefetchGenerator(loader, num_prefetch=1)
```

### `DataLoader2`

Experimental DataLoader with improved performance features.

```python
class ncrsh.data.DataLoader2(
    dataset,
    batch_size=1,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    multiprocessing_context=None,
    generator=None,
    prefetch_factor=2,
    persistent_workers=False,
    pin_memory_device=""
)
```
