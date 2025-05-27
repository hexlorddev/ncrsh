"""
Sampling strategies for data loading in ncrsh.
"""
import math
import warnings
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, TypeVar, Union, overload

import numpy as np

from ..tensor import Tensor

T_co = TypeVar('T_co', covariant=True)

class Sampler(Generic[T_co]):
    """Base class for all samplers.
    
    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.
    """
    
    def __init__(self, data_source: Optional[Any] = None) -> None:
        pass
    
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError
    
    def __len__(self) -> int:
        raise NotImplementedError


class SequentialSampler(Sampler[int]):
    """Samples elements sequentially, always in the same order.
    
    Args:
        data_source: Dataset to sample from
    """
    
    def __init__(self, data_source: Any) -> None:
        self.data_source = data_source
    
    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))
    
    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler[int]):
    """Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    
    If with replacement, then user can specify :attr:`num_samples` to draw.
    
    Args:
        data_source: Dataset to sample from
        replacement: Samples are drawn with replacement if ``True``
        num_samples: Number of samples to draw, default is the length of the dataset.
            This argument is supposed to be specified only when `replacement` is ``True``.
        generator: Generator used in sampling.
    """
    
    def __init__(
        self,
        data_source: Any,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator = None
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        
        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))
                            
        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")
                             
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))
    
    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples
    
    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator
            
        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            yield from torch.randperm(n, generator=generator).tolist()
    
    def __len__(self) -> int:
        return self.num_samples


class SubsetRandomSampler(Sampler[int]):
    """Samples elements randomly from a given list of indices, without replacement.
    
    Args:
        indices: A sequence of indices
        generator: Generator used in sampling.
    """
    
    def __init__(self, indices: Sequence[int], generator = None) -> None:
        self.indices = indices
        self.generator = generator
    
    def __iter__(self) -> Iterator[int]:
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator
            
        for i in torch.randperm(len(self.indices), generator=generator):
            yield self.indices[i]
    
    def __len__(self) -> int:
        return len(self.indices)


class WeightedRandomSampler(Sampler[int]):
    """Samples elements from [0,..,len(weights)-1] with given probabilities (weights).
    
    Args:
        weights: A sequence of weights, not necessary summing up to one
        num_samples: Number of samples to draw
        replacement: If ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator: Generator used in sampling.
    """
    
    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        generator = None
    ) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
                             
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
                             
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
    
    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        yield from iter(rand_tensor.tolist())
    
    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler[List[int]]):
    """Wraps another sampler to yield a mini-batch of indices.
    
    Args:
        sampler: Base sampler. Can be any iterable object
        batch_size: Size of mini-batch.
        drop_last: If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """
    
    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
                             
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
                             
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
