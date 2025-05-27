"""
Data loading utilities for ncrsh.
"""
import os
import sys
import time
import traceback
import warnings
import threading
import itertools
import queue
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, Generic, Iterable, Iterator, List, Optional, 
    Sequence, TypeVar, Union, overload, Type, cast
)

from ..tensor import Tensor
from .sampler import Sampler, SequentialSampler, RandomSampler, BatchSampler
from .dataset import Dataset, IterableDataset, _DatasetKind
from .dataloader_iter import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
from ._utils import get_worker_info as _get_worker_info
from ._utils import default_collate

__all__ = ['DataLoader', 'get_worker_info']

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

# Default collation function
def _default_collate(batch):
    return default_collate(batch)

class DataLoader(Generic[T_co]):
    """Data loader that loads data in mini-batches.
    
    Args:
        dataset: Dataset to load data from
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        sampler: Sampler to use for indices
        batch_sampler: Sampler that returns batches of indices
        num_workers: Number of worker processes for data loading
        collate_fn: Function to collate samples into batches
        pin_memory: Whether to pin memory for CUDA
        drop_last: Whether to drop the last incomplete batch
        timeout: Timeout for collecting a batch from workers
        worker_init_fn: Function to initialize workers
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Whether to keep workers alive between epochs
    """
    
    def __init__(
        self,
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler[int]] = None,
        batch_sampler: Optional[Sampler[Sequence]] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[List[T_co]], Any]] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        # Initialize dataset and dataset kind
        self.dataset = dataset
        self._dataset_kind = _DatasetKind.get_kind(dataset)
        
        # Initialize sampler and batch_sampler
        if batch_sampler is not None:
            # User provided batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with '
                               'batch_size, shuffle, sampler, and drop_last')
            self.batch_size = None
            self.drop_last = False
            self.sampler = None
            self.batch_sampler = batch_sampler
        else:
            # Construct sampler and batch_sampler
            if sampler is not None:
                # User provided sampler
                self.sampler = sampler
            else:
                # Default sampler
                if self._dataset_kind == _DatasetKind.Map:
                    self.sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
                else:  # Iterable
                    self.sampler = None
            
            if batch_size is None:
                batch_size = 1
            
            self.batch_size = batch_size
            self.drop_last = drop_last
            self.batch_sampler = BatchSampler(self.sampler, batch_size, drop_last)
        
        # Initialize other parameters
        self.num_workers = num_workers
        self.collate_fn = collate_fn if collate_fn is not None else _default_collate
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        
        # Validate arguments
        if num_workers < 0:
            raise ValueError('num_workers should be non-negative')
        
        if timeout < 0:
            raise ValueError('timeout should be non-negative')
        
        if num_workers == 0 and persistent_workers:
            raise ValueError('persistent_workers option needs num_workers > 0')
        
        # Set default collate function for common cases
        if self._dataset_kind == _DatasetKind.Iterable and collate_fn is None:
            self.collate_fn = _utils.collate.default_convert
    
    @property
    def _index_sampler(self):
        if self.batch_sampler is not None:
            return self.batch_sampler
        else:
            return self.sampler
    
    def __iter__(self) -> Iterator[Any]:
        """Create an iterator over the dataset."""
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)
    
    def __len__(self) -> int:
        """Return the number of batches in the dataset."""
        if self.batch_sampler is not None:
            return len(self.batch_sampler)
        elif self.batch_size is not None:
            if self.drop_last:
                return len(self.sampler) // self.batch_size
            else:
                return (len(self.sampler) + self.batch_size - 1) // self.batch_size
        else:
            return len(self.sampler)
    
    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for the sampler.
        
        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        DataLoader will yield the same ordering.
        
        Args:
            epoch: Epoch number.
        """
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)


def get_worker_info() -> Optional[Any]:
    """Get information about the current data loading worker.
    
    Returns:
        WorkerInfo: An object containing information about the current worker.
            Returns ``None`` if the current process is not a data loading worker.
    """
    return _get_worker_info()
