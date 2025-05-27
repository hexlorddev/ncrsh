"""
DataLoader iterators for single and multi-process data loading.
"""
import os
import sys
import time
import traceback
import warnings
import threading
import itertools
import queue
import random
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, TypeVar, Union, overload, Generic

from ..tensor import Tensor
from .sampler import Sampler, SequentialSampler, RandomSampler, BatchSampler
from .dataset import Dataset, IterableDataset
from ._utils import (
    get_worker_info, WorkerInfo, default_collate, default_convert,
    _set_worker_signal_handlers, _disable_worker_signal_handling,
    _is_main_process, _is_fork, _set_SIGCHLD_handler
)

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

class _BaseDataLoaderIter(Generic[T_co]):
    """Base class for DataLoader iterator."""
    
    def __init__(self, loader: 'DataLoader') -> None:
        self._dataset = loader.dataset
        self._dataset_kind = loader._dataset_kind
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        self._prefetch_factor = loader.prefetch_factor
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._pin_memory = loader.pin_memory and torch.cuda.is_available()
        self._worker_init_fn = loader.worker_init_fn
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = torch.empty((), dtype=torch.int64).random_().item()
        self._persistent_workers = loader.persistent_workers
        
        # For single-process loading
        self._sampler = loader.sampler
        self._batch_sampler = loader.batch_sampler
        self._drop_last = loader.drop_last
        
        # For multi-process loading
        self._worker_queue_idx = 0
        self._worker_result_queue = None
        self._workers_done_event = None
        self._workers = []
        self._worker_pids_set = False
        self._shutdown = False
        
        self._send_idx = 0
        self._rcvd_idx = 0
        self._task_info = {}
        self._tasks_outstanding = 0
        self._data_queue = None
        
        # We need to ensure that the worker processes are properly terminated
        # when the main process exits.
        self._finalizer = None
    
    def __iter__(self) -> '_BaseDataLoaderIter[T_co]':
        return self
    
    def _next_data(self):
        raise NotImplementedError
    
    def __next__(self) -> T_co:
        data = self._next_data()
        self._rcvd_idx += 1
        return data
    
    def __len__(self) -> int:
        return len(self._index_sampler)
    
    def _get_data(self):
        # Get the next batch of indices (and possibly data)
        if self._timeout > 0:
            success, data = self._try_get_data()
            if not success:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self._timeout))
            return data
        else:
            return self._get_data_no_timeout()
    
    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        # Try to get data with a timeout
        try:
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
        except queue.Empty:
            return (False, None)
    
    def _get_data_no_timeout(self):
        # Get data without a timeout
        while True:
            try:
                return self._data_queue.get()
            except queue.Empty:
                time.sleep(0.1)
    
    def _process_data(self, data):
        # Process data from the worker
        self._tasks_outstanding -= 1
        if self._dataset_kind == _DatasetKind.Iterable:
            # For iterable-style datasets, the data is already processed
            return data
        else:
            # For map-style datasets, we need to fetch the data
            return self._get_data()
    
    def _try_put_index(self):
        # Try to put an index into the index queue
        assert self._tasks_outstanding < self._prefetch_factor * self._num_workers
        
        try:
            index = next(self._sampler_iter)
        except StopIteration:
            return False
            
        self._put_index(index)
        return True
    
    def _put_index(self, index):
        # Put an index into the index queue
        self._index_queues[self._worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (self._worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1
        self._worker_queue_idx = (self._worker_queue_idx + 1) % self._num_workers
    
    def _shutdown_workers(self):
        # Shut down the worker processes
        if not self._shutdown:
            self._shutdown = True
            
            # Signal the workers to exit
            if hasattr(self, '_workers'):
                for _ in range(self._num_workers):
                    if self._persistent_workers:
                        self._index_queues[0].put(None)
                    else:
                        self._index_queues[0].put(None)
                
                # Wait for workers to exit
                for w in self._workers:
                    w.join(timeout=self._timeout)
                
                # Clean up queues
                for q in self._index_queues:
                    q.cancel_join_thread()
                    q.close()
                
                if self._worker_result_queue is not None:
                    self._worker_result_queue.cancel_join_thread()
                    self._worker_result_queue.close()
    
    def __del__(self):
        self._shutdown_workers()


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter[T_co]):
    """Iterator for single-process data loading."""
    
    def __init__(self, loader: 'DataLoader') -> None:
        super().__init__(loader)
        
        assert self._num_workers == 0
        
        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._collate_fn, self._drop_last
        )
    
    def _next_data(self):
        index = self._next_index()
        data = self._dataset_fetcher.fetch(index)
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data
    
    def _next_index(self):
        return next(self._sampler_iter)


class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter[T_co]):
    """Iterator for multi-process data loading."""
    
    def __init__(self, loader: 'DataLoader') -> None:
        super().__init__(loader)
        
        assert self._num_workers > 0
        
        # Set up worker result queue
        self._worker_result_queue = queue.Queue()
        
        # Set up index queues
        self._index_queues = []
        for _ in range(self._num_workers):
            index_queue = queue.Queue()
            self._index_queues.append(index_queue)
        
        # Start worker processes
        self._workers = []
        for i in range(self._num_workers):
            worker = threading.Thread(
                target=_worker_loop,
                args=(
                    self._dataset_kind,
                    self._dataset,
                    self._index_queues[i],
                    self._worker_result_queue,
                    self._workers_done_event,
                    False,  # auto_collation
                    self._collate_fn,
                    self._drop_last,
                    self._base_seed + i,
                    self._worker_init_fn,
                    i,  # worker_id
                    self._num_workers,
                    self._persistent_workers,
                ),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)
        
        # Prime the prefetch loop
        for _ in range(2 * self._num_workers):
            self._try_put_index()
    
    def _next_data(self):
        while True:
            # Get the next batch of data
            success, data = self._try_get_data()
            
            if not success:
                # No data available, try to put more indices in the queue
                if not self._try_put_index():
                    # No more indices to process
                    if self._tasks_outstanding == 0:
                        raise StopIteration
            elif data is not None:
                # Got some data, process it
                return data
            
            # Wait a bit before trying again
            time.sleep(0.1)
    
    def _try_put_index(self):
        # Try to put an index into the index queue
        if self._tasks_outstanding >= 2 * self._num_workers:
            return False
            
        try:
            index = next(self._sampler_iter)
        except StopIteration:
            return False
            
        # Put the index in a round-robin fashion
        self._index_queues[self._worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (self._worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1
        self._worker_queue_idx = (self._worker_queue_idx + 1) % self._num_workers
        return True
    
    def _process_data(self, data):
        # Process data from the worker
        self._tasks_outstanding -= 1
        
        if isinstance(data, Exception):
            # Worker raised an exception, re-raise it
            raise data
            
        # Get the index and the actual data
        idx, batch = data
        
        # Clean up task info
        del self._task_info[idx]
        
        # Try to put more indices in the queue
        self._try_put_index()
        
        return batch
