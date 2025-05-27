"""Worker utilities for DataLoader."""
import os
import signal
import sys
import threading
import warnings
from dataclasses import dataclass
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np

from ...tensor import Tensor

__all__ = ['get_worker_info', 'WorkerInfo']

T_co = TypeVar('T_co', covariant=True)

@dataclass(frozen=True)
class WorkerInfo:
    """Information about the current worker process."""
    id: int
    num_workers: int
    seed: int
    dataset: Any
    
    def __repr__(self) -> str:
        return f"WorkerInfo(id={self.id}, num_workers={self.num_workers}, seed={self.seed})"

# Global worker info variable
_worker_info_globals = threading.local()

def get_worker_info() -> Optional[WorkerInfo]:
    """Get information about the current data loading worker."""
    return getattr(_worker_info_globals, 'worker_info', None)

def _worker_loop(dataset_kind, dataset, index_queue, data_queue, done_event, 
                auto_collation, collate_fn, drop_last, base_seed, init_fn, worker_id, 
                num_workers, persistent_workers):
    """Main worker loop for data loading."""
    # Set up worker info
    _worker_info = WorkerInfo(
        id=worker_id,
        num_workers=num_workers,
        seed=base_seed + worker_id,
        dataset=dataset
    )
    _worker_info_globals.worker_info = _worker_info
    
    # Set random seed for this worker
    np.random.seed(base_seed + worker_id)
    
    # Initialize worker if needed
    if init_fn is not None:
        init_fn(worker_id)
    
    # Main loop
    while not done_event.is_set():
        try:
            # Get index(es) to process
            r = index_queue.get(timeout=0.1)
        except queue.Empty:
            continue
            
        if r is None:  # Sentinel value to exit
            break
            
        idx, batch_indices = r
        
        try:
            # Fetch data
            if batch_indices is None:
                data = dataset[idx]
            else:
                data = [dataset[i] for i in batch_indices]
                if collate_fn is not None:
                    data = collate_fn(data)
            
            # Put result in output queue
            data_queue.put((idx, data))
            del data, idx, batch_indices
            
        except Exception as e:
            # Propagate exception to main process
            data_queue.put((idx, e))
            break
    
    # Clean up
    if persistent_workers:
        data_queue.put(None)  # Signal that this worker is done


def _worker_manager_loop(in_queue, out_queue, done_event, pin_memory, device_id, done_event_set_done=True):
    """Manager loop for worker processes."""
    # Set up CUDA if needed
    if pin_memory and device_id is not None:
        import torch.cuda
        torch.cuda.set_device(device_id)
    
    # Process data from workers
    while not done_event.is_set():
        try:
            r = in_queue.get(timeout=0.1)
        except queue.Empty:
            continue
            
        if r is None:  # Sentinel value to exit
            break
            
        idx, data = r
        
        # Handle errors
        if isinstance(data, Exception):
            out_queue.put((idx, data))
            break
            
        # Pin memory if needed
        if pin_memory:
            data = _pin_memory(data)
            
        # Send to output queue
        out_queue.put((idx, data))
    
    # Signal that we're done
    if done_event_set_done:
        done_event.set()


def _pin_memory(data):
    """Recursively pin tensors in nested data structures."""
    if isinstance(data, Tensor):
        return data.pin_memory()
    elif isinstance(data, (str, bytes)):
        return data
    elif isinstance(data, (tuple, list)):
        return type(data)(_pin_memory(x) for x in data)
    elif isinstance(data, dict):
        return {k: _pin_memory(v) for k, v in data.items()}
    else:
        return data
