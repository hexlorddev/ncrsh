"""Collate functions for DataLoader."""
import collections
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, Sequence

import numpy as np

from ...tensor import Tensor

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

def default_convert(data: Any) -> Any:
    """Convert data to Tensor if it's a numpy array or a number."""
    if isinstance(data, (str, bytes)):
        return data
    elif isinstance(data, (np.ndarray, np.number)):
        return Tensor(data)
    elif isinstance(data, (int, float, bool)):
        return Tensor([data])
    elif isinstance(data, collections.abc.Mapping):
        return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return type(data)(*(default_convert(d) for d in data))
    elif isinstance(data, collections.abc.Sequence):
        return [default_convert(d) for d in data]
    else:
        return data

def default_collate(batch: List[T]) -> Any:
    """Collate a batch of data into a single batch.
    
    Args:
        batch: List of samples from the dataset.
        
    Returns:
        Collated batch of data.
    """
    elem = batch[0]
    elem_type = type(elem)
    
    if isinstance(elem, Tensor):
        out = None
        if len(batch) == 1:
            return batch[0].unsqueeze(0)
        return Tensor.stack(batch, 0)
    
    elif isinstance(elem, np.ndarray):
        return default_collate([Tensor(b) for b in batch])
    
    elif isinstance(elem, (int, float, bool)):
        return Tensor(batch)
    
    elif isinstance(elem, str):
        return batch
    
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    
    elif isinstance(elem, collections.abc.Sequence):
        # Check if we should try to stack the results
        if len(elem) == 0:
            return []
            
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    
    raise TypeError(f"default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {elem_type.__name__}")


def collate_tensor_fn(batch: List[Tensor], *, collate_fn_map: Dict[Any, Callable] = None) -> Tensor:
    """Collate function for Tensor objects."""
    return Tensor.stack(batch, 0)


def collate_numpy_array_fn(batch: List[np.ndarray], *, collate_fn_map: Dict[Any, Callable] = None) -> Tensor:
    """Collate function for numpy arrays."""
    return Tensor(np.stack(batch, 0))


def collate_numpy_scalar_fn(batch: List[Union[np.number, int, float]], *, collate_fn_map: Dict[Any, Callable] = None) -> Tensor:
    """Collate function for numpy scalars and Python numbers."""
    return Tensor(batch)


def collate_str_fn(batch: List[Union[str, bytes]], *, collate_fn_map: Dict[Any, Callable] = None) -> List[Union[str, bytes]]:
    """Collate function for strings and bytes."""
    return batch


def collate_dict_fn(batch: List[Dict[Any, Any]], *, collate_fn_map: Dict[Any, Callable] = None) -> Dict[Any, Any]:
    """Collate function for dictionaries."""
    if len(batch) == 0:
        return {}
    
    collated = {}
    for key in batch[0].keys():
        values = [d[key] for d in batch]
        collated[key] = default_collate(values)
    
    return collated


def collate_namedtuple_fn(batch: List[Any], *, collate_fn_map: Dict[Any, Callable] = None) -> Any:
    """Collate function for namedtuples."""
    return type(batch[0])(*[default_collate(samples) for samples in zip(*batch)])


# Default collation function map
DEFAULT_COLLATE_FN_MAP = {
    Tensor: collate_tensor_fn,
    np.ndarray: collate_numpy_array_fn,
    (int, float, bool, np.number): collate_numpy_scalar_fn,
    str: collate_str_fn,
    bytes: collate_str_fn,
    collections.abc.Mapping: collate_dict_fn,
}

# Register collation functions for numpy number types
for t in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,
          np.float16, np.float32, np.float64, np.bool_]:
    DEFAULT_COLLATE_FN_MAP[t] = collate_numpy_scalar_fn

# Register collation functions for Python built-in types
for t in [int, float, bool]:
    DEFAULT_COLLATE_FN_MAP[t] = collate_numpy_scalar_fn
