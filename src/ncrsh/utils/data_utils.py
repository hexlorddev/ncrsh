"""
Data Loading and Preprocessing Utilities
---------------------------------------
This module provides utilities for data loading, preprocessing, and augmentation.
"""
import os
import random
import numpy as np
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Iterator
)

import ncrsh
from ncrsh.tensor import Tensor

# Type aliases
ArrayLike = Union[np.ndarray, Tensor]
PathLike = Union[str, os.PathLike]
Transform = Callable[[Any], Any]

class Dataset:
    """An abstract class representing a Dataset.
    
    All datasets that represent a map from keys to data samples should subclass it.
    All subclasses should override ``__getitem__``, supporting fetching a data sample
    for a given key. Subclasses could also optionally override ``__len__``, which is
    expected to return the size of the dataset.
    """
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        raise NotImplementedError
    
    def __len__(self) -> int:
        raise NotImplementedError
    
    def __add__(self, other: 'Dataset') -> 'ConcatDataset':
        return ConcatDataset([self, other])


class ConcatDataset(Dataset):
    """Dataset as a concatenation of multiple datasets.
    
    This class is useful to assemble different existing datasets.
    """
    
    datasets: List[Dataset]
    cumulative_sizes: List[int]
    
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r
    
    def __init__(self, datasets: Sequence[Dataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be empty'
        
        for d in self.datasets[1:]:
            assert isinstance(d, Dataset), 'ConcatDataset expects a Dataset object'
        
        self.cumulative_sizes = self.cumsum(self.datasets)
    
    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        
        dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        return self.datasets[dataset_idx][sample_idx]
    
    @property
    def cummulative_sizes(self):
        return self.cumulative_sizes


def split_dataset(
    dataset: Dataset,
    lengths: Sequence[Union[int, float]],
    random_seed: Optional[int] = None
) -> List[Dataset]:
    """Randomly split a dataset into non-overlapping new datasets of given lengths.
    
    Args:
        dataset: Dataset to be split
        lengths: Lengths of splits to be produced. If a float is provided, it will
            be treated as a fraction of the dataset length.
        random_seed: Random seed for reproducibility
        
    Returns:
        List of datasets
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Convert fractions to absolute lengths
    if all(isinstance(x, float) for x in lengths):
        assert sum(lengths) == 1.0, "Sum of input lengths does not equal 1.0"
        lengths = [int(len(dataset) * x) for x in lengths]
        # Handle rounding errors
        lengths[-1] = len(dataset) - sum(lengths[:-1])
    
    # Validate lengths
    assert sum(lengths) == len(dataset), "Sum of input lengths does not equal the length of the input dataset!"
    assert all(l > 0 for l in lengths), "All lengths must be positive!"
    
    # Generate random indices
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    # Split indices
    split_indices = []
    start = 0
    for l in lengths[:-1]:
        end = start + l
        split_indices.append(indices[start:end])
        start = end
    split_indices.append(indices[start:])
    
    # Create subset datasets
    return [Subset(dataset, idx) for idx in split_indices]


class Subset(Dataset):
    """Subset of a dataset at specified indices.
    
    Args:
        dataset: The dataset to create a subset from
        indices: Indices in the whole set selected for subset
    """
    
    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def __len__(self):
        return len(self.indices)


def download_file(
    url: str,
    root: Union[str, Path],
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    max_redirect_hops: int = 3
) -> Path:
    """Download a file from a URL and place it in root.
    
    Args:
        url: URL to download file from
        root: Directory to place downloaded file in
        filename: Name to save the file under. If None, use the filename from the URL.
        md5: MD5 checksum of the download. If None, do not check.
        max_redirect_hops: Maximum number of redirects to follow
        
    Returns:
        Path to the downloaded file
    """
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    
    if not filename:
        filename = os.path.basename(url)
    
    fpath = root / filename
    
    # Check if file already exists and verify checksum
    if fpath.exists():
        if md5 is not None:
            if check_md5(fpath, md5):
                return fpath
            print(f"MD5 check failed for {fpath}. Redownloading...")
        else:
            return fpath
    
    # Expand redirect chains if needed
    for _ in range(max_redirect_hops + 1):
        try:
            import urllib.request
            import urllib.parse
            import urllib.error
            
            req = urllib.request.Request(url, headers={"User-Agent": "ncrsh"})
            with urllib.request.urlopen(req) as response, open(fpath, 'wb') as f:
                f.write(response.read())
            
            # Verify checksum if provided
            if md5 is not None and not check_md5(fpath, md5):
                raise RuntimeError(f"MD5 check failed for {url}")
                
            return fpath
            
        except urllib.error.HTTPError as e:
            if e.code == 302:  # Redirect
                url = e.headers.get('Location')
                if not url:
                    raise RuntimeError("Redirect missing Location header")
                continue
            raise
    
    raise RuntimeError(f"Too many redirects for {url}")

def check_md5(fpath: PathLike, md5: str) -> bool:
    """Check MD5 checksum of a file."""
    import hashlib
    
    hash_md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() == md5


def extract_archive(
    from_path: PathLike,
    to_path: Optional[PathLike] = None,
    remove_finished: bool = False
) -> Path:
    """Extract an archive.
    
    Args:
        from_path: Path to the archive
        to_path: Directory to extract to. If None, extracts to the parent directory of from_path.
        remove_finished: If True, remove the archive after extraction.
        
    Returns:
        Path to the extracted directory
    """
    from_path = Path(from_path)
    if to_path is None:
        to_path = from_path.parent
    else:
        to_path = Path(to_path)
    
    to_path.mkdir(parents=True, exist_ok=True)
    
    suffix = from_path.suffix.lower()
    
    if suffix == '.zip':
        import zipfile
        with zipfile.ZipFile(from_path, 'r') as zip_ref:
            zip_ref.extractall(to_path)
        
    elif suffix in ['.tar', '.gz', '.bz2']:
        import tarfile
        mode = 'r:'
        if suffix == '.gz':
            mode += 'gz'
        elif suffix == '.bz2':
            mode += 'bz2'
            
        with tarfile.open(from_path, mode) as tar_ref:
            tar_ref.extractall(to_path)
    else:
        raise ValueError(f"Unsupported archive format: {suffix}")
    
    if remove_finished:
        from_path.unlink()
    
    return to_path


def worker_init_fn(worker_id: int) -> None:
    """Worker init function to ensure different random seeds for each worker."""
    worker_seed = ncrsh.initial_seed() + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def default_collate(batch):
    """Collate function that converts numpy arrays to tensors."""
    import collections
    import torch
    
    elem = batch[0]
    elem_type = type(elem)
    
    if isinstance(elem, ncrsh.Tensor):
        return ncrsh.stack(batch, 0)
    elif isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(elem, np.ndarray):
        return ncrsh.tensor(np.stack(batch, 0))
    elif isinstance(elem, (int, float)):
        return ncrsh.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # Check if the elements are tensors and can be stacked
        if all(isinstance(x, (ncrsh.Tensor, torch.Tensor, np.ndarray)) for x in elem):
            return [default_collate([b[i] for b in batch]) for i in range(len(elem))]
        return [default_collate([b[i] for b in batch]) for i in range(len(elem))]
    
    raise TypeError(f"default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {elem_type}")
