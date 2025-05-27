"""
Utility Functions and Classes
---------------------------
This module provides various utility functions and classes for the ncrsh library.
"""

from .data_utils import (
    Dataset,
    ConcatDataset,
    Subset,
    split_dataset,
    download_file,
    extract_archive,
    worker_init_fn,
    default_collate,
    check_md5
)

from .checkpoint import (
    CheckpointManager,
    Logger,
    save_model,
    load_model
)

# For backward compatibility
__all__ = [
    # Data utilities
    'Dataset',
    'ConcatDataset',
    'Subset',
    'split_dataset',
    'download_file',
    'extract_archive',
    'worker_init_fn',
    'default_collate',
    'check_md5',
    
    # Checkpoint and logging
    'CheckpointManager',
    'Logger',
    'save_model',
    'load_model',
]

# Add version information
__version__ = '0.1.0'
