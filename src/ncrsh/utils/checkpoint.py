"""
Model Checkpointing and Logging Utilities
-----------------------------------------
This module provides utilities for saving and loading model checkpoints,
as well as logging training metrics.
"""
import os
import json
import time
import shutil
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar, Generic, Type, cast
)

import numpy as np

import ncrsh
from ncrsh.nn import Module
from ncrsh.optim import Optimizer

PathLike = Union[str, os.PathLike]
T = TypeVar('T')

class CheckpointManager:
    """Manages saving and loading model checkpoints.
    
    This class handles:
    - Saving and loading model state dictionaries
    - Tracking the best model based on a metric
    - Managing multiple checkpoints
    - Saving and loading optimizer state
    - Saving and loading training metadata
    """
    
    def __init__(
        self,
        model: Module,
        optimizer: Optional[Optimizer] = None,
        checkpoint_dir: PathLike = 'checkpoints',
        max_to_keep: int = 5,
        save_best_only: bool = False,
        metric_name: str = 'val_loss',
        mode: str = 'min',
        save_freq: int = 1,
        **metadata
    ) -> None:
        """Initialize the checkpoint manager.
        
        Args:
            model: The model to save/load
            optimizer: The optimizer to save/load (optional)
            checkpoint_dir: Directory to save checkpoints
            max_to_keep: Maximum number of checkpoints to keep
            save_best_only: If True, only save when the monitored metric improves
            metric_name: Name of the metric to monitor
            mode: One of 'min' or 'max' for the monitored metric
            save_freq: Save frequency in epochs
            **metadata: Additional metadata to store with checkpoints
        """
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_to_keep = max_to_keep
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        self.save_freq = save_freq
        self.metadata = metadata
        
        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        self.mode = mode
        
        # Track best metric value
        self.best_metric = float('inf') if mode == 'min' else -float('inf')
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing checkpoints
        self.checkpoints = self._discover_checkpoints()
    
    def _discover_checkpoints(self) -> List[Path]:
        """Discover existing checkpoints in the checkpoint directory."""
        checkpoints = []
        
        for f in self.checkpoint_dir.glob('checkpoint_*.pth'):
            try:
                # Extract epoch number from filename
                epoch = int(f.stem.split('_')[1])
                checkpoints.append((epoch, f))
            except (IndexError, ValueError):
                continue
        
        # Sort by epoch number
        checkpoints.sort()
        return [f for _, f in checkpoints]
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if we've exceeded max_to_keep."""
        if len(self.checkpoints) > self.max_to_keep:
            # Keep the most recent checkpoints
            for f in self.checkpoints[:-self.max_to_keep]:
                try:
                    f.unlink()
                except OSError:
                    pass
            
            # Update checkpoints list
            self.checkpoints = self._discover_checkpoints()
    
    def save(
        self,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        **kwargs
    ) -> Path:
        """Save a checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics to save
            is_best: Whether this is the best model so far
            **kwargs: Additional data to save
            
        Returns:
            Path to the saved checkpoint
        """
        # Skip saving if not at save frequency
        if (epoch + 1) % self.save_freq != 0 and not is_best:
            return None
        
        # Update best metric
        if metrics and self.metric_name in metrics:
            current_metric = metrics[self.metric_name]
            if (self.mode == 'min' and current_metric < self.best_metric) or \
               (self.mode == 'max' and current_metric > self.best_metric):
                self.best_metric = current_metric
                is_best = True
        
        # Skip saving if not best and save_best_only is True
        if self.save_best_only and not is_best:
            return None
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'metrics': metrics or {},
            'metadata': {
                **self.metadata,
                'best_metric': self.best_metric,
                'metric_name': self.metric_name,
                'mode': self.mode,
                'timestamp': time.time()
            },
            **kwargs
        }
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_{epoch:04d}.pth'
        ncrsh.save(checkpoint, checkpoint_path)
        
        # Update checkpoints list
        self.checkpoints.append(checkpoint_path)
        
        # Save a copy if this is the best model
        if is_best:
            best_path = self.checkpoint_dir / 'model_best.pth'
            shutil.copyfile(checkpoint_path, best_path)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load(
        self,
        checkpoint_path: Optional[PathLike] = None,
        load_best: bool = False,
        strict: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file. If None, loads the latest checkpoint.
            load_best: If True, load the best checkpoint instead of the latest.
            strict: Whether to strictly enforce that the keys in checkpoint match the model.
            **kwargs: Additional arguments to pass to model.load_state_dict()
            
        Returns:
            Dictionary containing the checkpoint data
        """
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.checkpoint_dir / 'model_best.pth'
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"No best model found at {checkpoint_path}")
            else:
                if not self.checkpoints:
                    raise FileNotFoundError("No checkpoints found")
                checkpoint_path = self.checkpoints[-1]
        
        checkpoint = ncrsh.load(checkpoint_path)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict, **kwargs)
        
        # Load optimizer state if available
        if self.optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Update best metric
        if 'metadata' in checkpoint and 'best_metric' in checkpoint['metadata']:
            self.best_metric = checkpoint['metadata']['best_metric']
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the path to the latest checkpoint."""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get the path to the best checkpoint."""
        best_path = self.checkpoint_dir / 'model_best.pth'
        return best_path if best_path.exists() else None


class Logger:
    """Simple logger for training metrics."""
    
    def __init__(self, log_dir: PathLike = 'logs'):
        """Initialize the logger.
        
        Args:
            log_dir: Directory to save log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        self.log_file = self.log_dir / f'training_{timestamp}.json'
        self.log_data = []
    
    def log(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics to log
        """
        log_entry = {
            'epoch': epoch,
            'timestamp': time.time(),
            **metrics
        }
        
        # Add to log data
        self.log_data.append(log_entry)
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all log entries."""
        return self.log_data
    
    def plot_metrics(self, save_path: Optional[PathLike] = None, show: bool = True) -> None:
        """Plot logged metrics.
        
        Args:
            save_path: Path to save the plot (optional)
            show: Whether to display the plot
        """
        if not self.log_data:
            print("No log data to plot")
            return
        
        import matplotlib.pyplot as plt
        
        # Get all metric names (excluding epoch and timestamp)
        metric_names = [k for k in self.log_data[0].keys() if k not in ['epoch', 'timestamp']]
        
        if not metric_names:
            print("No metrics to plot")
            return
        
        # Prepare data for plotting
        epochs = [entry['epoch'] for entry in self.log_data]
        metrics = {name: [entry.get(name, None) for entry in self.log_data] 
                  for name in metric_names}
        
        # Create figure and axes
        num_metrics = len(metric_names)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 3 * num_metrics))
        
        if num_metrics == 1:
            axes = [axes]
        
        # Plot each metric
        for i, name in enumerate(metric_names):
            ax = axes[i]
            ax.plot(epochs, metrics[name], 'b-')
            ax.set_title(name.replace('_', ' ').title())
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()


def save_model(
    model: Module,
    path: PathLike,
    optimizer: Optional[Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    **kwargs
) -> None:
    """Save a model checkpoint.
    
    Args:
        model: Model to save
        path: Path to save the checkpoint
        optimizer: Optimizer to save (optional)
        epoch: Current epoch number (optional)
        metrics: Dictionary of metrics to save (optional)
        **kwargs: Additional data to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'epoch': epoch,
        'metrics': metrics or {},
        'timestamp': time.time(),
        **kwargs
    }
    
    # Ensure directory exists
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    ncrsh.save(checkpoint, path)


def load_model(
    model: Module,
    path: PathLike,
    optimizer: Optional[Optimizer] = None,
    strict: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Load a model checkpoint.
    
    Args:
        model: Model to load the weights into
        path: Path to the checkpoint file
        optimizer: Optimizer to load state into (optional)
        strict: Whether to strictly enforce that the keys in checkpoint match the model
        **kwargs: Additional arguments to pass to model.load_state_dict()
        
    Returns:
        Dictionary containing the checkpoint data
    """
    checkpoint = ncrsh.load(path)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict, **kwargs)
    
    # Load optimizer state if available
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint
