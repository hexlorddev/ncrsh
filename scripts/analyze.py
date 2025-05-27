#!/usr/bin/env python3
"""
Model Analysis and Visualization Script
--------------------------------------
This script provides utilities for analyzing and visualizing models and training metrics.
"""
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import ncrsh
from ncrsh.nn import Module

def plot_training_curves(
    log_dir: str,
    metrics: List[str] = ['loss', 'accuracy'],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot training and validation metrics from log files.
    
    Args:
        log_dir: Directory containing training logs
        metrics: List of metrics to plot (e.g., ['loss', 'accuracy'])
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    log_dir = Path(log_dir)
    
    # Find all log files
    log_files = list(log_dir.glob('*.json'))
    if not log_files:
        print(f"No JSON log files found in {log_dir}")
        return
    
    # Load and process logs
    logs = []
    for log_file in log_files:
        with open(log_file, 'r') as f:
            try:
                logs.append(json.load(f))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {log_file}")
    
    if not logs:
        print("No valid log files found.")
        return
    
    # Plot each metric
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4*num_metrics))
    if num_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for log in logs:
            if 'train_' + metric in log and 'val_' + metric in log:
                epochs = range(1, len(log['train_' + metric]) + 1)
                ax.plot(epochs, log['train_' + metric], 'b-', label='Train', alpha=0.7)
                ax.plot(epochs, log['val_' + metric], 'r-', label='Validation', alpha=0.7)
        
        ax.set_title(f'Training and Validation {metric.capitalize()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()

def count_parameters(model: Module) -> Dict[str, int]:
    """
    Count the number of trainable and total parameters in a model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dict with 'total_params' and 'trainable_params' counts
    """
    total_params = 0
    trainable_params = 0
    
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }

def get_model_summary(
    model: Module, 
    input_size: Tuple[int, ...],
    device: str = 'cpu',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate a summary of the model architecture.
    
    Args:
        model: The model to analyze
        input_size: Input tensor size (batch dimension is not needed)
        device: Device to run the model on
        verbose: Whether to print the summary
        
    Returns:
        Dict containing model summary
    """
    model = model.to(device)
    model.eval()
    
    # Create a dummy input tensor
    x = ncrsh.randn(1, *input_size, device=device)
    
    # Forward pass to get layer information
    layers = []
    hooks = []
    
    def hook(module, input, output):
        # Get layer information
        layer_type = module.__class__.__name__
        
        # Get input and output shapes
        if isinstance(input, (list, tuple)):
            input_shape = [tuple(i.shape) for i in input if i is not None]
        else:
            input_shape = tuple(input.shape) if input is not None else None
        
        if isinstance(output, (list, tuple)):
            output_shape = [tuple(o.shape) for o in output if o is not None]
        else:
            output_shape = tuple(output.shape) if output is not None else None
        
        # Count parameters
        params = 0
        for p in module.parameters():
            params += p.numel()
        
        layers.append({
            'name': module.__class__.__name__,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'params': params,
            'trainable': any(p.requires_grad for p in module.parameters())
        })
    
    # Register hooks on all modules
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            hooks.append(module.register_forward_hook(hook))
    
    # Run forward pass
    with ncrsh.no_grad():
        _ = model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate total parameters
    param_counts = count_parameters(model)
    
    # Create summary
    summary = {
        'layers': layers,
        'total_params': param_counts['total_params'],
        'trainable_params': param_counts['trainable_params'],
        'non_trainable_params': param_counts['non_trainable_params'],
        'input_size': input_size,
        'output_size': layers[-1]['output_shape'] if layers else None
    }
    
    # Print summary if verbose
    if verbose:
        print(f"Model: {model.__class__.__name__}")
        print(f"Input size: {input_size}")
        print(f"Total params: {summary['total_params']:,}")
        print(f"Trainable params: {summary['trainable_params']:,}")
        print(f"Non-trainable params: {summary['non_trainable_params']:,}")
        print("\nLayer (type)         Output Shape         Param #")
        print("=" * 50)
        
        for i, layer in enumerate(summary['layers']):
            print(f"{i:2d} {layer['name']:20} {str(layer['output_shape']):20} {layer['params']:,}")
        
        print("=" * 50)
        print(f"Total params: {summary['total_params']:,}")
        print(f"Trainable params: {summary['trainable_params']:,}")
        print(f"Non-trainable params: {summary['non_trainable_params']:,}")
    
    return summary

def parse_args():
    parser = argparse.ArgumentParser(description='Model Analysis and Visualization')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Plot training curves
    plot_parser = subparsers.add_parser('plot', help='Plot training curves')
    plot_parser.add_argument('--log-dir', type=str, required=True,
                            help='Directory containing training logs')
    plot_parser.add_argument('--metrics', type=str, nargs='+', 
                            default=['loss', 'accuracy'],
                            help='Metrics to plot')
    plot_parser.add_argument('--save-path', type=str, default=None,
                            help='Path to save the plot')
    
    # Model summary
    summary_parser = subparsers.add_parser('summary', help='Show model summary')
    summary_parser.add_argument('--model', type=str, required=True,
                               help='Path to model file')
    summary_parser.add_argument('--input-size', type=int, nargs='+', required=True,
                               help='Input size (e.g., 3 224 224 for images)')
    summary_parser.add_argument('--device', type=str, default='cpu',
                               help='Device to run the model on')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.command == 'plot':
        plot_training_curves(
            log_dir=args.log_dir,
            metrics=args.metrics,
            save_path=args.save_path
        )
    elif args.command == 'summary':
        # Load model
        if not os.path.exists(args.model):
            print(f"Error: Model file {args.model} not found.")
            return
        
        # This is a placeholder - in practice, you'd load your actual model
        print(f"Loading model from {args.model}")
        print("Note: Model loading not implemented in this example.")
        print("Please implement model loading based on your model architecture.")
        
        # Example of how you might use get_model_summary with a real model:
        # model = YourModel()
        # model.load_state_dict(ncrsh.load(args.model))
        # get_model_summary(model, tuple(args.input_size), device=args.device)
    else:
        print("Please specify a valid command. Use --help for more information.")

if __name__ == "__main__":
    main()
