#!/usr/bin/env python3
"""
Training script for ncrsh models.

This script provides a flexible training pipeline that can be configured
via command-line arguments or a YAML configuration file.
"""
import argparse
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a model with ncrsh')
    
    # Required arguments
    parser.add_argument('--config', type=str, default='../configs/default.yaml',
                      help='Path to config file (default: ../configs/default.yaml)')
    parser.add_argument('--output-dir', type=str, default='../output',
                      help='Directory to save outputs (default: ../output)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=None,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=None,
                      help='Learning rate')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='../data',
                      help='Path to dataset directory')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='simple_cnn',
                      help='Model architecture to use')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use for training (cpu, cuda, cuda:0, etc.)')
    parser.add_argument('--num-workers', type=int, default=None,
                      help='Number of workers for data loading')
    
    # Logging arguments
    parser.add_argument('--log-interval', type=int, default=10,
                      help='How many batches to wait before logging training status')
    parser.add_argument('--save-interval', type=int, default=1,
                      help='How many epochs to wait before saving a checkpoint')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_output_dir(output_dir):
    """Create output directory and return its path."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(output_dir) / f'run_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    (run_dir / 'logs').mkdir(exist_ok=True)
    
    return run_dir

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)
    
    # Override config with command line arguments
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.device is not None:
        config['device'] = args.device
    if args.num_workers is not None:
        config['data']['num_workers'] = args.num_workers
    
    # Setup output directory
    output_dir = setup_output_dir(args.output_dir)
    
    # Save the config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Configuration saved to {output_dir / 'config.yaml'}")
    print("Starting training with the following configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # TODO: Add actual training loop
    print("\nTraining would start here with the above configuration.")
    print(f"Checkpoints and logs will be saved to: {output_dir}")

if __name__ == "__main__":
    main()
