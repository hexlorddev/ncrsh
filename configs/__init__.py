"""
Configuration management for ncrsh.

This module provides utilities for managing configuration files and parameters.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class Config(dict):
    """Configuration class that allows dot notation access."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return super().__getattribute__(name)
    
    def __setattr__(self, name, value):
        self[name] = value

def load_config(config_path: str) -> Config:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Config: Configuration object with dot notation access
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f) or {}
    
    # Convert nested dictionaries to Config objects
    def dict_to_config(d):
        if isinstance(d, dict):
            return Config({k: dict_to_config(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_config(i) for i in d]
        else:
            return d
    
    return dict_to_config(config_dict)

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration with values to override
        
    Returns:
        Dict: Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if (key in result and isinstance(result[key], dict) 
                and isinstance(value, dict)):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration.
    
    Returns:
        Dict: Default configuration
    """
    return {
        'model': {
            'name': 'simple_cnn',
            'params': {
                'num_classes': 10,
                'in_channels': 3
            }
        },
        'data': {
            'batch_size': 32,
            'num_workers': 4,
            'pin_memory': True,
            'shuffle': True
        },
        'training': {
            'epochs': 10,
            'optimizer': {
                'name': 'adam',
                'params': {
                    'lr': 0.001,
                    'weight_decay': 0.0001
                }
            },
            'scheduler': {
                'name': 'step_lr',
                'params': {
                    'step_size': 30,
                    'gamma': 0.1
                }
            },
            'checkpoint': {
                'save_dir': 'checkpoints',
                'save_freq': 1
            }
        },
        'logging': {
            'log_dir': 'logs',
            'log_freq': 10,
            'use_tensorboard': True
        }
    }
