#!/usr/bin/env python3
"""
Data Preprocessing and Augmentation Script
----------------------------------------
This script provides utilities for preprocessing and augmenting data for deep learning.
"""
import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import random
from typing import List, Tuple, Dict, Any, Optional, Callable, Union

import ncrsh
from ncrsh.data import Dataset
from ncrsh.tensor import Tensor

class ImageTransform:
    """Base class for image transformations."""
    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class RandomHorizontalFlip(ImageTransform):
    """Randomly flip the image horizontally with a given probability."""
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            return np.fliplr(img).copy()
        return img

class RandomCrop(ImageTransform):
    """Crop the image at a random location."""
    def __init__(self, size: Union[int, Tuple[int, int]], padding: int = 0):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.padding > 0:
            img = np.pad(img, 
                        ((self.padding, self.padding), 
                         (self.padding, self.padding), 
                         (0, 0)) if len(img.shape) == 3 else 
                        ((self.padding, self.padding), 
                         (self.padding, self.padding)))
        
        h, w = img.shape[0], img.shape[1]
        th, tw = self.size
        
        if h == th and w == tw:
            return img
        
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        
        if len(img.shape) == 3:
            return img[i:i+th, j:j+tw, :]
        else:
            return img[i:i+th, j:j+tw]

class Normalize(ImageTransform):
    """Normalize image with mean and standard deviation."""
    def __init__(self, mean: List[float], std: List[float], inplace: bool = False):
        self.mean = np.array(mean).reshape(1, 1, -1)
        self.std = np.array(std).reshape(1, 1, -1)
        self.inplace = inplace
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not self.inplace:
            img = img.copy()
        
        # Handle both (H, W, C) and (C, H, W) formats
        if img.shape[0] == 3 or img.shape[0] == 1:  # (C, H, W)
            img = img.transpose(1, 2, 0)  # Convert to (H, W, C)
        
        img = (img - self.mean) / self.std
        
        return img

class ToTensor(ImageTransform):
    """Convert a PIL Image or numpy.ndarray to tensor."""
    def __call__(self, img: np.ndarray) -> Tensor:
        # If image is (H, W, C), transpose to (C, H, W)
        if len(img.shape) == 3 and img.shape[2] in [1, 3]:
            img = img.transpose(2, 0, 1)
        elif len(img.shape) == 2:  # Grayscale image
            img = img[np.newaxis, :, :]
        
        return Tensor(img.astype(np.float32))

class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, img: np.ndarray) -> Any:
        for t in self.transforms:
            img = t(img)
        return img

def load_image(path: str) -> np.ndarray:
    """Load an image from file."""
    img = Image.open(path)
    return np.array(img)

def save_image(img: np.ndarray, path: str):
    """Save an image to file."""
    if isinstance(img, Tensor):
        img = img.numpy()
    
    # Handle (C, H, W) format
    if len(img.shape) == 3 and img.shape[0] in [1, 3]:
        img = img.transpose(1, 2, 0)
    
    # Convert to uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    Image.fromarray(img).save(path)

def create_augmentation_pipeline(augment: bool = True) -> Compose:
    """Create a data augmentation pipeline."""
    if augment:
        return Compose([
            RandomHorizontalFlip(p=0.5),
            RandomCrop(32, padding=4),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

def preprocess_directory(
    input_dir: str,
    output_dir: str,
    transform: Optional[Callable] = None,
    extensions: Tuple[str] = ('.jpg', '.jpeg', '.png'),
    recursive: bool = True
) -> None:
    """
    Preprocess all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save preprocessed images
        transform: Optional transform to apply
        extensions: File extensions to include
        recursive: Whether to search subdirectories
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_paths = []
    if recursive:
        for ext in extensions:
            image_paths.extend(input_path.rglob(f'*{ext}'))
    else:
        for ext in extensions:
            image_paths.extend(input_path.glob(f'*{ext}'))
    
    print(f"Found {len(image_paths)} images in {input_dir}")
    
    # Process each image
    for img_path in image_paths:
        try:
            # Load image
            img = load_image(str(img_path))
            
            # Apply transform if provided
            if transform is not None:
                img = transform(img)
            
            # Create output path
            rel_path = img_path.relative_to(input_path)
            out_path = output_path / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save processed image
            save_image(img, str(out_path))
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description='Data Preprocessing and Augmentation')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing images')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for preprocessed images')
    parser.add_argument('--augment', action='store_true',
                        help='Apply data augmentation')
    parser.add_argument('--recursive', action='store_true',
                        help='Search for images recursively')
    parser.add_argument('--extensions', type=str, nargs='+', 
                        default=['.jpg', '.jpeg', '.png'],
                        help='Image file extensions to include')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create transform pipeline
    transform = create_augmentation_pipeline(augment=args.augment)
    
    # Preprocess directory
    preprocess_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        transform=transform,
        extensions=tuple(args.extensions),
        recursive=args.recursive
    )
    
    print(f"Preprocessing complete. Output saved to {args.output_dir}")

if __name__ == "__main__":
    main()
