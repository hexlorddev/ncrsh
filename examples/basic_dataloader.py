"""
Basic example demonstrating how to use the ncrsh DataLoader with a custom dataset.
"""
import os
import numpy as np
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from ncrsh.data import DataLoader, Dataset
from ncrsh.tensor import Tensor

class CustomDataset(Dataset):
    """A simple custom dataset for demonstration purposes."""
    
    def __init__(self, size=100, input_shape=(3, 32, 32), num_classes=10):
        """
        Args:
            size: Number of samples in the dataset
            input_shape: Shape of each input sample (channels, height, width)
            num_classes: Number of output classes
        """
        self.size = size
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Generate random data for demonstration
        self.data = np.random.randn(size, *input_shape).astype(np.float32)
        self.targets = np.random.randint(0, num_classes, size=size, dtype=np.int64)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            tuple: (input_tensor, target_tensor)
        """
        return Tensor(self.data[idx]), Tensor([self.targets[idx]])

def main():
    # Create a dataset
    dataset = CustomDataset(size=1000, input_shape=(3, 32, 32), num_classes=10)
    
    # Create a DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Iterate through the DataLoader
    print("Starting training loop...")
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # In a real training loop, you would:
        # 1. Move data to GPU (if available)
        # 2. Forward pass
        # 3. Compute loss
        # 4. Backward pass and optimize
        
        # For demonstration, just print some info
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Input shape: {inputs.shape}, Target shape: {targets.shape}")
    
    print("Training complete!")

if __name__ == "__main__":
    main()
