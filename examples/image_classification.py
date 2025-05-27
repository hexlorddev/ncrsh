"""
Image Classification Example
---------------------------
This example demonstrates how to train a simple CNN for image classification
using the ncrsh library.
"""
import os
import numpy as np
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import ncrsh
from ncrsh.data import DataLoader, Dataset
from ncrsh.nn import (
    Module, Sequential, Conv2d, MaxPool2d, 
    Linear, ReLU, Flatten, CrossEntropyLoss
)
from ncrsh.optim import Adam
from ncrsh.tensor import Tensor

# Set random seed for reproducibility
ncrsh.manual_seed(42)

class SimpleCNN(Module):
    """A simple CNN for image classification."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = Sequential(
            # Input: 3x32x32
            Conv2d(3, 16, kernel_size=3, padding=1),  # 16x32x32
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),  # 16x16x16
            
            Conv2d(16, 32, kernel_size=3, padding=1),  # 32x16x16
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),  # 32x8x8
            
            Conv2d(32, 64, kernel_size=3, padding=1),  # 64x8x8
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),  # 64x4x4
            
            Flatten()  # 1024
        )
        
        self.classifier = Sequential(
            Linear(1024, 512),
            ReLU(),
            Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CIFAR10Dataset(Dataset):
    """A simple CIFAR-10 dataset for demonstration."""
    def __init__(self, num_samples=1000, train=True):
        # In a real scenario, you would load CIFAR-10 data here
        # This is a simplified version with random data
        self.data = np.random.randn(num_samples, 3, 32, 32).astype(np.float32)
        self.targets = np.random.randint(0, 10, size=num_samples, dtype=np.int64)
        self.train = train
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return Tensor(self.data[idx]), Tensor([self.targets[idx]])

def train_one_epoch(model, dataloader, criterion, optimizer, device=None):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if device:
            inputs = inputs.to(device)
            targets = targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device=None):
    """Evaluate the model on the validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with ncrsh.no_grad():
        for inputs, targets in dataloader:
            if device:
                inputs = inputs.to(device)
                targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def main():
    # Hyperparameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    num_workers = 4
    
    # Check for CUDA
    device = ncrsh.device('cuda' if ncrsh.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = CIFAR10Dataset(num_samples=5000, train=True)
    val_dataset = CIFAR10Dataset(num_samples=1000, train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initialize model, loss, and optimizer
    model = SimpleCNN(num_classes=10).to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Model architecture:\n{model}")
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    print("\nTraining complete!")
    
    # Save the trained model
    os.makedirs("checkpoints", exist_ok=True)
    model_path = "checkpoints/cifar10_cnn.pth"
    ncrsh.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
