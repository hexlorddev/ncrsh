# Quick Start

This guide will help you quickly get started with ncrsh by walking through a simple example of training a model.

## Basic Usage

### 1. Import Required Modules

```python
import ncrsh
from ncrsh.data import DataLoader, Dataset
from ncrsh.nn import Linear, ReLU, Sequential, CrossEntropyLoss
from ncrsh.optim import SGD
```

### 2. Create a Simple Model

```python
model = Sequential(
    Linear(784, 128),  # Input layer
    ReLU(),
    Linear(128, 64),   # Hidden layer
    ReLU(),
    Linear(64, 10)     # Output layer
)
```

### 3. Prepare Your Data

```python
# Example with random data
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, size=1000):
        self.data = np.random.randn(size, 784).astype(np.float32)
        self.targets = np.random.randint(0, 10, size=size, dtype=np.int64)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Create datasets
train_dataset = CustomDataset(1000)
val_dataset = CustomDataset(200)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
```

### 4. Training Loop

```python
# Initialize loss function and optimizer
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    # Training
    model.train()
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    total = 0
    correct = 0
    with ncrsh.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')
```

## Next Steps

- Explore more advanced models in the [Examples](../examples/index.md) section
- Learn about custom datasets and data loading in the [Data Loading](../guide/data_loading.md) guide
- Check out the [API Reference](../api/index.md) for detailed documentation of all modules
