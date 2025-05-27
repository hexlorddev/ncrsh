# Neural Network Module

This module contains the core neural network components for the ncrsh library.

## Core Components

### Layers
- **Linear**: Fully connected layer
- **Conv1d/2d/3d**: Convolutional layers
- **RNN/LSTM/GRU**: Recurrent layers
- **BatchNorm/LayerNorm**: Normalization layers
- **Dropout**: Regularization layer

### Loss Functions
- **MSELoss**: Mean squared error
- **CrossEntropyLoss**: For classification tasks
- **BCELoss**: Binary cross-entropy
- **L1Loss/SmoothL1Loss**: Regression losses

### Utilities
- **Sequential**: Sequential container
- **ModuleList/ModuleDict**: Containers for modules
- **Parameter/ParameterList**: For model parameters

## Usage Example

```python
import ncrsh
from ncrsh.nn import Sequential, Linear, ReLU, CrossEntropyLoss

# Define a simple model
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

# Define loss function
criterion = CrossEntropyLoss()

# Forward pass
output = model(input_data)
loss = criterion(output, target)

# Backward pass
loss.backward()
```

## Adding New Layers

1. Create a new class inheriting from `nn.Module`
2. Initialize parameters in `__init__`
3. Implement the forward pass in `forward`
4. Add documentation and type hints
5. Add unit tests in `tests/nn/test_layers.py`

## Best Practices

- Use `nn.Parameter` for learnable parameters
- Register buffers for non-trainable parameters
- Implement `extra_repr` for better string representation
- Add type hints for better IDE support
- Document expected input/output shapes
