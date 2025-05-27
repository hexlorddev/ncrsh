# Optimization Module

This module implements various optimization algorithms for training neural networks.

## Available Optimizers

- **SGD**: Stochastic Gradient Descent with momentum and Nesterov acceleration
- **Adam**: Adaptive Moment Estimation
- **AdamW**: Adam with decoupled weight decay
- **RMSprop**: Root Mean Square Propagation
- **Adagrad**: Adaptive Gradient Algorithm
- **Adadelta**: An adaptive learning rate method

## Usage Example

```python
import ncrsh
from ncrsh.nn import Linear, MSELoss
from ncrsh.optim import Adam

# Define model and loss
model = Linear(10, 1)
criterion = MSELoss()

# Initialize optimizer
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Learning Rate Scheduling

```python
from ncrsh.optim import StepLR, MultiStepLR, ReduceLROnPlateau

# Step learning rate decay
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Multi-step learning rate decay
scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

# Reduce learning rate on plateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# In training loop
for epoch in range(num_epochs):
    # Train for one epoch
    train(...)
    
    # Step the scheduler
    scheduler.step()  # or scheduler.step(metrics) for ReduceLROnPlateau
```

## Adding a New Optimizer

1. Create a new class inheriting from `Optimizer`
2. Implement the `step()` method
3. Add parameter update logic
4. Include proper documentation and type hints
5. Add unit tests in `tests/test_optim.py`

## Best Practices

- Use `param_group` for different parameter groups
- Implement state management for optimization statistics
- Include support for sparse gradients where applicable
- Add proper error checking for hyperparameters
- Document the mathematical formulation in the class docstring
