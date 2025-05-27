"""
Tests for the ncrsh.nn module.
"""
import pytest
import numpy as np

from ncrsh.tensor import Tensor
from ncrsh.nn import Module, Linear, ReLU, Sequential

class TestModule(Module):
    """A simple test module for testing the base Module class."""
    
    def __init__(self):
        super().__init__()
        self.linear = Linear(10, 5)
        self.activation = ReLU()
    
    def forward(self, x):
        return self.activation(self.linear(x))

def test_module_initialization():
    """Test module initialization and parameter registration."""
    model = TestModule()
    
    # Check that parameters are registered
    params = list(model.parameters())
    assert len(params) == 2  # weight and bias
    assert params[0].shape == (5, 10)  # weight
    assert params[1].shape == (5,)  # bias

def test_linear_layer():
    """Test the linear layer."""
    layer = Linear(3, 2)
    x = Tensor(np.ones((1, 3)))
    
    # Forward pass
    output = layer(x)
    assert output.shape == (1, 2)
    
    # Check that gradients will flow
    output.sum().backward()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None

def test_sequential():
    """Test the sequential container."""
    model = Sequential(
        Linear(10, 5),
        ReLU(),
        Linear(5, 2)
    )
    
    x = Tensor(np.ones((1, 10)))
    output = model(x)
    
    assert output.shape == (1, 2)
    assert not np.allclose(output.numpy(), np.zeros_like(output.numpy()))

def test_training_mode():
    """Test switching between training and evaluation modes."""
    model = TestModule()
    
    # Default should be training mode
    assert model.training
    
    # Switch to evaluation mode
    model.eval()
    assert not model.training
    
    # Switch back to training mode
    model.train()
    assert model.training
