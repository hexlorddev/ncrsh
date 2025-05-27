"""
Tests for optimization module.
"""
import numpy as np
import pytest

import ncrsh
from ncrsh.nn import Linear, MSELoss, Sequential
from ncrsh.optim import SGD, Adam, AdamW, RMSprop

class TestOptimizers:
    @pytest.mark.parametrize("optim_class", [SGD, Adam, AdamW, RMSprop])
    def test_optimizer_step(self, optim_class):
        """Test that optimizers can perform a training step."""
        # Create a simple model
        model = Sequential(
            Linear(10, 20),
            Linear(20, 1)
        )
        
        # Create dummy data
        x = ncrsh.randn(5, 10)
        y = ncrsh.randn(5, 1)
        
        # Initialize loss and optimizer
        criterion = MSELoss()
        optimizer = optim_class(model.parameters(), lr=0.01)
        
        # Initial loss
        outputs = model(x)
        initial_loss = criterion(outputs, y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        # Check that loss changed
        outputs = model(x)
        new_loss = criterion(outputs, y)
        assert not ncrsh.allclose(initial_loss, new_loss)
    
    def test_sgd_momentum(self):
        """Test SGD with momentum."""
        model = Linear(10, 1)
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        # First step
        x = ncrsh.randn(5, 10)
        y = ncrsh.randn(5, 1)
        
        def compute_loss():
            return ((model(x) - y) ** 2).mean()
        
        # First step
        optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        optimizer.step()
        
        # Second step (should use momentum)
        initial_params = [p.clone() for p in model.parameters()]
        
        optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        optimizer.step()
        
        # Check that parameters were updated
        for p_initial, p_updated in zip(initial_params, model.parameters()):
            assert not ncrsh.allclose(p_initial, p_updated)
    
    def test_adam_weight_decay(self):
        """Test Adam with weight decay."""
        model = Linear(10, 1)
        optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.1)
        
        # Take a step
        x = ncrsh.randn(5, 10)
        y = ncrsh.randn(5, 1)
        
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        # Check that parameters were updated
        for param in model.parameters():
            assert not ncrsh.allclose(param.grad, ncrsh.zeros_like(param))
    
    def test_optimizer_state_dict(self):
        """Test saving and loading optimizer state."""
        model = Linear(10, 1)
        optimizer = Adam(model.parameters(), lr=0.01)
        
        # Take a step
        x = ncrsh.randn(5, 10)
        y = ncrsh.randn(5, 1)
        
        def compute_loss():
            return ((model(x) - y) ** 2).mean()
        
        # First step
        optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        optimizer.step()
        
        # Save state
        state = optimizer.state_dict()
        
        # Take another step
        optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        optimizer.step()
        
        # Load previous state
        optimizer.load_state_dict(state)
        
        # Check that optimizer state was restored
        for p in model.parameters():
            assert 'exp_avg' in optimizer.state[p]
            assert 'exp_avg_sq' in optimizer.state[p]

if __name__ == "__main__":
    pytest.main([__file__])
