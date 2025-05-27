"""
Tests for loss functions.
"""
import numpy as np
import pytest

import ncrsh
from ncrsh.nn import (
    MSELoss, L1Loss, SmoothL1Loss, 
    CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
)

class TestLossFunctions:
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_mse_loss(self, reduction):
        """Test Mean Squared Error loss."""
        criterion = MSELoss(reduction=reduction)
        
        # Test with random inputs
        input = ncrsh.randn(3, 5, requires_grad=True)
        target = ncrsh.randn(3, 5)
        
        loss = criterion(input, target)
        
        # Check output shape
        if reduction == "none":
            assert loss.shape == (3, 5)
        elif reduction == "sum":
            assert loss.dim() == 0
        else:  # mean
            assert loss.dim() == 0
        
        # Test backward
        loss.backward()
        assert input.grad is not None
    
    def test_l1_loss(self):
        """Test L1 loss."""
        criterion = L1Loss()
        
        input = ncrsh.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = ncrsh.tensor([[0.0, 2.0], [3.0, 5.0]])
        
        loss = criterion(input, target)
        expected = (1.0 + 0.0 + 0.0 + 1.0) / 4.0  # mean of differences
        
        assert ncrsh.allclose(loss, ncrsh.tensor(expected))
    
    def test_smooth_l1_loss(self):
        """Test Smooth L1 loss."""
        criterion = SmoothL1Loss(beta=1.0)
        
        # Test case where |x| < beta
        input = ncrsh.tensor([0.5])
        target = ncrsh.tensor([0.0])
        loss = criterion(input, target)
        expected = 0.5 * (0.5 ** 2) / 1.0  # 0.5 * x^2 / beta
        assert ncrsh.allclose(loss, ncrsh.tensor(expected))
        
        # Test case where |x| >= beta
        input = ncrsh.tensor([2.0])
        loss = criterion(input, target)
        expected = 2.0 - 1.0/2.0  # |x| - beta/2
        assert ncrsh.allclose(loss, ncrsh.tensor(expected))
    
    def test_cross_entropy_loss(self):
        """Test Cross Entropy loss."""
        criterion = CrossEntropyLoss()
        
        # 2D case (batch, classes)
        input = ncrsh.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        target = ncrsh.tensor([2, 0])  # class indices
        
        loss = criterion(input, target)
        
        # Manual computation
        log_softmax = ncrsh.log_softmax(input, dim=1)
        expected = -1/2 * (log_softmax[0, 2] + log_softmax[1, 0])
        
        assert ncrsh.allclose(loss, expected)
    
    def test_bce_loss(self):
        """Test Binary Cross Entropy loss."""
        criterion = BCELoss()
        
        input = ncrsh.tensor([0.8, 0.2, 0.4], requires_grad=True)
        target = ncrsh.tensor([1.0, 0.0, 1.0])
        
        loss = criterion(input, target)
        
        # Manual computation
        expected = - (ncrsh.log(ncrsh.tensor(0.8)) + 
                     ncrsh.log(ncrsh.tensor(0.8)) + 
                     ncrsh.log(ncrsh.tensor(0.4))) / 3.0
        
        assert ncrsh.allclose(loss, expected, rtol=1e-4)
        
        # Test backward
        loss.backward()
        assert input.grad is not None
    
    def test_bce_with_logits_loss(self):
        """Test BCE with logits loss."""
        criterion = BCEWithLogitsLoss()
        
        # Test with sigmoid activation
        logits = ncrsh.tensor([1.0, -1.0, 0.0], requires_grad=True)
        target = ncrsh.tensor([1.0, 0.0, 1.0])
        
        loss = criterion(logits, target)
        
        # Compare with BCELoss + sigmoid
        sigmoid = 1 / (1 + ncrsh.exp(-logits.detach()))
        expected = BCELoss()(sigmoid, target)
        
        assert ncrsh.allclose(loss, expected, rtol=1e-4)
        
        # Test backward
        loss.backward()
        assert logits.grad is not None

if __name__ == "__main__":
    pytest.main([__file__])
