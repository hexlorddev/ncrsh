"""
Tests for the ncrsh.tensor module.
"""
import pytest
import numpy as np

from ncrsh.tensor import Tensor

def test_tensor_creation():
    """Test basic tensor creation."""
    # Test creation from list
    t = Tensor([1, 2, 3])
    assert t.shape == (3,)
    assert t.dtype == np.int64
    
    # Test creation from numpy array
    arr = np.array([1.0, 2.0, 3.0])
    t = Tensor(arr)
    assert t.shape == (3,)
    assert t.dtype == np.float64

def test_tensor_operations():
    """Test basic tensor operations."""
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    
    # Test addition
    c = a + b
    assert isinstance(c, Tensor)
    assert c.shape == (3,)
    assert np.array_equal(c.numpy(), np.array([5, 7, 9]))
    
    # Test multiplication
    d = a * 2
    assert isinstance(d, Tensor)
    assert np.array_equal(d.numpy(), np.array([2, 4, 6]))

def test_tensor_device():
    """Test tensor device handling."""
    t = Tensor([1, 2, 3])
    assert t.device == 'cpu'
    
    # Test moving to CPU (should be a no-op in this basic implementation)
    t_cpu = t.to('cpu')
    assert t_cpu.device == 'cpu'

@pytest.mark.skip(reason="GPU testing requires CUDA support")
def test_tensor_gpu():
    """Test tensor movement to GPU."""
    t = Tensor([1, 2, 3])
    t_gpu = t.to('cuda')
    assert t_gpu.device == 'cuda'
