"""
Core tensor operations for ncrsh.

This module provides the base Tensor class and core operations that are compatible
with PyTorch's API while providing additional optimizations.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union
import numpy as np

class Tensor:
    """
    A multi-dimensional array with GPU acceleration support.
    
    This class serves as the foundation for all ncrsh operations, providing
    a NumPy-like interface with automatic differentiation capabilities.
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, List, int, float],
        requires_grad: bool = False,
        device: Optional[str] = None,
        dtype: Optional[np.dtype] = None,
    ) -> None:
        """
        Initialize a new tensor.
        
        Args:
            data: Array-like data to initialize the tensor
            requires_grad: If True, tracks operations for gradient computation
            device: Device to store the tensor ('cpu' or 'cuda')
            dtype: Data type of the tensor elements
        """
        if isinstance(data, (list, tuple, int, float)):
            data = np.array(data, dtype=dtype)
        
        self._data = np.asarray(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.device = device or 'cpu'
        self.grad = None
        self.grad_fn = None
        
        # For automatic differentiation
        self._ctx = None
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the tensor."""
        return self._data.shape
    
    @property
    def dtype(self):
        """Return the data type of the tensor."""
        return self._data.dtype
    
    def numpy(self) -> np.ndarray:
        """Convert the tensor to a NumPy array."""
        return self._data.copy()
    
    def to(self, device: str) -> 'Tensor':
        """Move the tensor to the specified device."""
        if device == self.device:
            return self
        # In this basic implementation, we just change the device attribute
        # Actual device transfer would be implemented for GPU support
        new_tensor = Tensor(self._data, self.requires_grad, device, self.dtype)
        return new_tensor
    
    def backward(self, grad: Optional['Tensor'] = None) -> None:
        """Compute the gradient of the tensor."""
        if not self.requires_grad:
            raise RuntimeError("Trying to backward through a tensor that doesn't require grad")
            
        if grad is None:
            if self.shape != ():
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            grad = Tensor(1.0)
        
        self.grad = grad if self.grad is None else self.grad + grad
        
        if self._ctx is not None:
            self._ctx.backward(grad)
    
    def __repr__(self) -> str:
        return f"Tensor({self._data}, device='{self.device}', requires_grad={self.requires_grad})"
    
    # Basic arithmetic operations
    def __add__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        return add(self, other)
    
    def __mul__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        return mul(self, other)
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return matmul(self, other)
    
    def __truediv__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        return div(self, other)
    
    def __sub__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        return sub(self, other)
    
    def __neg__(self) -> 'Tensor':
        return neg(self)
    
    def __pow__(self, power: Union[int, float]) -> 'Tensor':
        return pow(self, power)

# Basic operation functions
def add(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise addition of two tensors."""
    # Implementation would go here
    pass

def mul(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise multiplication of two tensors."""
    # Implementation would go here
    pass

def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication of two tensors."""
    # Implementation would go here
    pass

def div(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise division of two tensors."""
    # Implementation would go here
    pass

def sub(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise subtraction of two tensors."""
    # Implementation would go here
    pass

def neg(t: Tensor) -> Tensor:
    """Negate a tensor."""
    # Implementation would go here
    pass

def pow(t: Tensor, power: Union[int, float]) -> Tensor:
    """Element-wise power operation."""
    # Implementation would go here
    pass
