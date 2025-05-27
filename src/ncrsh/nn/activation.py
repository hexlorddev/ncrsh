"""
Activation functions for ncrsh.

This module contains various activation functions that can be used as non-linearities
in neural networks.
"""
from __future__ import annotations
from typing import Optional

from ..tensor import Tensor
from .modules import Module


class ReLU(Module):
    """
    Applies the rectified linear unit function element-wise:
    
    ReLU(x) = max(0, x)
    """
    def __init__(self, inplace: bool = False) -> None:
        """
        Args:
            inplace: can optionally do the operation in-place. Default: False
        """
        super().__init__()
        self.inplace = inplace
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: input tensor
        """
        # Implementation would go here
        pass
    
    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class GELU(Module):
    """
    Applies the Gaussian Error Linear Units function:
    
    GELU(x) = x * Φ(x)
    where Φ(x) is the Cumulative Distribution Function for Gaussian Distribution.
    """
    def __init__(self, approximate: str = 'none') -> None:
        """
        Args:
            approximate: the gelu approximation algorithm to use:
                        'none' | 'tanh'. Default: 'none'
        """
        super().__init__()
        self.approximate = approximate
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: input tensor
        """
        # Implementation would go here
        pass
    
    def extra_repr(self) -> str:
        return f'approximate={repr(self.approximate)}'


class Sigmoid(Module):
    """
    Applies the element-wise function:
    
    Sigmoid(x) = 1 / (1 + exp(-x))
    """
    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: input tensor
        """
        # Implementation would go here
        pass


class Tanh(Module):
    """
    Applies the element-wise function:
    
    Tanh(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: input tensor
        """
        # Implementation would go here
        pass
