"""
Base loss classes for ncrsh.
"""
from typing import Optional, Union, List, Tuple, Dict, Any

from ...tensor import Tensor
from ... import functional as F

class _Loss:
    """Base class for all loss functions.
    
    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'mean'
    """
    
    def __init__(self, reduction: str = 'mean'):
        self.reduction = reduction
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        """
        raise NotImplementedError("forward() must be implemented in subclasses")
    
    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        """Call the forward method."""
        return self.forward(input, target)
    
    def extra_repr(self) -> str:
        """Set the extra representation of the module."""
        return f'reduction={self.reduction}'


class _WeightedLoss(_Loss):
    """Base class for weighted loss functions.
    
    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the weighted mean of the output is taken,
            'sum': the output will be summed. Default: 'mean'
    """
    
    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean'):
        super().__init__(reduction)
        self.register_buffer('weight', weight)
    
    def extra_repr(self) -> str:
        """Set the extra representation of the module."""
        s = f'reduction={self.reduction}'
        if self.weight is not None:
            s += f', weight={self.weight}'
        return s
