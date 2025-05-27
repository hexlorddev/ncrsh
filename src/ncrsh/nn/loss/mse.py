""
Mean Squared Error (MSE) loss for ncrsh.
"""
from typing import Optional

from ...tensor import Tensor
from .loss import _Loss

class MSELoss(_Loss):
    """Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input
    and target.
    
    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:
    
    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,
    
    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:
    
    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}
    
    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.
    
    The mean operation still operates over all the elements, and divides by :math:`n`.
    
    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.
    
    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'mean'
    """
    
    __constants__ = ['reduction']
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction)
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the MSE loss.
        
        Args:
            input (Tensor): Input tensor of shape (N, *), where * means any number of additional dimensions
            target (Tensor): Target tensor of the same shape as the input
            
        Returns:
            Tensor: The computed loss
        """
        return F.mse_loss(input, target, reduction=self.reduction)
