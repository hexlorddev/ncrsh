""
L1 and Smooth L1 loss for ncrsh.
"""
from typing import Optional

from ...tensor import Tensor
from .loss import _Loss

class L1Loss(_Loss):
    """Creates a criterion that measures the mean absolute error (MAE) between each element in
    the input and target.
    
    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:
    
    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left| x_n - y_n \right|,
    
    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:
    
    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
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
        """Compute the L1 loss.
        
        Args:
            input (Tensor): Input tensor of shape (N, *), where * means any number of additional dimensions
            target (Tensor): Target tensor of the same shape as the input
            
        Returns:
            Tensor: The computed loss
        """
        return F.l1_loss(input, target, reduction=self.reduction)


class SmoothL1Loss(_Loss):
    """Creates a criterion that uses a squared term if the absolute
    element-wise error falls below beta and an L1 term otherwise.
    
    It is less sensitive to outliers than the `MSELoss` and in some cases
    prevents exploding gradients (e.g. see the paper `Fast R-CNN`_ by Ross Girshick).
    
    For a batch of size :math:`N`, the loss can be described as:
    
    .. math::
        \ell(x, y) = L = \{l_1, ..., l_N\}^\top
    
    with
    
    .. math::
        \ell(x, y) = \begin{cases}
        0.5 (x_i - y_i)^2 / beta, & \text{if } |x_i - y_i| < beta \\
        |x_i - y_i| - 0.5 * beta, & \text{otherwise}
        \end{cases}
    
    If :attr:`reduction` is not ``'none'`` (default ``'mean'``), then:
    
    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}
    
    .. note::
        Smooth L1 loss can be seen as exactly :class:`L1Loss`, but with the :math:`|x - y| < beta`
        portion replaced with a quadratic function such that its slope is 1 at :math:`|x - y| = 0`.
        The quadratic segment smooths the L1 loss near :math:`x = y`.
    
    .. note::
        Smooth L1 loss is closely related to Huber loss, being
        equivalent to :math:`huber(x, y) / beta` (note that Smooth L1's beta hyper-parameter is
        also known as delta for Huber). This leads to the following differences:
        
        * As beta -> 0, this loss becomes equivalent to :class:`L1Loss` (ignoring the 0.5 * beta term).
        * As beta -> +inf, this loss becomes more similar to :class:`MSELoss` (ignoring the beta term).
        * The square term is used when the absolute element-wise error is less than beta.
    
    .. _Fast R-CNN: https://openaccess.thecvf.com/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf
    
    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'mean'
        beta (float, optional): Specifies the threshold at which to change between L1 and L2 loss.
            The value must be non-negative. Default: 1.0
    """
    
    __constants__ = ['reduction', 'beta']
    
    def __init__(self, reduction: str = 'mean', beta: float = 1.0):
        super().__init__(reduction)
        self.beta = beta
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the Smooth L1 loss.
        
        Args:
            input (Tensor): Input tensor of shape (N, *), where * means any number of additional dimensions
            target (Tensor): Target tensor of the same shape as the input
            
        Returns:
            Tensor: The computed loss
        """
        return F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)
