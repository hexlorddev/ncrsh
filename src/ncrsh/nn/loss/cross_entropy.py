""
Cross-entropy loss functions for ncrsh.
"""
from typing import Optional, List, Union

from ...tensor import Tensor
from .loss import _WeightedLoss

class CrossEntropyLoss(_WeightedLoss):
    """This criterion computes the cross entropy loss between input logits and target.
    
    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes. This is particularly useful when
    you have an unbalanced training set.
    
    The `input` is expected to contain the unnormalized logits for each class.
    `input` has to be a Tensor of size :math:`(C)` for unbatched input,
    :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
    for the `K`-dimensional case.
    
    The `target` that this criterion expects should contain either:
    - Class indices in the range :math:`[0, C)` where :math:`C` is the number of classes; if
      `ignore_index` is specified, this loss also accepts this class index (this index
      may not necessarily be in the class range). The unreduced (i.e. with :attr:`reduction`
      set to ``'none'``) loss for this case can be described as:
    
      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
          \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}
    
      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` is the batch size. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then
    
      .. math::
          \ell(x, y) = \begin{cases}
              \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n} \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}} l_n, &
               \text{if reduction} = \text{`mean';}\\
              \sum_{n=1}^N l_n,  &
              \text{if reduction} = \text{`sum'.}
          \end{cases}
    
    - Probabilities for each class; useful when labels beyond a single class per minibatch item
      are required, such as for blended labels, label smoothing, etc. The unreduced (i.e.
      with :attr:`reduction` set to ``'none'``) loss for this case can be described as:
    
      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - \sum_{c=1}^C w_c \log \frac{\exp(x_{n,c})}{\sum_{i=1}^C \exp(x_{n,i})} y_{n,c}
    
      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` is the batch size. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then
    
      .. math::
          \ell(x, y) = \begin{cases}
              \frac{\sum_{n=1}^N l_n}{N}, &
               \text{if reduction} = \text{`mean';}\\
              \sum_{n=1}^N l_n,  &
              \text{if reduction} = \text{`sum'.}
          \end{cases}
    
    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C`
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'mean'
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Note that
            :attr:`ignore_index` is only applicable when the target contains class indices.
        label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount
            of smoothing when computing the loss, where 0.0 means no smoothing.
            The targets become a mixture of the original ground truth and a uniform
            distribution as described in `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__.
            Default: :math:`0.0`.
    """
    
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float
    
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        reduction: str = 'mean',
        ignore_index: int = -100,
        label_smoothing: float = 0.0
    ) -> None:
        super().__init__(weight=weight, reduction=reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the cross entropy loss.
        
        Args:
            input (Tensor): Predicted unnormalized logits; see Shape section above.
            target (Tensor): Ground truth class indices or class probabilities; see Shape section above.
            
        Returns:
            Tensor: The computed loss
        """
        return F.cross_entropy(
            input, target, weight=self.weight,
            ignore_index=self.ignore_index, reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )


class BCELoss(_WeightedLoss):
    """Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities.
    
    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:
    
    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],
    
    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then
    
    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}
    
    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `y` should be numbers between 0 and 1.
    
    Notice that if any of the elements of :math:`x` is not in the range :math:`[0, 1]`,
    the loss will be non-negative and may return `nan`.
    
    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size `nbatch`.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'mean'
    """
    
    __constants__ = ['reduction']
    
    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean') -> None:
        super().__init__(weight=weight, reduction=reduction)
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the binary cross entropy loss.
        
        Args:
            input (Tensor): Predicted probabilities (must be in [0, 1] range).
            target (Tensor): Ground truth probabilities (must be in [0, 1] range).
            
        Returns:
            Tensor: The computed loss
        """
        return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)


class BCEWithLogitsLoss(_WeightedLoss):
    """This loss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.
    
    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:
    
    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log \sigma(x_n)
        + (1 - y_n) \cdot \log (1 - \sigma(x_n)) \right],
    
    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then
    
    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}
    
    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `t[i]` should be numbers
    between 0 and 1.
    
    It's possible to trade off recall and precision by adding weights to positive examples.
    In the case of multi-label classification the loss can be described as:
    
    .. math::
        \ell_c(x, y) = L_c = \{l_{1,c},\dots,l_{N,c}\}^\top, \quad
        l_{n,c} = - w_{n,c} \left[ p_c y_{n,c} \cdot \log \sigma(x_{n,c})
        + (1 - y_{n,c}) \cdot \log (1 - \sigma(x_{n,c})) \right],
    
    where :math:`c` is the class number (:math:`c > 1` for multi-label binary classification,
    :math:`c = 1` for single-label binary classification),
    :math:`n` is the batch size and :math:`p_c` is the weight of the positive answer for the class :math:`c`.
    
    :math:`p_c > 1` increases the recall, :math:`p_c < 1` increases the precision.
    
    For example, if a dataset contains 100 positive and 300 negative examples of a single class,
    then `pos_weight` for the class should be equal to :math:`\frac{300}{100}=3`.
    The loss would act as if the dataset contains :math:`3\times 100=300` positive examples.
    
    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size `nbatch`.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'mean'
        pos_weight (Tensor, optional): a weight of positive examples.
            Must be a vector with length equal to the number of classes.
    """
    
    __constants__ = ['reduction']
    
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        reduction: str = 'mean',
        pos_weight: Optional[Tensor] = None
    ) -> None:
        super().__init__(weight=weight, reduction=reduction)
        self.register_buffer('pos_weight', pos_weight)
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the binary cross entropy with logits loss.
        
        Args:
            input (Tensor): Predicted logits (before sigmoid).
            target (Tensor): Ground truth probabilities (must be in [0, 1] range).
            
        Returns:
            Tensor: The computed loss
        """
        return F.binary_cross_entropy_with_logits(
            input, target, self.weight, reduction=self.reduction, pos_weight=self.pos_weight
        )
