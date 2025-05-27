"""
Loss functions for ncrsh.

This module contains various loss functions that can be used for training neural networks.
"""

from .loss import _Loss, _WeightedLoss
from .mse import MSELoss
from .l1 import L1Loss, SmoothL1Loss
from .cross_entropy import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from .nll import NLLLoss
from .hinge import HingeEmbeddingLoss, MultiMarginLoss, MultiLabelMarginLoss
from .cosine_embedding import CosineEmbeddingLoss
from .ctc import CTCLoss
from .poisson_nll import PoissonNLLLoss
from .kl_div import KLDivLoss
from .margin_ranking import MarginRankingLoss
from .multi_label_soft_margin import MultiLabelSoftMarginLoss
from .soft_margin import SoftMarginLoss
from .triplet_margin import TripletMarginLoss, TripletMarginWithDistanceLoss

__all__ = [
    '_Loss',
    '_WeightedLoss',
    'MSELoss',
    'L1Loss',
    'SmoothL1Loss',
    'CrossEntropyLoss',
    'BCELoss',
    'BCEWithLogitsLoss',
    'NLLLoss',
    'HingeEmbeddingLoss',
    'MultiMarginLoss',
    'MultiLabelMarginLoss',
    'CosineEmbeddingLoss',
    'CTCLoss',
    'PoissonNLLLoss',
    'KLDivLoss',
    'MarginRankingLoss',
    'MultiLabelSoftMarginLoss',
    'SoftMarginLoss',
    'TripletMarginLoss',
    'TripletMarginWithDistanceLoss',
]
