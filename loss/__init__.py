"""
Loss functions for DCAL.

Provides cross-entropy, triplet loss, and uncertainty-based loss weighting.
"""

from .cross_entropy import CrossEntropyLoss, LabelSmoothingCrossEntropy
from .triplet import TripletLoss
from .uncertainty_loss import UncertaintyWeighting

__all__ = [
    'CrossEntropyLoss', 'LabelSmoothingCrossEntropy',
    'TripletLoss',
    'UncertaintyWeighting'
]

