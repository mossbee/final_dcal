"""
Training and evaluation engine for DCAL.

Provides trainers and evaluators for FGVC and ReID tasks.
"""

from .trainer import BaseTrainer
from .fgvc_trainer import FGVCTrainer
from .reid_trainer import ReIDTrainer
from .evaluator import BaseEvaluator
from .fgvc_evaluator import FGVCEvaluator
from .reid_evaluator import ReIDEvaluator

__all__ = [
    'BaseTrainer', 'FGVCTrainer', 'ReIDTrainer',
    'BaseEvaluator', 'FGVCEvaluator', 'ReIDEvaluator'
]

