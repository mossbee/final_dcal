"""
DCAL: Dual Cross-Attention Learning for Fine-Grained Visual Categorization and Object Re-Identification

A PyTorch implementation of Dual Cross-Attention Learning that combines:
- Self-Attention (SA) for global feature learning
- Global-Local Cross-Attention (GLCA) for discriminative local regions
- Pair-Wise Cross-Attention (PWCA) for regularization during training

Tasks supported:
- Fine-Grained Visual Categorization (FGVC): CUB-200-2011, Stanford Cars, FGVC-Aircraft
- Re-Identification (ReID): VeRi-776, Market1501, DukeMTMC, MSMT17
"""

__version__ = '0.1.0'
__author__ = 'DCAL Implementation Team'

from .models import DCALModel, VisionTransformer, CONFIGS
from .configs import BaseConfig

__all__ = ['DCALModel', 'VisionTransformer', 'CONFIGS', 'BaseConfig']

