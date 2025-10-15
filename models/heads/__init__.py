"""
Task-specific heads for DCAL.

Provides classification heads for FGVC and embedding heads for ReID.
"""

from .classification import ClassificationHead
from .reid import ReIDHead

__all__ = ['ClassificationHead', 'ReIDHead']

