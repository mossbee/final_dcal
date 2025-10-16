"""
FGVC (Fine-Grained Visual Categorization) dataset implementations.

Provides dataset loaders for CUB-200-2011, Stanford Cars, and FGVC-Aircraft.
"""

from .cub import CUBDataset
from .cars import CarsDataset
from .aircraft import AircraftDataset

__all__ = ['CUBDataset', 'CarsDataset', 'AircraftDataset']

