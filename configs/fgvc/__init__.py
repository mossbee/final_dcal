"""
FGVC (Fine-Grained Visual Categorization) task configurations.

This module contains dataset-specific configurations for FGVC tasks including
CUB-200-2011, Stanford Cars, and FGVC-Aircraft.
"""

from .cub import CUBConfig
from .cars import CarsConfig
from .aircraft import AircraftConfig

__all__ = ['CUBConfig', 'CarsConfig', 'AircraftConfig']

