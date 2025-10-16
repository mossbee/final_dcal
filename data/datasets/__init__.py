"""
Dataset implementations for FGVC and ReID tasks.

Provides dataset classes for loading and preprocessing images from various
fine-grained classification and re-identification benchmarks.
"""

from .base import BaseDataset
from .fgvc import CUBDataset, CarsDataset, AircraftDataset
from .reid import VeRiDataset, Market1501Dataset, DukeDataset, MSMT17Dataset

__all__ = [
    'BaseDataset',
    'CUBDataset', 'CarsDataset', 'AircraftDataset',
    'VeRiDataset', 'Market1501Dataset', 'DukeDataset', 'MSMT17Dataset'
]

