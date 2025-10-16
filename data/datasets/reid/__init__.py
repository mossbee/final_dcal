"""
ReID (Re-Identification) dataset implementations.

Provides dataset loaders for VeRi-776, Market1501, DukeMTMC-ReID, and MSMT17.
"""

from .veri import VeRiDataset
from .market1501 import Market1501Dataset
from .duke import DukeDataset
from .msmt17 import MSMT17Dataset

__all__ = ['VeRiDataset', 'Market1501Dataset', 'DukeDataset', 'MSMT17Dataset']

