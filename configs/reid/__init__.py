"""
ReID (Re-Identification) task configurations.

This module contains dataset-specific configurations for ReID tasks including
VeRi-776 (vehicle), Market1501, DukeMTMC, and MSMT17 (person).
"""

from .veri import VeRiConfig
from .market1501 import Market1501Config
from .duke import DukeConfig
from .msmt17 import MSMT17Config

__all__ = ['VeRiConfig', 'Market1501Config', 'DukeConfig', 'MSMT17Config']

