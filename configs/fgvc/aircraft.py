"""
Configuration for FGVC-Aircraft dataset.

FGVC-Aircraft contains 10,000 images of 100 aircraft variants.
Training set: 6,667 images, Test set: 3,333 images
"""

from ..base import BaseConfig


class AircraftConfig(BaseConfig):
    """
    Configuration for FGVC-Aircraft fine-grained aircraft classification.
    
    Uses similar settings as CUB but for 100 aircraft variants.
    
    Attributes:
        num_classes (int): 100 aircraft variants
        data_root (str): Path to FGVC-Aircraft dataset
        Other settings similar to CUB
    """
    
    def __init__(self):
        pass

