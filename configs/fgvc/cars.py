"""
Configuration for Stanford Cars dataset.

Stanford Cars contains 16,185 images of 196 car classes.
Training set: 8,144 images, Test set: 8,041 images
"""

from ..base import BaseConfig


class CarsConfig(BaseConfig):
    """
    Configuration for Stanford Cars fine-grained car classification.
    
    Uses similar settings as CUB but for 196 car classes.
    
    Attributes:
        num_classes (int): 196 car classes
        data_root (str): Path to Stanford Cars dataset
        Other settings similar to CUB
    """
    
    def __init__(self):
        pass

