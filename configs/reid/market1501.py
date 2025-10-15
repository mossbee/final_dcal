"""
Configuration for Market1501 person re-identification dataset.

Market1501 contains 1,501 identities captured by 6 cameras.
Training set: 12,936 images of 751 identities
Test set: 19,732 images of 750 identities
Query set: 3,368 images
"""

from ..base import BaseConfig


class Market1501Config(BaseConfig):
    """
    Configuration for Market1501 person re-identification.
    
    Uses ReID settings with pedestrian image size (256x128).
    
    Attributes:
        num_classes (int): 751 person identities (training set)
        data_root (str): Path to Market1501 dataset
        img_size (tuple): (256, 128) for height x width
        Other ReID settings similar to VeRi
    """
    
    def __init__(self):
        pass

