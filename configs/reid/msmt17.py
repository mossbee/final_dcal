"""
Configuration for MSMT17 person re-identification dataset.

MSMT17 is a large-scale person ReID dataset with 4,101 identities.
Training set: 32,621 images of 1,041 identities
Test set: 93,820 images of 3,060 identities
Query set: 11,659 images
"""

from ..base import BaseConfig


class MSMT17Config(BaseConfig):
    """
    Configuration for MSMT17 person re-identification.
    
    Uses ReID settings with pedestrian image size (256x128).
    Larger scale than Market1501 and Duke.
    
    Attributes:
        num_classes (int): 1,041 person identities (training set)
        data_root (str): Path to MSMT17 dataset
        img_size (tuple): (256, 128) for height x width
        Other ReID settings similar to VeRi
    """
    
    def __init__(self):
        pass

