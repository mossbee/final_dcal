"""
Configuration for DukeMTMC-ReID person re-identification dataset.

DukeMTMC contains 1,812 identities captured by 8 cameras.
Training set: 16,522 images of 702 identities
Test set: 17,661 images of 1,110 identities (702 + 408 distractors)
Query set: 2,228 images
"""

from ..base import BaseConfig


class DukeConfig(BaseConfig):
    """
    Configuration for DukeMTMC-ReID person re-identification.
    
    Uses ReID settings with pedestrian image size (256x128).
    
    Attributes:
        num_classes (int): 702 person identities (training set)
        data_root (str): Path to DukeMTMC-ReID dataset
        img_size (tuple): (256, 128) for height x width
        Other ReID settings similar to VeRi
    """
    
    def __init__(self):
        pass

