"""
FGVC-Aircraft dataset loader.

FGVC-Aircraft contains 10,000 images of 100 aircraft variants.
Training set: 6,667 images, Test set: 3,333 images
"""

from ..base import BaseDataset


class AircraftDataset(BaseDataset):
    """
    FGVC-Aircraft dataset.
    
    Dataset structure:
        fgvc-aircraft-2013b/
        ├── data/
        │   ├── images/
        │   ├── images_variant_train.txt
        │   ├── images_variant_test.txt
        │   └── variants.txt
        └── ...
    
    Attributes:
        num_classes (int): 100 aircraft variants
        images (list): List of image paths
        labels (list): List of class labels (0-99)
        is_train (bool): Whether this is training or test set
        variant_names (list): List of aircraft variant names
        
    Methods:
        _load_metadata(): Load annotations from text files
    """
    
    def __init__(self, data_root, split='train', transform=None, return_paths=False):
        """
        Initialize FGVC-Aircraft dataset.
        
        Args:
            data_root (str): Path to fgvc-aircraft-2013b directory
            split (str): 'train' or 'test'
            transform (callable, optional): Transform to apply
            return_paths (bool): Whether to return image paths
        """
        pass
    
    def _load_metadata(self):
        """
        Load and parse FGVC-Aircraft annotations from text files.
        """
        pass

