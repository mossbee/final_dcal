"""
Stanford Cars dataset loader.

Stanford Cars contains 16,185 images of 196 car classes.
Training set: 8,144 images, Test set: 8,041 images
"""

from ..base import BaseDataset


class CarsDataset(BaseDataset):
    """
    Stanford Cars dataset.
    
    Dataset structure:
        stanford_cars/
        ├── cars_train/
        ├── cars_test/
        ├── devkit/
        │   ├── cars_train_annos.mat
        │   ├── cars_test_annos.mat
        │   └── cars_meta.mat
        └── ...
    
    Attributes:
        num_classes (int): 196 car classes
        images (list): List of image paths
        labels (list): List of class labels (0-195)
        is_train (bool): Whether this is training or test set
        
    Methods:
        _load_metadata(): Load annotations from .mat files
    """
    
    def __init__(self, data_root, split='train', transform=None, return_paths=False):
        """
        Initialize Stanford Cars dataset.
        
        Args:
            data_root (str): Path to stanford_cars directory
            split (str): 'train' or 'test'
            transform (callable, optional): Transform to apply
            return_paths (bool): Whether to return image paths
        """
        pass
    
    def _load_metadata(self):
        """
        Load and parse Stanford Cars annotations from .mat files.
        
        Uses scipy.io.loadmat to read MATLAB annotation files.
        """
        pass

