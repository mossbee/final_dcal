"""
MSMT17 person re-identification dataset loader.

MSMT17 is a large-scale person ReID dataset with 4,101 identities.
Training: 32,621 images of 1,041 identities
Test: 93,820 images of 3,060 identities
Query: 11,659 images
"""

from ..base import BaseDataset


class MSMT17Dataset(BaseDataset):
    """
    MSMT17 person re-identification dataset.
    
    Dataset structure:
        MSMT17/
        ├── train/
        ├── test/
        ├── query/ (part of test)
        └── list_*.txt files
    
    Filename format: person_id_camera_id_frame_number.jpg
    
    Attributes:
        num_classes (int): Number of person identities
        images (list): List of image paths
        pids (list): List of person IDs
        camids (list): List of camera IDs
        split (str): 'train', 'test', or 'query'
        
    Methods:
        _load_metadata(): Parse directory structure and annotation files
        _parse_filename(filename): Extract ID and camera info from filename
    """
    
    def __init__(self, data_root, split='train', transform=None, return_paths=False):
        """
        Initialize MSMT17 dataset.
        
        Args:
            data_root (str): Path to MSMT17 directory
            split (str): 'train', 'test', or 'query'
            transform (callable, optional): Transform to apply
            return_paths (bool): Whether to return image paths
        """
        pass
    
    def _load_metadata(self):
        """
        Load MSMT17 metadata by parsing directory and annotation files.
        """
        pass
    
    def _parse_filename(self, filename):
        """
        Parse MSMT17 filename to extract person ID and camera ID.
        
        Args:
            filename (str): Image filename
            
        Returns:
            tuple: (person_id, camera_id)
        """
        pass

