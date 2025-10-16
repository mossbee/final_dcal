"""
CUB-200-2011 dataset loader.

CUB-200-2011 (Caltech-UCSD Birds) is a fine-grained bird classification dataset
with 200 bird species, 11,788 images.
"""

from ..base import BaseDataset


class CUBDataset(BaseDataset):
    """
    CUB-200-2011 bird classification dataset.
    
    Dataset structure:
        CUB_200_2011/
        ├── images/
        │   ├── 001.Black_footed_Albatross/
        │   ├── 002.Laysan_Albatross/
        │   └── ...
        ├── images.txt
        ├── image_class_labels.txt
        ├── train_test_split.txt
        └── bounding_boxes.txt
    
    Attributes:
        num_classes (int): 200 bird species
        images (list): List of image paths
        labels (list): List of class labels (0-199)
        is_train (bool): Whether this is training or test set
        
    Methods:
        _load_metadata(): Parse dataset annotation files
        _get_image_path(image_id): Get full path for an image
    """
    
    def __init__(self, data_root, split='train', transform=None, return_paths=False):
        """
        Initialize CUB-200-2011 dataset.
        
        Args:
            data_root (str): Path to CUB_200_2011 directory
            split (str): 'train' or 'test'
            transform (callable, optional): Transform to apply
            return_paths (bool): Whether to return image paths
        """
        super(CUBDataset, self).__init__(data_root, transform, return_paths)
        self.split = split
        self.is_train = (split == 'train')
        self.num_classes = 200
        
        self._load_metadata()
    
    def _load_metadata(self):
        """
        Load and parse CUB dataset metadata files.
        
        Reads:
        - images.txt: Image IDs and paths
        - image_class_labels.txt: Class labels for each image
        - train_test_split.txt: Train/test split indicator
        """
        import os
        
        # Load image paths
        images_file = os.path.join(self.data_root, 'images.txt')
        with open(images_file, 'r') as f:
            image_lines = f.readlines()
        image_dict = {}
        for line in image_lines:
            img_id, img_path = line.strip().split()
            image_dict[int(img_id)] = img_path
        
        # Load class labels
        labels_file = os.path.join(self.data_root, 'image_class_labels.txt')
        with open(labels_file, 'r') as f:
            label_lines = f.readlines()
        label_dict = {}
        for line in label_lines:
            img_id, class_id = line.strip().split()
            label_dict[int(img_id)] = int(class_id) - 1  # Convert to 0-indexed
        
        # Load train/test split
        split_file = os.path.join(self.data_root, 'train_test_split.txt')
        with open(split_file, 'r') as f:
            split_lines = f.readlines()
        
        # Build dataset
        for line in split_lines:
            img_id, is_train = line.strip().split()
            img_id = int(img_id)
            is_train = int(is_train)
            
            if (self.is_train and is_train == 1) or (not self.is_train and is_train == 0):
                img_path = os.path.join(self.data_root, 'images', image_dict[img_id])
                self.images.append(img_path)
                self.labels.append(label_dict[img_id])
    
    def _get_image_path(self, image_id):
        """Get full path for an image given its ID."""
        return self.images[image_id]

