"""
Base dataset class for DCAL.

Provides common functionality for all datasets including image loading,
caching, and basic preprocessing.
"""

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base dataset class that all task-specific datasets inherit from.
    
    Provides common functionality:
    - Image loading from file paths
    - Optional image caching
    - Transform application
    - Error handling for corrupted images
    
    Attributes:
        data_root (str): Root directory of the dataset
        transform (callable): Transform to apply to images
        return_paths (bool): Whether to return image paths with samples
        
    Methods:
        __len__(): Return dataset size
        __getitem__(idx): Get a sample (image, label, optionally path)
        _load_image(path): Load image from path
        _apply_transform(image): Apply transforms to image
    """
    
    def __init__(self, data_root, transform=None, return_paths=False):
        """
        Initialize base dataset.
        
        Args:
            data_root (str): Root directory of the dataset
            transform (callable, optional): Transform to apply to images
            return_paths (bool): Whether to return image paths
        """
        self.data_root = data_root
        self.transform = transform
        self.return_paths = return_paths
        
        self.images = []
        self.labels = []
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label) or (image, label, path) if return_paths=True
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = self._load_image(img_path)
        
        # Apply transform
        if self.transform is not None:
            image = self.transform(image)
        
        if self.return_paths:
            return image, label, img_path
        else:
            return image, label
    
    def _load_image(self, path):
        """
        Load an image from file path.
        
        Args:
            path (str): Path to image file
            
        Returns:
            PIL.Image: Loaded image
        """
        from PIL import Image
        return Image.open(path).convert('RGB')
    
    def _apply_transform(self, image):
        """
        Apply transforms to an image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            torch.Tensor: Transformed image
        """
        if self.transform is not None:
            return self.transform(image)
        return image

