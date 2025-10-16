"""
Configuration for CUB-200-2011 dataset.

CUB-200-2011 contains 11,788 images of 200 bird species.
Training set: ~5,994 images, Test set: ~5,794 images
"""

from ..base import BaseConfig


class CUBConfig(BaseConfig):
    """
    Configuration for CUB-200-2011 fine-grained bird classification.
    
    Settings follow the paper specifications:
    - Image size: 550x550 -> random crop to 448x448
    - Sequence length: 28x28 = 784 patches (patch size 16)
    - Top R = 10% for GLCA
    - Optimizer: Adam with weight decay 0.05
    - LR: 5e-4 / 512 * batch_size, cosine decay
    - Epochs: 100
    - Batch size: 16 (paper used 4 GPUs; use 4-8 for single GPU)
    - Loss: Cross-entropy with uncertainty weighting
    - Stochastic depth enabled
    
    Note: The paper used 4 GPUs for training. For single GPU (16GB):
    - Recommended batch size: 4-8
    - Adjust learning rate proportionally: lr = (5e-4 / 512) * batch_size
    
    Attributes:
        num_classes (int): 200 bird species
        data_root (str): Path to CUB_200_2011 dataset
        img_size (int): 550 (resize before crop)
        crop_size (int): 448 (training crop size)
        glca_top_r (float): 0.1 (10% top attention regions)
        And other FGVC-specific settings...
    """
    
    def __init__(self):
        super(CUBConfig, self).__init__()
        
        # Dataset specific
        self.num_classes = 200
        self.data_root = 'data/CUB_200_2011'
        
        # Image size (FGVC specific)
        self.img_size = 550  # Resize size
        self.crop_size = 448  # Crop size
        
        # GLCA settings for FGVC
        self.glca_top_r = 0.1  # 10% top regions
        
        # Training settings for FGVC
        self.task = 'fgvc'
        self.optimizer = 'adam'
        self.weight_decay = 0.05
        self.epochs = 100
        # Batch size: 8 for single GPU (16GB), paper used 16 with 4 GPUs
        self.batch_size = 8
        
        # LR scaling: 5e-4 / 512 * batch_size
        self.learning_rate = (5e-4 / 512) * self.batch_size
        
        # Loss
        self.use_uncertainty_weighting = True
        self.label_smoothing = 0.0
        
        # Stochastic depth
        self.stochastic_depth_prob = 0.1

