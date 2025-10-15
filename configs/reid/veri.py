"""
Configuration for VeRi-776 vehicle re-identification dataset.

VeRi-776 contains 776 vehicles captured by 20 cameras.
Training set: 37,778 images of 576 vehicles
Test set: 11,579 images of 200 vehicles
Query set: 1,678 images
"""

from ..base import BaseConfig


class VeRiConfig(BaseConfig):
    """
    Configuration for VeRi-776 vehicle re-identification.
    
    Settings follow the paper specifications:
    - Image size: 256x256 (vehicle dataset)
    - Top R = 30% for GLCA
    - Optimizer: SGD with momentum 0.9, weight decay 1e-4
    - LR: 0.008, cosine decay
    - Epochs: 120
    - Batch size: 64 (P=16 IDs, K=4 images per ID)
    - Loss: Cross-entropy + Triplet with uncertainty weighting
    - Sampler: RandomIdentitySampler for P-K sampling
    
    Attributes:
        num_classes (int): 776 vehicle identities
        data_root (str): Path to VeRi dataset
        img_size (int): 256 for both height and width
        glca_top_r (float): 0.3 (30% top attention regions)
        num_instances (int): 4 (K images per ID)
        batch_size (int): 64 (P=16 IDs * K=4 images)
        And other ReID-specific settings...
    """
    
    def __init__(self):
        pass

