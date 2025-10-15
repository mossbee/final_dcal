"""
Classification head for FGVC tasks.

Simple linear classifier that projects CLS token to class logits.
"""

import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Classification head for fine-grained visual categorization.
    
    Projects the CLS token embedding to class logits using a linear layer.
    
    Args:
        in_features (int): Input feature dimension (from ViT)
        num_classes (int): Number of output classes
        dropout (float): Dropout rate before classification
        
    Attributes:
        dropout: Dropout layer
        fc: Linear classification layer
        
    Methods:
        forward(x): Project features to logits
    """
    
    def __init__(self, in_features, num_classes, dropout=0.0):
        """Initialize classification head."""
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        """
        Forward pass through classification head.
        
        Args:
            x (torch.Tensor): CLS token features [B, D]
            
        Returns:
            torch.Tensor: Class logits [B, num_classes]
        """
        x = self.dropout(x)
        x = self.fc(x)
        return x

