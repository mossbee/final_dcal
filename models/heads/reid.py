"""
ReID head for person/vehicle re-identification.

Combines identity classification and embedding extraction for metric learning.
"""

import torch
import torch.nn as nn


class ReIDHead(nn.Module):
    """
    Re-identification head for metric learning.
    
    Consists of:
    1. BN-Dropout-Linear for embedding (bottleneck layer)
    2. BN-Dropout-Linear for classification (identity prediction)
    
    During training: Both classification and embedding losses
    During testing: Only use embedding for similarity matching
    
    Args:
        in_features (int): Input feature dimension
        num_classes (int): Number of identities (for classification)
        bottleneck_dim (int): Embedding dimension (default: 512)
        dropout (float): Dropout rate
        
    Attributes:
        bottleneck: Embedding layer (BN + Linear)
        dropout: Dropout layer
        classifier: Identity classification layer
        
    Methods:
        forward(x, return_embedding_only): Forward pass
    """
    
    def __init__(self, in_features, num_classes, bottleneck_dim=512, dropout=0.5):
        """Initialize ReID head."""
        super(ReIDHead, self).__init__()
        
        # Bottleneck layer for embedding
        self.bottleneck = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Classification layer
        self.classifier = nn.Linear(bottleneck_dim, num_classes, bias=False)
    
    def forward(self, x, return_embedding_only=False):
        """
        Forward pass through ReID head.
        
        Args:
            x (torch.Tensor): CLS token features [B, D]
            return_embedding_only (bool): Only return embeddings (for testing)
            
        Returns:
            If return_embedding_only=False (training):
                tuple: (embeddings [B, bottleneck_dim], logits [B, num_classes])
            If return_embedding_only=True (testing):
                torch.Tensor: Normalized embeddings [B, bottleneck_dim]
        """
        # Get bottleneck features
        feat = self.bottleneck(x)
        
        if return_embedding_only:
            # Normalize for testing
            return torch.nn.functional.normalize(feat, p=2, dim=1)
        
        # For training: return both embeddings and logits
        feat_drop = self.dropout(feat)
        logits = self.classifier(feat_drop)
        
        return feat, logits

