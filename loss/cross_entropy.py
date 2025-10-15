"""
Cross-entropy loss with optional label smoothing.

Standard loss function for classification tasks in FGVC and ReID.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Standard cross-entropy loss.
    
    Args:
        weight (Tensor, optional): Class weights
        reduction (str): Reduction method ('mean', 'sum', 'none')
        
    Methods:
        forward(logits, labels): Compute cross-entropy loss
    """
    
    def __init__(self, weight=None, reduction='mean'):
        """Initialize cross-entropy loss."""
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    
    def forward(self, logits, labels):
        """
        Compute cross-entropy loss.
        
        Args:
            logits (torch.Tensor): Predicted logits [B, num_classes]
            labels (torch.Tensor): Ground truth labels [B]
            
        Returns:
            torch.Tensor: Loss value
        """
        return self.ce(logits, labels)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Label smoothing prevents overfitting by mixing one-hot labels with
    a uniform distribution.
    
    Smoothed label: y_smooth = (1 - epsilon) * y_hot + epsilon / num_classes
    
    Args:
        smoothing (float): Label smoothing factor (default: 0.1)
        reduction (str): Reduction method
        
    Methods:
        forward(logits, labels): Compute label smoothing cross-entropy
    """
    
    def __init__(self, smoothing=0.1, reduction='mean'):
        """Initialize label smoothing cross-entropy."""
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits, labels):
        """
        Compute label smoothing cross-entropy.
        
        Args:
            logits (torch.Tensor): Predicted logits [B, num_classes]
            labels (torch.Tensor): Ground truth labels [B]
            
        Returns:
            torch.Tensor: Loss value
        """
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

