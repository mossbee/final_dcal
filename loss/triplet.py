"""
Triplet loss for ReID metric learning.

Implements batch hard triplet loss for learning discriminative embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Batch hard triplet loss for re-identification.
    
    For each anchor in the batch:
    - Hardest positive: Sample with same ID and maximum distance
    - Hardest negative: Sample with different ID and minimum distance
    
    Loss = max(0, d(a, p) - d(a, n) + margin)
    
    where:
    - d(a, p): Distance to hardest positive
    - d(a, n): Distance to hardest negative
    - margin: Margin for triplet separation
    
    Args:
        margin (float): Margin for triplet loss (default: 0.3)
        distance_metric (str): 'euclidean' or 'cosine'
        
    Methods:
        forward(embeddings, labels): Compute batch hard triplet loss
        _pairwise_distance(x): Compute pairwise distances
        _get_anchor_positive_triplet_mask(labels): Mask for valid (a, p) pairs
        _get_anchor_negative_triplet_mask(labels): Mask for valid (a, n) pairs
    """
    
    def __init__(self, margin=0.3, distance_metric='euclidean'):
        """Initialize triplet loss."""
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
    
    def forward(self, embeddings, labels):
        """
        Compute batch hard triplet loss.
        
        Args:
            embeddings (torch.Tensor): Feature embeddings [B, D]
            labels (torch.Tensor): Identity labels [B]
            
        Returns:
            torch.Tensor: Triplet loss value
        """
        # Compute pairwise distance
        pairwise_dist = self._pairwise_distance(embeddings)
        
        # Get hardest positive and negative for each anchor
        mask_anchor_positive = self._get_anchor_positive_mask(labels).float()
        mask_anchor_negative = self._get_anchor_negative_mask(labels).float()
        
        # Hardest positive: max distance among same ID
        anchor_positive_dist = pairwise_dist * mask_anchor_positive
        hardest_positive_dist, _ = anchor_positive_dist.max(dim=1, keepdim=True)
        
        # Hardest negative: min distance among different ID
        # Add large value to masked positions
        max_anchor_negative_dist = pairwise_dist.max().detach()
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        hardest_negative_dist, _ = anchor_negative_dist.min(dim=1, keepdim=True)
        
        # Combine hardest positive and negative
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        
        return triplet_loss.mean()
    
    def _pairwise_distance(self, x):
        """
        Compute pairwise distance matrix.
        
        Args:
            x (torch.Tensor): Embeddings [B, D]
            
        Returns:
            torch.Tensor: Distance matrix [B, B]
        """
        if self.distance_metric == 'euclidean':
            # Compute euclidean distance
            # ||a - b||^2 = ||a||^2 - 2 <a, b> + ||b||^2
            dot_product = torch.matmul(x, x.t())
            square_norm = dot_product.diag()
            distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
            distances = F.relu(distances)  # Numerical stability
            distances = torch.sqrt(distances + 1e-16)
        elif self.distance_metric == 'cosine':
            # Normalize embeddings
            x_normalized = F.normalize(x, p=2, dim=1)
            # Compute cosine distance (1 - cosine similarity)
            distances = 1.0 - torch.matmul(x_normalized, x_normalized.t())
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        return distances
    
    def _get_anchor_positive_mask(self, labels):
        """
        Get mask for valid anchor-positive pairs.
        
        Args:
            labels (torch.Tensor): Labels [B]
            
        Returns:
            torch.Tensor: Boolean mask [B, B]
        """
        # Check if labels[i] == labels[j]
        indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        indices_not_equal = ~indices_equal
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        # Valid anchor-positive: same label but not same sample
        mask = labels_equal & indices_not_equal
        return mask
    
    def _get_anchor_negative_mask(self, labels):
        """
        Get mask for valid anchor-negative pairs.
        
        Args:
            labels (torch.Tensor): Labels [B]
            
        Returns:
            torch.Tensor: Boolean mask [B, B]
        """
        # Check if labels[i] != labels[j]
        labels_not_equal = labels.unsqueeze(0) != labels.unsqueeze(1)
        return labels_not_equal

