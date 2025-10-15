"""
Uncertainty-based loss weighting for multi-task learning.

Automatically balances losses from SA, GLCA, and PWCA branches using
learnable uncertainty parameters.
"""

import torch
import torch.nn as nn


class UncertaintyWeighting(nn.Module):
    """
    Uncertainty-based loss weighting for DCAL.
    
    Automatically balances multiple loss terms using learnable weights.
    From: "Multi-Task Learning Using Uncertainty to Weigh Losses for 
    Scene Geometry and Semantics" (Kendall et al., CVPR 2018)
    
    Formula:
        L_total = 1/2 * (L1/exp(w1) + L2/exp(w2) + L3/exp(w3) + w1 + w2 + w3)
    
    where w1, w2, w3 are learnable parameters that automatically determine
    the relative importance of each loss term.
    
    Args:
        num_losses (int): Number of loss terms (default: 3 for SA/GLCA/PWCA)
        init_weights (list, optional): Initial weights (default: all zeros)
        
    Attributes:
        log_vars: Learnable log-variance parameters [num_losses]
        
    Methods:
        forward(losses): Combine losses with uncertainty weighting
        get_weights(): Get current loss weights (exp(-w))
    """
    
    def __init__(self, num_losses=3, init_weights=None):
        """Initialize uncertainty weighting module."""
        super(UncertaintyWeighting, self).__init__()
        if init_weights is None:
            init_weights = [0.0] * num_losses
        self.log_vars = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))
    
    def forward(self, losses):
        """
        Combine multiple losses with uncertainty weighting.
        
        Args:
            losses (list or dict): List of loss values or dict of losses
            
        Returns:
            tuple: (weighted_loss, loss_weights)
        """
        if isinstance(losses, dict):
            losses = list(losses.values())
        
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        
        total_loss = 0.5 * total_loss
        
        return total_loss, self.get_weights()
    
    def get_weights(self):
        """
        Get current loss weights.
        
        Returns:
            torch.Tensor: Weights (exp(-log_var)) for each loss term
        """
        return torch.exp(-self.log_vars)

