"""
Weight initialization utilities.

Provides truncated normal initialization and other weight init methods.
"""

import math
import torch
import torch.nn as nn


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    Initialize tensor with truncated normal distribution.
    
    Values are drawn from a normal distribution with specified mean and std,
    then truncated to [a*std + mean, b*std + mean].
    
    Args:
        tensor (torch.Tensor): Tensor to initialize
        mean (float): Mean of the normal distribution
        std (float): Standard deviation
        a (float): Minimum cutoff (in std units)
        b (float): Maximum cutoff (in std units)
        
    Returns:
        torch.Tensor: Initialized tensor
    """
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a * std + mean, max=b * std + mean)
        
        return tensor


def init_weights(module):
    """
    Initialize weights for various layer types.
    
    - Linear layers: Xavier uniform initialization
    - LayerNorm layers: weight=1, bias=0
    - Conv2d layers: Kaiming normal initialization
    
    Args:
        module (nn.Module): Module to initialize
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

