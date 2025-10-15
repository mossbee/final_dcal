"""
Miscellaneous utility functions.

Provides random seed setting, device management, and other helpers.
"""

import torch
import random
import numpy as np


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU and CUDA)
    - CUDA deterministic operations
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cuda=True, device_id=0):
    """
    Get torch device.
    
    Args:
        cuda (bool): Whether to use CUDA
        device_id (int): CUDA device ID
        
    Returns:
        torch.device: Device object
    """
    if cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
    else:
        device = torch.device('cpu')
    return device


def count_parameters(model, trainable_only=True):
    """
    Count model parameters.
    
    Args:
        model (nn.Module): Model
        trainable_only (bool): Only count trainable parameters
        
    Returns:
        int: Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_time(seconds):
    """
    Format seconds into human-readable time.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def create_dir(path):
    """
    Create directory if it doesn't exist.
    
    Args:
        path (str): Directory path
    """
    import os
    os.makedirs(path, exist_ok=True)

