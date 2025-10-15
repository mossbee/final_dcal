"""
Checkpoint saving and loading utilities.

Handles model state saving/loading, optimizer state, and scheduler state.
"""

import torch
import os


def save_checkpoint(state, filepath, is_best=False, best_filepath=None):
    """
    Save model checkpoint.
    
    Args:
        state (dict): Checkpoint state containing:
            - 'epoch': Current epoch
            - 'model_state_dict': Model state
            - 'optimizer_state_dict': Optimizer state
            - 'scheduler_state_dict': Scheduler state (optional)
            - 'best_metric': Best validation metric
            - 'config': Configuration
        filepath (str): Path to save checkpoint
        is_best (bool): Whether this is the best model
        best_filepath (str, optional): Path for best model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    
    if is_best and best_filepath is not None:
        import shutil
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        filepath (str): Path to checkpoint file
        model (nn.Module): Model to load state into
        optimizer (optional): Optimizer to load state into
        scheduler (optional): Scheduler to load state into
        device: Device to load checkpoint to
        
    Returns:
        dict: Checkpoint state (epoch, best_metric, etc.)
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def load_pretrained_weights(model, pretrained_path, strict=True):
    """
    Load pretrained weights (e.g., ImageNet pretrained).
    
    Handles loading from .npz (Google ViT weights) or .pth files.
    
    Args:
        model (nn.Module): Model to load weights into
        pretrained_path (str): Path to pretrained weights
        strict (bool): Whether to strictly enforce weight matching
    """
    if pretrained_path.endswith('.npz'):
        # Load Google ViT pretrained weights
        import numpy as np
        weights = np.load(pretrained_path)
        # This will be handled by the ViT model's load_from method
        if hasattr(model, 'load_from'):
            model.load_from(weights)
        else:
            raise AttributeError("Model does not have load_from method for .npz weights")
    elif pretrained_path.endswith('.pth') or pretrained_path.endswith('.pt'):
        # Load PyTorch weights
        state_dict = torch.load(pretrained_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=strict)
    else:
        raise ValueError(f"Unsupported pretrained weight format: {pretrained_path}")

