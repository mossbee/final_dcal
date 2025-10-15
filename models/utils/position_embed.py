"""
Position embedding utilities for Vision Transformer.

Provides 2D sinusoidal position embeddings and interpolation for different
image sizes.
"""

import numpy as np
import torch


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True):
    """
    Generate 2D sinusoidal position embeddings.
    
    Args:
        embed_dim (int): Embedding dimension
        grid_size (int or tuple): Grid size (H, W) or single int for square
        cls_token (bool): Whether to include CLS token position
        
    Returns:
        np.ndarray: Position embeddings [grid_size*grid_size (+1), embed_dim]
    """
    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        grid_h, grid_w = grid_size
    
    grid_h_pos = np.arange(grid_h, dtype=np.float32)
    grid_w_pos = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w_pos, grid_h_pos)  # W goes first
    grid = np.stack(grid, axis=0)  # [2, H, W]
    
    grid = grid.reshape([2, 1, grid_h, grid_w])
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sinusoidal position embeddings from grid.
    
    Args:
        embed_dim (int): Embedding dimension
        pos (np.ndarray): Position grid [2, 1, H, W]
        
    Returns:
        np.ndarray: Position embeddings [H*W, embed_dim]
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # [D/2]
    
    pos = pos.reshape(-1)  # [H*W*2]
    out = np.einsum('m,d->md', pos, omega)  # [H*W*2, D/2]
    
    emb_sin = np.sin(out)  # [H*W*2, D/2]
    emb_cos = np.cos(out)  # [H*W*2, D/2]
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # [H*W*2, D]
    return emb


def interpolate_pos_embed(pos_embed, orig_size, new_size):
    """
    Interpolate position embeddings for different image sizes.
    
    Used when fine-tuning a model on different resolution than pre-training.
    
    Args:
        pos_embed (torch.Tensor): Original position embeddings
        orig_size (int): Original grid size
        new_size (int): New grid size
        
    Returns:
        torch.Tensor: Interpolated position embeddings
    """
    import torch.nn.functional as F
    
    # Extract CLS token and patch embeddings
    cls_pos_embed = pos_embed[:, 0:1, :]
    patch_pos_embed = pos_embed[:, 1:, :]
    
    # Reshape patch embeddings
    patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
    
    # Interpolate
    patch_pos_embed = F.interpolate(
        patch_pos_embed,
        size=(new_size, new_size),
        mode='bicubic',
        align_corners=False
    )
    
    # Reshape back
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, pos_embed.shape[-1])
    
    # Concatenate with CLS token
    pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)
    
    return pos_embed

