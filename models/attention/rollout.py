"""
Attention Rollout implementation.

Computes accumulated attention scores across Transformer layers to identify
high-response regions for GLCA.
"""

import torch
import torch.nn as nn


def attention_rollout(attentions, head_fusion='mean', discard_ratio=0.0):
    """
    Compute attention rollout from attention weights.
    
    Following the paper equation:
    S_hat_i = S_bar_i ⊗ S_bar_{i-1} ⊗ ... ⊗ S_bar_1
    where S_bar_l = 0.5 * S_l + 0.5 * I
    
    Args:
        attentions (list): List of attention weight tensors from each layer
                          Each tensor: [B, num_heads, N+1, N+1]
        head_fusion (str): Method to fuse attention heads ('mean', 'max', 'min')
        discard_ratio (float): Fraction of lowest attention weights to discard (optional)
    
    Returns:
        torch.Tensor: Accumulated attention rollout map [B, N+1, N+1]
    """
    B = attentions[0].shape[0]
    num_tokens = attentions[0].shape[-1]
    device = attentions[0].device
    
    # Use float32 for stability but save memory
    dtype = torch.float32
    
    # Create identity matrix once and reuse
    I = torch.eye(num_tokens, device=device, dtype=dtype).unsqueeze(0)  # [1, N+1, N+1]
    
    # Initialize result as identity matrix for each sample in batch
    result = I.expand(B, -1, -1).clone()  # [B, N+1, N+1]
    
    for attention in attentions:
        # Detach attention to save memory (rollout doesn't need gradients)
        attention = attention.detach().to(dtype=dtype)
        # Fuse attention heads: [B, num_heads, N+1, N+1] -> [B, N+1, N+1]
        if head_fusion == "mean":
            attention_heads_fused = attention.mean(dim=1)
        elif head_fusion == "max":
            attention_heads_fused = attention.max(dim=1)[0]
        elif head_fusion == "min":
            attention_heads_fused = attention.min(dim=1)[0]
        else:
            raise ValueError(f"Attention head fusion type '{head_fusion}' not supported")
        
        # Optional: Drop the lowest attentions
        if discard_ratio > 0:
            # Flatten for topk operation
            flat = attention_heads_fused.view(B, -1)  # [B, (N+1)*(N+1)]
            num_keep = int(flat.size(-1) * (1 - discard_ratio))
            _, indices = flat.topk(num_keep, dim=-1, largest=True)
            
            # Create mask
            mask = torch.zeros_like(flat)
            mask.scatter_(1, indices, 1)
            mask = mask.view(B, num_tokens, num_tokens)
            
            # Don't drop class token attention
            mask[:, 0, :] = 1
            mask[:, :, 0] = 1
            
            attention_heads_fused = attention_heads_fused * mask
        
        # Add residual connection (identity): S_bar = 0.5 * S + 0.5 * I
        # Use in-place operations to save memory
        attention_heads_fused.mul_(0.5).add_(I.expand(B, -1, -1), alpha=0.5)
        
        # Normalize rows to sum to 1
        attention_heads_fused.div_(attention_heads_fused.sum(dim=-1, keepdim=True))
        
        # Multiply with accumulated result
        result = torch.bmm(attention_heads_fused, result)
    
    return result


def get_cls_attention_map(rollout_map):
    """
    Extract CLS token's attention to all patches.
    
    Args:
        rollout_map (torch.Tensor): Rollout map [B, N+1, N+1]
    
    Returns:
        torch.Tensor: CLS attention to patches [B, N] (excluding CLS token)
    """
    # First row is CLS token's attention to all tokens
    # Exclude first column (CLS to CLS) to get CLS to patches
    cls_attention = rollout_map[:, 0, 1:]  # [B, N]
    return cls_attention


def get_top_k_indices(cls_attention, top_r=0.1):
    """
    Get indices of top-R patches with highest attention scores.
    
    Args:
        cls_attention (torch.Tensor): CLS attention to patches [B, N]
        top_r (float): Ratio of top patches to select (e.g., 0.1 for 10%)
    
    Returns:
        torch.Tensor: Indices of top-R patches [B, R]
    """
    B, N = cls_attention.shape
    R = max(1, int(N * top_r))
    
    # Get top-R indices
    _, top_indices = torch.topk(cls_attention, R, dim=1)  # [B, R]
    
    return top_indices