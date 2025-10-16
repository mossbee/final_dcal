"""
Global-Local Cross-Attention (GLCA) module.

Enhances interaction between global images and local high-response regions
by selecting top-R discriminative patches based on attention rollout.
"""

import torch
import torch.nn as nn
from .rollout import attention_rollout, get_cls_attention_map, get_top_k_indices


class GlobalLocalCrossAttention(nn.Module):
    """
    Global-Local Cross-Attention module.
    
    Process:
    1. Use attention rollout to compute accumulated attention from CLS token
    2. Select top-R patches with highest attention scores as local queries
    3. Compute cross-attention: Q_local @ K_global @ V_global
    4. Only selected local queries interact with all global key-values
    
    Formula:
        GLCA(Q^l, K^g, V^g) = softmax(Q^l K^{gT} / sqrt(d)) V^g
    
    where:
        Q^l: Selected local queries (top-R patches)
        K^g, V^g: Global keys and values (all patches)
    
    Args:
        dim (int): Input dimension
        num_heads (int): Number of attention heads
        top_r (float): Ratio of top patches to select (e.g., 0.1 for 10%)
        qkv_bias (bool): Use bias in QKV projections
        attn_drop (float): Attention dropout rate
        proj_drop (float): Projection dropout rate
        
    Attributes:
        num_heads: Number of attention heads
        head_dim: Dimension per head
        top_r: Top-R selection ratio
        q_proj: Query projection (for local queries)
        kv_proj: Key-Value projection (for global KV)
        attn_drop: Attention dropout
        proj: Output projection
        
    Methods:
        forward(x, attention_weights): Compute GLCA
        select_top_patches(attention_rollout_map): Select top-R patches
    """
    
    def __init__(self, dim, num_heads=8, top_r=0.1, qkv_bias=False, 
                 attn_drop=0., proj_drop=0.):
        """Initialize Global-Local Cross-Attention."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.top_r = top_r
        
        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, sa_attention_weights, return_attention=False):
        """
        Forward pass of GLCA.
        
        Args:
            x (torch.Tensor): Input embeddings [B, N+1, D] (with CLS token)
            sa_attention_weights (list): Attention weights from SA layers
            return_attention (bool): Whether to return attention weights
            
        Returns:
            torch.Tensor: Output embeddings [B, R, D] for top-R local patches
            torch.Tensor (optional): GLCA attention weights
        """
        B, N_plus_1, C = x.shape
        N = N_plus_1 - 1  # Number of patches (excluding CLS)
        
        # Step 1: Compute attention rollout (detach to save memory)
        # Note: attention_rollout already detaches internally
        rollout_map = attention_rollout(sa_attention_weights, head_fusion='mean')
        rollout_map = rollout_map.detach()  # Ensure no gradients are tracked
        
        # Step 2: Get CLS attention to patches and select top-R
        cls_attention = get_cls_attention_map(rollout_map)  # [B, N]
        top_indices = get_top_k_indices(cls_attention, self.top_r)  # [B, R]
        R = top_indices.shape[1]
        
        # Step 3: Select top-R local queries (excluding CLS, so add 1 to indices)
        # Indices are for patches, need to account for CLS token at position 0
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, R)
        top_indices_with_cls = top_indices + 1  # Shift by 1 for CLS token
        
        # Extract top-R embeddings as local queries
        x_local = x[batch_indices, top_indices_with_cls]  # [B, R, D]
        
        # Step 4: Compute Q, K, V
        q_local = self.q_proj(x_local)  # [B, R, D]
        k_global = self.k_proj(x)  # [B, N+1, D]
        v_global = self.v_proj(x)  # [B, N+1, D]
        
        # Reshape for multi-head attention
        q_local = q_local.reshape(B, R, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_global = k_global.reshape(B, N_plus_1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_global = v_global.reshape(B, N_plus_1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # q_local: [B, num_heads, R, head_dim]
        # k_global, v_global: [B, num_heads, N+1, head_dim]
        
        # Step 5: Compute cross-attention
        attn = (q_local @ k_global.transpose(-2, -1)) * self.scale  # [B, num_heads, R, N+1]
        attn_weights = attn.softmax(dim=-1)
        attn_weights_dropped = self.attn_drop(attn_weights)
        
        # Apply attention to values
        out = (attn_weights_dropped @ v_global).transpose(1, 2).reshape(B, R, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        if return_attention:
            return out, attn_weights, top_indices
        return out

