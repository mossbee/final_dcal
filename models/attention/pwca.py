"""
Pair-Wise Cross-Attention (PWCA) module.

Regularizes attention learning by treating paired images as distractors,
increasing training difficulty and reducing overfitting.
"""

import torch
import torch.nn as nn


class PairWiseCrossAttention(nn.Module):
    """
    Pair-Wise Cross-Attention module.
    
    Process:
    1. Sample image pairs (I1, I2) from training batch
    2. Compute Q, K, V for both images separately
    3. Concatenate K and V: K_c = [K1; K2], V_c = [V1; V2]
    4. Compute attention: Q1 @ K_c @ V_c
    5. Attention is "contaminated" by the paired image (distractor)
    
    Formula:
        PWCA(Q1, K_c, V_c) = softmax(Q1 K_c^T / sqrt(d)) V_c
    
    where:
        K_c = [K1; K2] ∈ R^{(2N+2) × d}
        V_c = [V1; V2] ∈ R^{(2N+2) × d}
    
    This increases training difficulty and prevents overfitting to
    sample-specific features. PWCA is ONLY used during training.
    
    Args:
        dim (int): Input dimension
        num_heads (int): Number of attention heads
        qkv_bias (bool): Use bias in QKV projections
        attn_drop (float): Attention dropout rate
        proj_drop (float): Projection dropout rate
        
    Note:
        PWCA shares weights with SA branch. During inference, PWCA is
        disabled and only SA + GLCA are used.
        
    Methods:
        forward(x1, x2): Compute PWCA with image pairs
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        """Initialize Pair-Wise Cross-Attention."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projections - these will be shared with SA branch in practice
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x1, x2, return_attention=False):
        """
        Forward pass of PWCA with image pairs.
        
        Args:
            x1 (torch.Tensor): Target image embeddings [B, N+1, D]
            x2 (torch.Tensor): Distractor image embeddings [B, N+1, D]
            return_attention (bool): Whether to return attention weights
            
        Returns:
            torch.Tensor: Output embeddings for x1 [B, N+1, D]
            torch.Tensor (optional): PWCA attention weights [B, heads, N+1, 2N+2]
        """
        B, N_plus_1, C = x1.shape
        
        # Compute Q, K, V for both images
        # For x1 (target image)
        qkv1 = self.qkv(x1).reshape(B, N_plus_1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]  # [B, num_heads, N+1, head_dim]
        
        # For x2 (distractor image)
        qkv2 = self.qkv(x2).reshape(B, N_plus_1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        _, k2, v2 = qkv2[0], qkv2[1], qkv2[2]  # We don't need q2
        
        # Concatenate keys and values from both images
        k_concat = torch.cat([k1, k2], dim=2)  # [B, num_heads, 2*(N+1), head_dim]
        v_concat = torch.cat([v1, v2], dim=2)  # [B, num_heads, 2*(N+1), head_dim]
        
        # Compute cross-attention: Q1 with concatenated K and V
        attn = (q1 @ k_concat.transpose(-2, -1)) * self.scale  # [B, num_heads, N+1, 2*(N+1)]
        attn_weights = attn.softmax(dim=-1)
        attn_weights_dropped = self.attn_drop(attn_weights)
        
        # Apply attention to concatenated values
        out = (attn_weights_dropped @ v_concat).transpose(1, 2).reshape(B, N_plus_1, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        if return_attention:
            return out, attn_weights
        return out

