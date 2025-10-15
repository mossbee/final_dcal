"""
Multi-Head Self-Attention and Transformer Block.

Standard self-attention mechanism as used in Vision Transformer.
"""

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MSA) module.
    
    Implements the attention mechanism from "Attention is All You Need".
    Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
    
    Args:
        dim (int): Input/output dimension
        num_heads (int): Number of attention heads
        qkv_bias (bool): Whether to use bias in QKV projections
        attn_drop (float): Dropout rate for attention weights
        proj_drop (float): Dropout rate for output projection
        
    Attributes:
        num_heads: Number of attention heads
        head_dim: Dimension per head
        scale: Scaling factor (1 / sqrt(head_dim))
        qkv: Linear projection for Q, K, V
        attn_drop: Attention dropout
        proj: Output projection
        proj_drop: Projection dropout
        
    Methods:
        forward(x): Compute self-attention
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        """Initialize Multi-Head Self-Attention."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass of multi-head self-attention.
        
        Args:
            x (torch.Tensor): Input tensor [B, N, D]
            return_attention (bool): Whether to return attention weights
            
        Returns:
            torch.Tensor: Output tensor [B, N, D]
            torch.Tensor (optional): Attention weights [B, num_heads, N, N]
        """
        B, N, C = x.shape
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn_weights = attn.softmax(dim=-1)
        attn_weights_dropped = self.attn_drop(attn_weights)
        
        # Apply attention to values
        x = (attn_weights_dropped @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attention:
            return x, attn_weights
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block.
    
    Consists of:
    1. Layer Normalization
    2. Multi-Head Self-Attention
    3. Residual connection
    4. Layer Normalization
    5. Feed-Forward Network (MLP)
    6. Residual connection
    
    Optional: Stochastic Depth for regularization
    
    Args:
        dim (int): Input/output dimension
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim
        qkv_bias (bool): Use bias in QKV projections
        drop (float): Dropout rate
        attn_drop (float): Attention dropout rate
        drop_path (float): Stochastic depth rate
        act_layer: Activation function (default: GELU)
        norm_layer: Normalization layer (default: LayerNorm)
        
    Methods:
        forward(x): Forward pass through transformer block
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """Initialize Transformer block."""
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                           attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, drop=drop)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass through transformer block.
        
        Args:
            x (torch.Tensor): Input tensor [B, N, D]
            return_attention (bool): Whether to return attention weights
            
        Returns:
            torch.Tensor: Output tensor [B, N, D]
            torch.Tensor (optional): Attention weights
        """
        if return_attention:
            attn_output, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + self.drop_path(attn_output)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn_weights
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network).
    
    Two-layer MLP with GELU activation:
    MLP(x) = Linear(GELU(Linear(x)))
    
    Args:
        in_features (int): Input dimension
        hidden_features (int, optional): Hidden dimension
        out_features (int, optional): Output dimension
        act_layer: Activation function
        drop (float): Dropout rate
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        """Initialize MLP."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        """Forward pass through MLP."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

