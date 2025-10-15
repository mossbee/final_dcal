"""
Model utility functions.

Includes stochastic depth, position embeddings, and weight initialization.
"""

from .position_embed import get_2d_sincos_pos_embed, interpolate_pos_embed
from .weight_init import trunc_normal_, init_weights

__all__ = [
    'get_2d_sincos_pos_embed', 'interpolate_pos_embed',
    'trunc_normal_', 'init_weights'
]

