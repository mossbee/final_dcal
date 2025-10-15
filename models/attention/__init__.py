"""
Attention mechanisms for DCAL.

Implements Self-Attention (SA), Global-Local Cross-Attention (GLCA),
Pair-Wise Cross-Attention (PWCA), and Attention Rollout.
"""

from .self_attention import MultiHeadSelfAttention, TransformerBlock, MLP, DropPath
from .glca import GlobalLocalCrossAttention
from .pwca import PairWiseCrossAttention
from .rollout import attention_rollout, get_cls_attention_map, get_top_k_indices

__all__ = [
    'MultiHeadSelfAttention', 'TransformerBlock', 'MLP', 'DropPath',
    'GlobalLocalCrossAttention',
    'PairWiseCrossAttention',
    'attention_rollout', 'get_cls_attention_map', 'get_top_k_indices'
]

