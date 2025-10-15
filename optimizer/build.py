"""
Optimizer builder for DCAL.

Creates optimizers with task-specific settings.
"""

import torch.optim as optim


def build_optimizer(config, model):
    """
    Build optimizer based on configuration.
    
    FGVC settings:
    - Optimizer: Adam
    - Weight decay: 0.05
    - LR: 5e-4 / 512 * batch_size (scaled by batch size)
    
    ReID settings:
    - Optimizer: SGD
    - Momentum: 0.9
    - Weight decay: 1e-4
    - LR: 0.008
    
    Args:
        config: Configuration object with optimizer settings
        model (nn.Module): Model to optimize
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    # Get parameter groups with/without weight decay
    param_groups = get_parameter_groups(model, config.weight_decay)
    
    if config.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            param_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            lr=config.learning_rate,
            momentum=config.momentum if hasattr(config, 'momentum') else 0.9,
            weight_decay=config.weight_decay,
            nesterov=True
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    
    return optimizer


def get_parameter_groups(model, weight_decay):
    """
    Separate parameters into groups with/without weight decay.
    
    No weight decay for:
    - Bias terms
    - LayerNorm parameters
    - Position embeddings
    
    Args:
        model (nn.Module): Model
        weight_decay (float): Weight decay value
        
    Returns:
        list: Parameter groups for optimizer
    """
    no_decay = []
    decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No weight decay for bias, LayerNorm, and position embeddings
        if 'bias' in name or 'norm' in name or 'pos_embed' in name or 'cls_token' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]

