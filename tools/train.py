"""
Training script for DCAL.

Main entry point for training models on FGVC or ReID tasks.
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import DCALModel, CONFIGS
from engine.fgvc_trainer import FGVCTrainer
from data.datasets.fgvc.cub import CUBDataset
from data.transforms import get_train_transforms, get_val_transforms
from optimizer.build import build_optimizer
from optimizer.scheduler import WarmupCosineSchedule
from loss.cross_entropy import CrossEntropyLoss
from utils.logger import setup_logger
from utils.misc import set_seed, create_dirs
from configs.fgvc.cub import CUBConfig


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train DCAL model')
    
    # Config
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--task', type=str, choices=['fgvc', 'reid'],
                        help='Task type (overrides config)')
    
    # Data
    parser.add_argument('--data-root', type=str,
                        help='Dataset root directory')
    parser.add_argument('--batch-size', type=int,
                        help='Batch size (overrides config)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model
    parser.add_argument('--model', type=str,
                        help='Model architecture (e.g., ViT-B_16)')
    parser.add_argument('--pretrained', type=str,
                        help='Path to pretrained weights')
    
    # Training
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float,
                        help='Learning rate')
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint to resume from')
    
    # Hardware
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed training')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='experiments/logs',
                        help='Directory for logs')
    parser.add_argument('--checkpoint-dir', type=str, default='experiments/checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Logging interval (iterations)')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    """
    Main training function.
    
    Steps:
    1. Parse arguments and load config
    2. Set up logging and random seeds
    3. Create datasets and data loaders
    4. Build model, optimizer, scheduler
    5. Create trainer
    6. Run training loop
    7. Save final model
    """
    args = parse_args()
    
    # Load config
    # For now, we'll use CUBConfig as an example
    # In practice, this would be loaded from args.config
    config = CUBConfig()
    
    # Override config with command-line arguments
    if args.data_root:
        config.data_root = args.data_root
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.pretrained:
        config.pretrained_path = args.pretrained
    
    config.log_dir = args.log_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.log_interval = args.log_interval
    
    # Create directories
    create_dirs([config.log_dir, config.checkpoint_dir])
    
    # Set up logging
    logger = setup_logger('DCAL', config.log_dir)
    logger.info(f"Config: {config}")
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_transform = get_train_transforms(config.task, config.img_size, config.crop_size)
    val_transform = get_val_transforms(config.task, config.img_size, config.crop_size)
    
    train_dataset = CUBDataset(config.data_root, split='train', transform=train_transform)
    val_dataset = CUBDataset(config.data_root, split='test', transform=val_transform)
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Build model
    logger.info("Building model...")
    vit_config = CONFIGS[config.model_type]
    model = DCALModel(
        config=vit_config,
        img_size=config.crop_size,
        num_classes=config.num_classes,
        glca_top_r=config.glca_top_r,
        glca_layer_idx=config.glca_layer_idx,
        use_pwca=config.pwca_enabled,
        task=config.task,
        pretrained_path=config.pretrained_path if hasattr(config, 'pretrained_path') else None
    )
    model = model.to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Build optimizer
    optimizer = build_optimizer(config, model)
    logger.info(f"Optimizer: {optimizer}")
    
    # Build scheduler
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=config.warmup_epochs * len(train_loader),
        t_total=config.epochs * len(train_loader)
    )
    
    # Build loss function
    loss_fn = CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = FGVCTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        config=config,
        device=device,
        logger=logger
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()

