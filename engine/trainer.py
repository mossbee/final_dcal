"""
Base trainer class for DCAL.

Provides common training functionality for both FGVC and ReID tasks.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


class BaseTrainer:
    """
    Base trainer class with common training logic.
    
    Handles:
    - Training loop (epochs, iterations)
    - Forward/backward passes
    - Optimizer steps
    - Learning rate scheduling
    - Gradient accumulation
    - Mixed precision training (AMP)
    - Checkpoint saving/loading
    - Logging (TensorBoard, console)
    - Distributed training (DDP)
    
    Args:
        model (nn.Module): DCAL model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        loss_fn: Loss function
        config: Configuration object
        device: Device to train on
        logger: Logger instance
        
    Attributes:
        model: DCAL model
        optimizer: Optimizer
        scheduler: LR scheduler
        loss_fn: Loss function
        train_loader, val_loader: Data loaders
        config: Config object
        device: Training device
        logger: Logger
        scaler: AMP gradient scaler
        current_epoch: Current epoch number
        global_step: Global training step
        best_metric: Best validation metric
        
    Methods:
        train(): Main training loop
        train_one_epoch(): Train for one epoch
        validate(): Validate on validation set
        save_checkpoint(): Save model checkpoint
        load_checkpoint(): Load model checkpoint
        _train_step(): Single training step (forward + backward)
        _update_metrics(): Update training metrics
    """
    
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler,
                 loss_fn, config, device, logger):
        """Initialize base trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.logger = logger
        
        # Mixed precision training
        self.use_amp = getattr(config, 'use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
    
    def train(self):
        """
        Main training loop.
        
        Iterates through epochs, trains and validates, saves checkpoints.
        """
        self.logger.info(f"Starting training for {self.config.epochs} epochs")
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            self.logger.info(f"\nEpoch {epoch+1}/{self.config.epochs}")
            
            # Train for one epoch
            train_metrics = self.train_one_epoch()
            self.logger.info(f"Train - {self._format_metrics(train_metrics)}")
            
            # Validate
            val_metrics = self.validate()
            self.logger.info(f"Val - {self._format_metrics(val_metrics)}")
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save checkpoint
            checkpoint_path = f"{self.config.checkpoint_dir}/epoch_{epoch+1}.pth"
            self.save_checkpoint(checkpoint_path)
            
            # Save best model
            current_metric = val_metrics.get('accuracy', val_metrics.get('mAP', 0))
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                best_path = f"{self.config.checkpoint_dir}/best.pth"
                self.save_checkpoint(best_path, is_best=True)
                self.logger.info(f"New best model saved with metric: {current_metric:.4f}")
    
    def train_one_epoch(self):
        """
        Train for one epoch.
        
        Returns:
            dict: Training metrics (loss, accuracy, etc.)
        """
        self.model.train()
        total_metrics = {}
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Training step
            metrics = self._train_step(batch)
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0
                total_metrics[key] += value
            
            self.global_step += 1
            
            # Log periodically
            if (batch_idx + 1) % self.config.log_interval == 0:
                self.logger.info(
                    f"Batch [{batch_idx+1}/{len(self.train_loader)}] - {self._format_metrics(metrics)}"
                )
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= len(self.train_loader)
        
        return total_metrics
    
    def validate(self):
        """
        Validate on validation set.
        
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        total_metrics = {}
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Get predictions
                metrics = self._val_step(batch)
                
                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0
                    total_metrics[key] += value
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= len(self.val_loader)
        
        return total_metrics
    
    def _train_step(self, batch):
        """
        Single training step.
        
        Args:
            batch: Batch of data (images, labels)
            
        Returns:
            dict: Step metrics (loss, etc.)
        """
        # To be implemented by subclasses
        raise NotImplementedError
    
    def _val_step(self, batch):
        """
        Single validation step.
        
        Args:
            batch: Batch of data
            
        Returns:
            dict: Metrics for this step
        """
        # To be implemented by subclasses
        raise NotImplementedError
    
    def save_checkpoint(self, filepath, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            filepath (str): Path to save checkpoint
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """
        Load model checkpoint.
        
        Args:
            filepath (str): Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
    
    def _format_metrics(self, metrics):
        """Format metrics for logging."""
        return ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

