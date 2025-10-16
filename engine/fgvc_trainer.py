"""
Trainer for FGVC (Fine-Grained Visual Categorization) tasks.

Extends BaseTrainer with FGVC-specific training logic.
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from .trainer import BaseTrainer


class FGVCTrainer(BaseTrainer):
    """
    Trainer for fine-grained visual categorization.
    
    FGVC-specific:
    - Cross-entropy loss with uncertainty weighting
    - Top-1/Top-5 accuracy metrics
    - Stochastic depth regularization
    - Prediction = P_SA + P_GLCA (sum of probabilities)
    
    Training procedure:
    1. Forward pass through SA, GLCA, PWCA branches
    2. Compute losses for each branch
    3. Combine losses with uncertainty weighting
    4. Backward pass and optimizer step
    5. Update metrics (accuracy, loss components)
    
    Methods:
        _compute_loss(outputs, labels): Compute FGVC loss with uncertainty weighting
        _compute_accuracy(logits, labels): Compute top-1 and top-5 accuracy
    """
    
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler,
                 loss_fn, config, device, logger):
        """Initialize FGVC trainer."""
        super().__init__(model, train_loader, val_loader, optimizer, scheduler,
                        loss_fn, config, device, logger)
    
    def _train_step(self, batch):
        """
        Training step for FGVC.
        
        Args:
            batch: (images, labels) or (images, labels, pairs)
            
        Returns:
            dict: Metrics (total_loss, sa_loss, glca_loss, pwca_loss, accuracy)
        """
        # Unpack batch
        if len(batch) == 2:
            images, labels = batch
            pairs = None
        else:
            images, labels, pairs = batch
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        if pairs is not None:
            pairs = pairs.to(self.device)
        
        # Forward pass with AMP
        if self.use_amp:
            with autocast():
                outputs = self.model(images, pairs=pairs)
                total_loss, loss_dict = self._compute_loss(outputs, labels)
        else:
            outputs = self.model(images, pairs=pairs)
            total_loss, loss_dict = self._compute_loss(outputs, labels)
        
        # Scale loss by accumulation steps
        total_loss = total_loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        # Update weights only after accumulating gradients
        self.accumulation_step += 1
        if self.accumulation_step % self.gradient_accumulation_steps == 0:
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.accumulation_step = 0
        
        # Compute accuracy (combine SA + GLCA for inference)
        with torch.no_grad():
            combined_logits = outputs['sa_logits'] + outputs['glca_logits']
            acc = self._compute_accuracy(combined_logits, labels)
        
        # Collect metrics (multiply by accumulation steps to get actual loss)
        metrics = {
            'total_loss': total_loss.item() * self.gradient_accumulation_steps,
            **loss_dict,
            **acc
        }
        
        return metrics
    
    def _val_step(self, batch):
        """
        Validation step for FGVC.
        
        Args:
            batch: (images, labels)
            
        Returns:
            dict: Metrics (loss, accuracy)
        """
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass (no PWCA during validation)
        outputs = self.model(images, pairs=None)
        
        # Compute loss
        total_loss, loss_dict = self._compute_loss(outputs, labels)
        
        # Compute accuracy (combine SA + GLCA)
        combined_logits = outputs['sa_logits'] + outputs['glca_logits']
        acc = self._compute_accuracy(combined_logits, labels)
        
        # Collect metrics
        metrics = {
            'total_loss': total_loss.item(),
            **loss_dict,
            **acc
        }
        
        return metrics
    
    def _compute_loss(self, outputs, labels):
        """
        Compute FGVC loss with uncertainty weighting.
        
        L_total = 1/2 * (L_SA/exp(w1) + L_GLCA/exp(w2) + L_PWCA/exp(w3) + w1 + w2 + w3)
        
        Args:
            outputs (dict): Model outputs (sa_logits, glca_logits, pwca_logits)
            labels (torch.Tensor): Ground truth labels
            
        Returns:
            tuple: (total_loss, loss_dict)
        """
        # Compute individual losses
        sa_loss = self.loss_fn(outputs['sa_logits'], labels)
        glca_loss = self.loss_fn(outputs['glca_logits'], labels)
        
        # Get uncertainty weights
        w = self.model.uncertainty_weights
        
        # Apply uncertainty weighting
        if 'pwca_logits' in outputs:
            pwca_loss = self.loss_fn(outputs['pwca_logits'], labels)
            total_loss = 0.5 * (
                sa_loss / torch.exp(w[0]) + w[0] +
                glca_loss / torch.exp(w[1]) + w[1] +
                pwca_loss / torch.exp(w[2]) + w[2]
            )
            loss_dict = {
                'sa_loss': sa_loss.item(),
                'glca_loss': glca_loss.item(),
                'pwca_loss': pwca_loss.item(),
            }
        else:
            total_loss = 0.5 * (
                sa_loss / torch.exp(w[0]) + w[0] +
                glca_loss / torch.exp(w[1]) + w[1]
            )
            loss_dict = {
                'sa_loss': sa_loss.item(),
                'glca_loss': glca_loss.item(),
            }
        
        return total_loss, loss_dict
    
    def _compute_accuracy(self, logits, labels, topk=(1, 5)):
        """
        Compute top-k accuracy.
        
        Args:
            logits (torch.Tensor): Predicted logits
            labels (torch.Tensor): Ground truth labels
            topk (tuple): Top-k values to compute
            
        Returns:
            dict: Accuracy metrics
        """
        maxk = max(topk)
        batch_size = labels.size(0)
        
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        
        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[f'top{k}_acc'] = (correct_k.item() / batch_size) * 100.0
        
        # Use top-1 as default accuracy
        res['accuracy'] = res['top1_acc']
        
        return res

