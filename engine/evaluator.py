"""
Base evaluator class for DCAL.

Provides common evaluation functionality for both FGVC and ReID tasks.
"""

import torch


class BaseEvaluator:
    """
    Base evaluator class with common evaluation logic.
    
    Handles:
    - Evaluation loop
    - Inference mode (no PWCA)
    - Metric computation
    - Result aggregation
    
    Args:
        model (nn.Module): DCAL model
        data_loader (DataLoader): Evaluation data loader
        device: Device for evaluation
        logger: Logger instance
        
    Attributes:
        model: DCAL model
        data_loader: Data loader
        device: Evaluation device
        logger: Logger
        
    Methods:
        evaluate(): Run evaluation
        _inference_step(batch): Single inference step
        _compute_metrics(predictions, labels): Compute evaluation metrics
    """
    
    def __init__(self, model, data_loader, device, logger=None):
        """Initialize base evaluator."""
        pass
    
    def evaluate(self):
        """
        Run evaluation on dataset.
        
        Returns:
            dict: Evaluation metrics
        """
        pass
    
    def _inference_step(self, batch):
        """
        Single inference step.
        
        Args:
            batch: Batch of data
            
        Returns:
            dict: Predictions
        """
        pass
    
    def _compute_metrics(self, all_predictions, all_labels):
        """
        Compute evaluation metrics.
        
        Args:
            all_predictions: All model predictions
            all_labels: All ground truth labels
            
        Returns:
            dict: Metrics
        """
        pass

