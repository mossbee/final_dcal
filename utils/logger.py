"""
Logging utilities for DCAL.

Provides console and TensorBoard logging.
"""

import logging
import sys
from torch.utils.tensorboard import SummaryWriter


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up logger with console and optional file output.
    
    Args:
        name (str): Logger name
        log_file (str, optional): Path to log file
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TensorboardLogger:
    """
    TensorBoard logger wrapper.
    
    Provides convenient methods for logging scalars, images, and histograms
    to TensorBoard during training.
    
    Args:
        log_dir (str): Directory for TensorBoard logs
        
    Attributes:
        writer: TensorBoard SummaryWriter
        
    Methods:
        log_scalar(tag, value, step): Log scalar value
        log_scalars(tag, value_dict, step): Log multiple scalars
        log_image(tag, image, step): Log image
        log_histogram(tag, values, step): Log histogram
        close(): Close writer
    """
    
    def __init__(self, log_dir):
        """Initialize TensorBoard logger."""
        import os
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
    
    def log_scalar(self, tag, value, step):
        """Log scalar value."""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag, value_dict, step):
        """Log multiple related scalars."""
        self.writer.add_scalars(tag, value_dict, step)
    
    def log_image(self, tag, image, step):
        """Log image."""
        self.writer.add_image(tag, image, step)
    
    def log_histogram(self, tag, values, step):
        """Log histogram of values."""
        self.writer.add_histogram(tag, values, step)
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()

