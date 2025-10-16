"""
Base configuration class for DCAL.

Contains common settings shared across all tasks and datasets.
Task-specific and dataset-specific configurations should inherit from this base.
"""


class BaseConfig:
    """
    Base configuration class with default settings.
    
    This class contains all the common hyperparameters and settings that can be
    shared across different tasks (FGVC, ReID) and datasets.
    
    Attributes:
        # Model Architecture
        model_type (str): Type of backbone model (e.g., 'ViT-B_16', 'DeiT-Tiny')
        pretrained_dir (str): Path to pretrained weights
        num_classes (int): Number of output classes
        
        # DCAL-specific
        num_sa_blocks (int): Number of self-attention blocks (default: 12)
        num_glca_blocks (int): Number of GLCA blocks (default: 1)
        num_pwca_blocks (int): Number of PWCA blocks (default: 12)
        glca_top_r (float): Top-R ratio for GLCA local query selection
        glca_layer_idx (int): Which SA layer to branch GLCA from (default: 11)
        pwca_enabled (bool): Whether to use PWCA during training
        share_pwca_weights (bool): Share weights between SA and PWCA branches
        
        # Data
        data_root (str): Root directory of dataset
        img_size (int): Input image size
        crop_size (int): Random crop size for training
        
        # Training
        task (str): Task type ('fgvc' or 'reid')
        optimizer (str): Optimizer type ('adam' or 'sgd')
        learning_rate (float): Initial learning rate
        lr_scheduler (str): Learning rate scheduler type
        weight_decay (float): Weight decay for optimizer
        epochs (int): Total training epochs
        warmup_epochs (int): Number of warmup epochs
        batch_size (int): Batch size for training
        num_workers (int): Number of data loading workers
        
        # Loss
        use_uncertainty_weighting (bool): Use uncertainty-based loss weighting
        label_smoothing (float): Label smoothing factor
        
        # Regularization
        stochastic_depth_prob (float): Stochastic depth drop probability
        
        # Logging and Checkpointing
        log_dir (str): Directory for logs
        checkpoint_dir (str): Directory for checkpoints
        log_interval (int): Logging interval in iterations
        eval_interval (int): Evaluation interval in epochs
        save_interval (int): Checkpoint saving interval in epochs
        
        # Distributed Training
        world_size (int): Number of distributed processes
        dist_backend (str): Distributed backend
        dist_url (str): Distributed init URL
    """
    
    def __init__(self):
        # Model architecture
        self.model_type = 'ViT-B_16'
        self.pretrained_dir = 'refs/ViT-B_16.npz'
        self.num_classes = 200
        
        # DCAL-specific
        self.num_sa_blocks = 12
        self.num_glca_blocks = 1
        self.num_pwca_blocks = 12
        self.glca_top_r = 0.1
        self.glca_layer_idx = 11  # Which SA layer to branch GLCA from (0-indexed)
        self.pwca_enabled = True
        self.share_pwca_weights = True
        
        # Data
        self.data_root = 'data'
        self.img_size = 448
        self.crop_size = 448
        
        # Training
        self.task = 'fgvc'
        self.optimizer = 'adam'
        self.learning_rate = 5e-4
        self.lr_scheduler = 'cosine'
        self.weight_decay = 0.05
        self.momentum = 0.9
        self.epochs = 100
        self.warmup_epochs = 5
        self.batch_size = 16
        self.num_workers = 4
        
        # Loss
        self.use_uncertainty_weighting = True
        self.label_smoothing = 0.0
        
        # Regularization
        self.stochastic_depth_prob = 0.1
        
        # Logging and checkpointing
        self.log_dir = 'experiments/logs'
        self.checkpoint_dir = 'experiments/checkpoints'
        self.log_interval = 10
        self.eval_interval = 1
        self.save_interval = 10
        
        # Distributed training
        self.world_size = 1
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'
    
    def get_config_dict(self):
        """Return configuration as dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def update_from_dict(self, config_dict):
        """Update configuration from dictionary."""
        for k, v in config_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
    
    def __repr__(self):
        """String representation of configuration."""
        config_str = f"{self.__class__.__name__}(\n"
        for k, v in self.get_config_dict().items():
            config_str += f"  {k}: {v}\n"
        config_str += ")"
        return config_str

