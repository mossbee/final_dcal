"""
Metric computation utilities.

Provides accuracy, mAP, and other evaluation metrics.
"""

import torch
import numpy as np


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking metrics during training (loss, accuracy, etc.).
    
    Attributes:
        val: Current value
        avg: Average value
        sum: Sum of all values
        count: Count of values
        
    Methods:
        reset(): Reset all values
        update(val, n): Update with new value(s)
    """
    
    def __init__(self, name=''):
        """Initialize average meter."""
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        Update meter with new value.
        
        Args:
            val: New value
            n: Number of samples (for averaging)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy.
    
    Args:
        output (torch.Tensor): Model predictions [B, num_classes]
        target (torch.Tensor): Ground truth labels [B]
        topk (tuple): Top-k values to compute
        
    Returns:
        list: Top-k accuracies
    """
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def compute_distance_matrix(query_features, gallery_features, metric='euclidean'):
    """
    Compute distance matrix between query and gallery features.
    
    Args:
        query_features (np.ndarray): Query features [N_q, D]
        gallery_features (np.ndarray): Gallery features [N_g, D]
        metric (str): Distance metric ('euclidean' or 'cosine')
        
    Returns:
        np.ndarray: Distance matrix [N_q, N_g]
    """
    if metric == 'euclidean':
        # Compute euclidean distance
        distmat = np.zeros((query_features.shape[0], gallery_features.shape[0]))
        for i in range(query_features.shape[0]):
            distmat[i] = np.sqrt(np.sum((gallery_features - query_features[i])**2, axis=1))
    elif metric == 'cosine':
        # Normalize features
        query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
        gallery_features = gallery_features / np.linalg.norm(gallery_features, axis=1, keepdims=True)
        # Compute cosine distance (1 - cosine similarity)
        distmat = 1 - np.dot(query_features, gallery_features.T)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    return distmat


def eval_cmc_map(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """
    Evaluate CMC (Cumulative Matching Characteristics) and mAP.
    
    Args:
        distmat (np.ndarray): Distance matrix [N_q, N_g]
        q_pids (np.ndarray): Query person IDs
        g_pids (np.ndarray): Gallery person IDs
        q_camids (np.ndarray): Query camera IDs
        g_camids (np.ndarray): Gallery camera IDs
        max_rank (int): Maximum rank for CMC
        
    Returns:
        tuple: (mAP, CMC curve)
    """
    num_q, num_g = distmat.shape
    
    if num_g < max_rank:
        max_rank = num_g
    
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # Compute CMC curve
    all_cmc = []
    all_AP = []
    num_valid_q = 0
    
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        
        # Remove gallery samples with same pid and camid
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        
        # Compute CMC curve
        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue
        
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1
        
        # Compute AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    
    assert num_valid_q > 0, "No valid query"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    
    return mAP, all_cmc

