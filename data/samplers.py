"""
Data samplers for ReID tasks.

Implements RandomIdentitySampler for P-K sampling strategy where we sample
P identities and K images per identity in each batch.
"""

import numpy as np
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Random identity sampler for ReID tasks.
    
    Samples P identities and K instances per identity for each batch.
    This is the standard P-K sampling strategy used in ReID.
    
    Example:
        If P=16 and K=4, each batch will have 64 images containing
        16 different identities with 4 images each.
    
    Args:
        data_source (Dataset): ReID dataset
        num_instances (int): K - number of instances per identity
        batch_size (int): Total batch size (should be P * K)
        
    Attributes:
        data_source: Dataset to sample from
        num_instances: Number of instances (K) per identity
        num_pids_per_batch: Number of identities (P) per batch
        index_dic: Dictionary mapping pid to list of sample indices
        pids: List of all person/vehicle IDs
        
    Methods:
        __iter__(): Generate batch indices
        __len__(): Return number of batches
    """
    
    def __init__(self, data_source, batch_size, num_instances):
        """
        Initialize RandomIdentitySampler.
        
        Args:
            data_source: Dataset with 'pids' attribute (list of identity labels)
            batch_size (int): Total batch size (P * K)
            num_instances (int): K - images per identity
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        
        # Build index dictionary: pid -> [idx1, idx2, ...]
        self.index_dic = {}
        for index, pid in enumerate(data_source.pids):
            if pid not in self.index_dic:
                self.index_dic[pid] = []
            self.index_dic[pid].append(index)
        
        self.pids = list(self.index_dic.keys())
        
        # Estimate number of examples per epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
    
    def __iter__(self):
        """
        Generate indices for P-K sampling.
        
        For each batch:
        1. Randomly sample P identities
        2. For each identity, randomly sample K instances
        3. Yield batch indices
        
        Yields:
            list: Indices for one batch
        """
        batch_idxs_dict = {}
        
        # For each identity, create list of indices
        for pid in self.pids:
            idxs = np.copy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                # Repeat indices if less than num_instances
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            np.random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid] = batch_idxs
                    batch_idxs = []
        
        # Sample P identities per batch
        avai_pids = np.copy(self.pids)
        final_idxs = []
        
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid]
                final_idxs.extend(batch_idxs)
            avai_pids = np.setdiff1d(avai_pids, selected_pids)
        
        return iter(final_idxs)
    
    def __len__(self):
        """Return number of batches per epoch."""
        return self.length // self.batch_size

