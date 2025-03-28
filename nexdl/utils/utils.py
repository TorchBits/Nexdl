import numpy as np
import pickle
from nexdl.nex import nx
import os
import gzip
import urllib.request

class Dataset:
    """Abstract Dataset class."""
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError

class DataLoader:
    """Dataloader for batching and shuffling."""
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))  # Fix here
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            if isinstance(batch[0], tuple):  # If dataset returns (data, label)
                batch = list(zip(*batch))  # Unzip into (data_batch, label_batch)
            yield [nx.stack(batch_part, axis=0) for batch_part in batch]  # Stack into batch

def save_model(parameters, filepath):
    """Save model parameters to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump([param.copy() for param in parameters], f)

def load_model(parameters, filepath):
    """Load model parameters from a file."""
    with open(filepath, 'rb') as f:
        saved_params = pickle.load(f)
        for param, saved_param in zip(parameters, saved_params):
            param[...] = saved_param

def amp_convert_to_fp16(parameters):
    """Convert parameters to float16 for mixed precision training."""
    for param in parameters:
        param[...] = param.astype(nx.float16)

def amp_convert_to_fp32(parameters):
    """Convert parameters back to float32."""
    for param in parameters:
        param[...] = param.astype(nx.float32)




