import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class DefaultDataset(Dataset):
    def __init__(self, N, D_in, D_out):
        self.N = N
        self.D_out = D_out
        self.D_in = D_in
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.N

class ContinuousDataset(DefaultDataset):
    def __init__(self, N, D_in, D_out):
        super(ContinuousDataset, self).__init__(N, D_in, D_out)
        self.x = np.random.random((N, D_in)).astype(np.float32)
        self.y = np.random.random((N, D_out)).astype(np.float32)
    
class BinaryDataset(DefaultDataset):
    def __init__(self, N, D_in, D_out):
        super(BinaryDataset, self).__init__(N, D_in, D_out)
        self.x = np.random.random((N, D_in)).astype(np.float32)
        self.y = np.random.randint(0, 2, (N, D_out)).astype(np.float32)
    
class CategoricalDataset(DefaultDataset):
    def __init__(self, N, D_in, D_out):
        super(CategoricalDataset, self).__init__(N, D_in, D_out)
        self.x = np.random.random((N, D_in)).astype(np.float32)
        self.y = np.random.randint(0, D_out, (N,)).astype(np.int)
        
