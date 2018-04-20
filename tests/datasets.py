import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class DefaultDataset(Dataset):
    def __init__(self, N):
        self.N = N
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.N

class ContinuousDataset(DefaultDataset):
    def __init__(self, N, D_in, D_out):
        super(ContinuousDataset, self).__init__(N)
        self.x = np.random.random((N, D_in)).astype(np.float32)
        self.y = self.x + 1
    
class BinaryDataset(DefaultDataset):
    def __init__(self, N, D_in, D_out):
        super(BinaryDataset, self).__init__(N)
        self.x = np.random.random((N, D_in)).astype(np.float32)
        self.y = np.random.randint(0,2, (N, 1)).astype(np.float32)
    
class BinaryDatasetMultiTasks(DefaultDataset):
    def __init__(self, N, D_in, D_out):
        super(BinaryDatasetMultiTasks, self).__init__(N)
        self.x = np.random.random((N, D_in)).astype(np.float32)
        self.y = np.random.randint(0,2, (N, D_out)).astype(np.float32)
    
# class CategoricalDataset(DefaultDataset):
#     def __init__(self, N, D_in, D_out):
#         super(CategoricalDataset, self).__init__(N, D_in, D_out)
#         self.x = np.random.random((N, D_in)).astype(np.float32)
#         self.y = np.random.randint(0, D_out, (N, 1)).astype(np.int)
        
class ContinuousDatasetMultiTasks(DefaultDataset):
    def __init__(self, N, D_in, D_out1, D_out2):
        super(CategoricalDatasetMultiTasks, self).__init__(N)
        
        self.x = np.random.random((N, D_in)).astype(np.float32)
        
        self.y1 = self.x + 1
        self.y2 = self.x + 2
        
        self.y = np.concatenate((self.y1.reshape(N,1), self.y2.reshape(N,1)), 1)

class CategoricalDatasetMultiTasks(DefaultDataset):
    def __init__(self, N, D_in, D_out1, D_out2):
        super(CategoricalDatasetMultiTasks, self).__init__(N)
        
        self.x = np.random.random((N, D_in)).astype(np.float32)
        
        self.y1 = (np.round(self.x.sum(1)) % D_out1).astype(np.int)
        self.y2 = (np.round(self.x.sum(1)) % D_out2).astype(np.int)
        
        self.y = np.concatenate((self.y1.reshape(N,1), self.y2.reshape(N,1)), 1)