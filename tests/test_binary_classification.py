import torcheras

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from datasets import BinaryDatasetMultiTasks, BinaryDataset
from models import TwoLayerNet

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 100000, 4, 4, 3
batch_size = 64
epochs = 200

# test_type = 'single task classification'
test_type = 'multi tasks classification'

if test_type == 'single task classification':
    # Create dataset
    train_dataset = BinaryDataset(N, D_in, 1)
    test_dataset = BinaryDataset(int(N*0.25), D_in, 1)
    model = TwoLayerNet(D_in, H, 1)

elif test_type == 'multi tasks classification':
    train_dataset = BinaryDatasetMultiTasks(N, D_in, D_out)
    test_dataset = BinaryDatasetMultiTasks(int(N * 0.25), D_in, D_out)
    model = TwoLayerNet(D_in, H, D_out)
    
# Create dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

parameters = model.parameters()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(parameters, lr=1e-4)

# torcheras
model = torcheras.Model(model, 'test_records')

# compile and train
if test_type == 'single task classification':
    model.compile(criterion, optimizer, metrics = ['binary_acc'])
elif test_type == 'multi tasks classification':
    multi_tasks = {'outputs': ['output_a', 'output_b', 'output_c']}
    model.compile(criterion, optimizer, metrics = ['binary_acc'], multi_tasks = multi_tasks)    

model.fit(train_dataloader, test_dataloader, epochs)
