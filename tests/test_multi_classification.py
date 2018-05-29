import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim

import torcheras
from models import LeNet, MultiTasksClassification
from datasets import CategoricalDatasetMultiTasks

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

batch_size = 100

N = 10000
D_in = 4
H1 = 64
H2 = 64
D_out1 = 2
D_out2 = 3

crossEntropy = nn.CrossEntropyLoss()

def multiCrossEntropy(y_pred, y_true):
    loss = 0
    for i, y_p in enumerate(list(y_pred)):
        loss = loss + crossEntropy(y_p, y_true[:, i])
        
    loss = loss / y_pred[0].shape[0]
    return loss

test_type = 'single task classification'
# test_type = 'multi tasks classification'

if test_type == 'single task classification':
    
    root = './data'
    download = True  # download MNIST dataset or not

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans)
      
    model = LeNet()
    model = torcheras.Model(model, logdir='test_records')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    model.compile(criterion, optimizer, metrics = ['top1', 'top2'], device = device)

elif test_type == 'multi tasks classification':
    
    train_set = CategoricalDatasetMultiTasks(N, D_in, D_out1, D_out2)
    test_set = CategoricalDatasetMultiTasks(int(N * 0.25), D_in ,D_out1, D_out2)
    
    model = MultiTasksClassification(D_in, H1, H2, D_out1, D_out2)
    model = torcheras.Model(model, logdir='test_records')
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    model.compile(multiCrossEntropy, optimizer, metrics = ['top1', 'top2'], multi_tasks = ['output_a', 'output_b'], device = device)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

model.fit(train_loader, test_loader, epochs = 200)