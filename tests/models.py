import torch
import torcheras
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TwoLayerNet(torcheras.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred
    
class LeNet(torcheras.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def name(self):
        return "LeNet"
    
    
class MultiTasksClassification(torcheras.Module):
    def __init__(self, D_in, H1, H2, D_out1, D_out2):
        super(MultiTasksClassification, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        
        self.classifier1 = torch.nn.Linear(H2, D_out1)
        self.classifier2 = torch.nn.Linear(H2, D_out2)
        
    def forward(self, x):
        x = self.linear1(x).clamp(min = 0)
        x = self.linear2(x).clamp(min = 0)
        
        y1 = self.classifier1(x)
        y2 = self.classifier2(x)
        
        return y1, y2
    