import torch
import torch.nn as nn
sigmoid = nn.Sigmoid()
softmax = nn.Softmax()

def binary_accuracy(y_true, y_pred):
    y_pred = sigmoid(y_pred)
    y_pred = torch.round(y_pred)
    return torch.mean(y_true.eq(y_pred).type(torch.FloatTensor), 0)

def categorical_accuracy(y_true, y_pred):
    y_pred = softmax(y_pred)
    y_pred = torch.max(y_pred, 1)[1]
    return torch.mean(y_true.eq(y_pred.long()).float(), 0)

def top2(y_true, y_pred):
    y_pred = softmax(y_pred)
    