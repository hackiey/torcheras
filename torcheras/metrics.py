import torch
import torch.nn as nn
sigmoid = nn.Sigmoid()
softmax = nn.Softmax()

def binary_accuracy(y_true, y_pred):
    y_true = y_true.contiguous()
    
    y_pred = sigmoid(y_pred)
    y_pred = torch.round(y_pred)
    return torch.mean(y_true.eq(y_pred).type(torch.FloatTensor), 0)

def categorical_accuracy(y_true, y_pred):
    y_true = y_true.contiguous()
    
    y_pred = softmax(y_pred)
    y_pred = torch.max(y_pred, 1)[1]
    return torch.mean(y_true.eq(y_pred.long()).float(), 0)

def topk(y_true, y_pred, topk):
    y_true = y_true.contiguous()
    
    y_pred = softmax(y_pred)
    maxk = max(topk)
    batch_size = y_true.size(0)

    _, pred = y_pred.topk(maxk, 1)
    pred = pred.t()
    correct = pred.eq(y_true.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
        
    return res