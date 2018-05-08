import torch
import torch.nn as nn
import numpy as np

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

def _confusion_matrix(y_true, y_pred, class_num=5):
    y_true = y_true.contiguous()
    
    y_pred = softmax(y_pred)
    y_pred = torch.max(y_pred, 1)[1]
    
    matrix = np.zeros((class_num,class_num))
    batch_size = y_true.size(0)
    
    for i in range(batch_size):
        row = int(y_true[i])
        column = int(y_pred[i])
        matrix[row,column] += 1
    return matrix

class MyException(Exception):
    def __init__(self,message):
        Exception.__init__(self)
        self.message=message
        
def _f1_score(matrix, class_num = "binary"):
    """
    class_num = "binary" or "multiclass"
    """
    f1_array = []
    
    row = matrix.shape[0]
    column = matrix.shape[1]
    if row != column:
        try:
            raise MyException("The matrix must be a square matrix")
        except MyException as e:
            print(e.message)
            
    correct_num = 0    
    for i in range(row):
        correct_num += matrix[i, i]  
        p_i = matrix[i, i] / np.sum(matrix[:, i])
        r_i = matrix[i, i] / np.sum(matrix[i, :])
        f1_i = 2 * p_i * r_i / (p_i + r_i)
        f1_array.append(f1_i)
    
    micro_f1 = correct_num / np.sum(matrix)  
    macro_f1 = sum(f1_array) / row
    
    if class_num == "binary":
        res = f1_array
    if class_num == "multiclass":
        res = {'micro': micro_f1, 'macro_f1': macro_f1, 'f1_pos_none': f1_array} 
    return res
