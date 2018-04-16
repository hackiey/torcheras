import torch
import numpy as np

def proportion(dataloader, label_index, label_nums, label_names = []):
    '''
    parameters:
        dataloader: dataloader of dataset
        label_index: ith returned value from dataset.__getitem(index), 
                        if returned value is (x, y), the label_index is 1
                        or (x1, x2, y)'s label_index is 2
        label_nums: number of label categories
        label_names: label name
        
    example:
        torcheras.utils.proportion(dataloader, 1, 2, label_names = ['col1', 'col2']
    '''
    
    labels_count = torch.zeros(label_nums)
    num = 0
    for data in dataloader:
        for v in data[label_index]:
            labels_count[v] += 1
        num += data[label_index].shape[0]
    
    if len(label_names) == 0:
        return (labels_count / num).numpy()
    else:
        proportion_result = (labels_count / num).numpy()
        result = []
        for i, label_name in enumerate(label_names):
            result.append((label_name, proportion_result[i]))
        return result
        