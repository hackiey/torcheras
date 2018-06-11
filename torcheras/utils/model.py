
import torch

def count_parameters(model):
    total_parameters = 0
    trainable_parameters = 0
    
    for name, parameter in model.named_parameters():
        print_str = name
        num_parameter = parameter.numel()

        if parameter.requires_grad:
            name += ' y '
        else:
            name += ' n '
            trainable_parameters += num_parameter
        total_parameters += num_parameter
        
        print_str += str(parameter.shape)
        print_str += ' '
        print_str += str(num_parameter)
            
        print(print_str)
   
    print('total parameters', total_parameters)
    print('trainable parameters', trainable_parameters)
 
    return total_parameters
