import torch

def count_parameters(model):
    for name, parameter in model.named_parameters():
        num_parameter = parameter.numel()
        print_str = name
        if parameter.requires_grad:
            print_str += ' y '      
        else:
            print_str += ' n '       
        print_str += str(parameter.shape)
        print_str += ' '
        print_str += str(num_parameter)
            
        print(print_str)
    
    total_parameters = 0
    trainable_parameters = 0
    for parameter in model.parameters():
        num_parameter = parameter.numel()
        if parameter.requires_grad:
            trainable_parameters += num_parameter
        total_parameters += num_parameter
        
    print('total parameters', total_parameters)
    print('trainable parameters', trainable_parameters)
 
    return total_parameters
