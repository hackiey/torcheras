from collections import OrderedDict

class EMA():
    def __init__(self, decay, parameters, device_store=None):
        self.decay = decay
        self.steps = 0
        
        self.shadow = OrderedDict()
        self.original = OrderedDict()
        
        for _name, _parameter in parameters:
            if _parameter.requires_grad:
                if device_store is not None:
                    self.shadow[_name] = _parameter.data.clone().to(device_store)
                else:
                    self.shadow[_name] = _parameter.data.clone()

    def __call__(self, parameters):
        self.steps += 1
        # decay = min((self.steps + 1) / (10 + self.steps), self.decay)
        decay = self.decay
        
        for _name, _parameter in parameters:
            if _parameter.requires_grad:
                new_average = (1.0 - decay) * _parameter.data + decay * self.shadow[_name]
                self.shadow[_name] = new_average.clone()
        return self.shadow
    
    def assign(self, model, device_store=None, device_execute=None):
        for _name, _parameter in model.named_parameters():
            if _parameter.requires_grad:
                if device_store is not None:
                    self.original[_name] = _parameter.data.clone().to(device_store)
                else:
                    self.original[_name] = _parameter.data.clone()
                if device_execute is not None:
                    _parameter.data = self.shadow[_name].to(device_execute)
                else:
                    _parameter.data = self.shadow[_name]
                    
    def resume(self, model, device_execute=None):
        for _name, _parameter in model.named_parameters():
            if _parameter.requires_grad:
                if device_execute is not None:
                    _parameter.data = self.original[_name].to(device_execute)
                else:
                    _parameter.data = self.original[_name]
                