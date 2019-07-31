from collections import OrderedDict
from datetime import datetime

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
                device_model = _parameter.device
                device_shadow = self.shadow[_name].device
                
                if device_model != device_shadow:
                    new_average = (1.0 - decay) * _parameter.to(device_shadow).data + decay * self.shadow[_name]
                else:    
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
                
                
class Timer():
    def __init__(self, length):
        
        self.length = length
        self.step = 0
        self.start_time = datetime.now()
        self.last_time = self.start_time
        
    def strf_deltatime(self, deltatime):
        m, s = divmod(deltatime, 60)
        h, m = divmod(m, 60)
        print_str = '%.2d:%.2d' % (int(m), int(s))
        if h != 0:
            print_str = str(h) + ':' + print_str
        
        return print_str
        
    def __call__(self):
        
        self.step += 1
        now = datetime.now()
        
        spent_time = (now - self.start_time).total_seconds()
        average_time = spent_time / self.step
        left_time = average_time * (self.length - self.step)
        self.last_time = now
        
        it_per_second = '%.2f it/s' % (1/average_time)
        spent_time = self.strf_deltatime(spent_time)
        left_time = self.strf_deltatime(left_time)
        
        return it_per_second, spent_time, left_time
        