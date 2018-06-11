from collections import OrderedDict

class EMA():
    def __init__(self, decay, parameters):
        self.decay = decay
        self.steps = 0
        self.shadow = OrderedDict()
        for _name, _parameter in parameters:
            if _parameter.requires_grad:
                self.shadow[_name] = _parameter.clone()

    def __call__(self, parameters):
        self.steps += 1
        decay = min((self.steps + 1) / (10 + self.steps), self.decay)

        for _name, _parameter in parameters:
            if _parameter.requires_grad:
                new_average = (1.0 - decay) * _parameter.data + decay * self.shadow[_name]
                self.shadow[_name] = new_average.clone()
        return self.shadow


# model = Model()
# ema = EMA(0.9999, model.named_parameters())
# ema(model.named_parameters())

# torch.save(ema.shadow, logdir)
