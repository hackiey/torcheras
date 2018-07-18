# Installation
```
$ git clone https://gitlab.com/GammaLab_AI/torcheras.git

$ python setup.py install
```

# Get started

## Import

```
import torcheras
```

## Define pyTorch model and dataloader

```
import torch
from torch.utils.data import DataLoader

model = torch.nn.Module() # define pytorch model
trian_dataloader = DataLoader(train_dataset)
val_dataloader = Dataloader(val_dataset)
```

## Define torcheras model
```
model = torcheras.Model(model, logdir) # logdir is the path to store model parameters and log data
```
torcheras will create a subfolder of the current model with a date-based name under the logdir directory, to store the model parameters and log data

## Define loss function, optimizer and metrics
```
multi_tasks = ['output_1', 'output_2']
model.compile(loss_function, optimizer, ['categorical_acc'], multi_tasks = multi_tasks, device = device)
```

## Training
```
model.fit(train_data, val_data, epochs)]
```

```
[1  55] loss: 0.55, acc: 0.88        # train
[test  20] loss: 0.57, acc: 0.82     # val
```

## Graphic scalars
```
$ cd logdir
$ torcheras --host=0.0.0.0 --port=8089
```

# More examples
/tests/test_binary_classification.py

/tests/test_multi_classifications.py

# To do
- [ ] time display
- [ ] profiler