# Introduction
torcheras is a framework with a keras-style API, provides a fast, flexible basis for model training.

# Installation
```
$ git clone https://github.com/hackiey/torcheras.git

$ python setup.py install
```

# Features
- Keras style API, compile, fit. etc. 
- multi task metrics
- Tensorboard support
- Apex fp16 support
- Exponential moving average(EMA)
- Timer
- lr_scheduler support

# basic API
model.compile(self,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    metrics=None,
    multi_tasks=None,
    lr_scheduler=None,
    custom_objects=None):

- optimizer: **pytorch optimizer**
- loss_fn: **loss function**
- metrics: **array of metric names**, now support ['binary_acc', 'categorical_acc', 'topk'], 'topk' should be top1, top2, ...
- multi_tasks: **array of task names**, ['task_a', 'task_b'], the dataset return format should be (input, (label_a, label_b))
- lr_scheduler: **pytorch lr_scheduler**
- custom_objects: **dict{key: function}**, for custom function, the key could be used in metrics(the function's parameters should be (y_true, y_pred))

model.fit(self,
    train_dataloader,
    test_dataloader=None,
    epochs=1,
    logdir='',
    sub_folder='',
    save_checkpoint='epoch',
    save_ema='epoch',
    ema_decay=None,
    fp16=False,
    grad_clip=5.0,
    batch_callback=None,
    epoch_callback=None,
    **kwargs):

- train_dataloader: **dataloader**
- test_dataloader: **dataloader**
- epochs: **int**
- logdir: **str**, the main folder for store models
- sub_folder: **str**, the model will be stored in logdir/sub_folder, if sub_folder is not provied, it will be named by the current time.
- save_checkpoint: **str**, only 'epoch' for now. Model will be not stored for other value. If you want to store the model at specific step, you should use batch_callback for checkpoint save.
- save_ema: **str**, only 'epoch' for now. Ema model will be not stored for other value.
- ema_decay: **float**, 0 for no ema.
- fp16: **bool**, support for apex.
- grad_clip: **float**
- batch_callback: **function(model, epoch, global_step, batch_metrics, tensorboard_writer, \*\*kwargs)**, will be called after each batch.
- epoch_callback: **function(model, epoch, epoch_train_metrics, tensorboard_writer, \*\*kwargs)** will be called after each epoch.

# Get started

```
import torcheras
```

## Define torcheras model
```
# inherit from torcheras.Module
class MyModel(torcheras.Module):
    def __init__(self):
        super(MyModel, self).__init__()
```

## Define loss function, optimizer and metrics
```
multi_tasks = ['output_1', 'output_2']
model.compile(optimizer, loss_function, metrics=['categorical_acc'], multi_tasks=multi_tasks)
```

## Training
```
model.fit(train_data, val_data, epochs, logdir='result', sub_folder='test1')]
```
If you don't provide the sub_folder, torcheras will create a subfolder with time format under the logdir directory to store the model parameters and log data.

**For GPU training**:
```
model.to(device)
```
The variable data will be sent to the device of the first parameter of the model.

```
[1  55, 100, 00:10<00:21, 252.64 it/s] loss: 0.55, acc: 0.88        # train
[test  20] loss: 0.57, acc: 0.82     # val
```

## Graphic scalars
```
$ cd logdir
$ tensorboard --logdir=.
```

# Examples
/tests/test_binary_classification.py

/tests/test_multi_classifications.py
