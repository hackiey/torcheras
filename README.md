# 安装
$ git clone https://gitlab.com/GammaLab_AI/torcheras.git

$ python setup.py install

# 使用

## 导入模块

```
import torcheras
```

## 定义pyTorch模型和dataloader

```
import torch
from torch.utils.data import DataLoader

model = torch.nn.Module() # 定义pytorch model
trian_dataloader = DataLoader(train_dataset)
val_dataloader = Dataloader(val_dataset, batch_size, shuffle=True, num_workers=4)
```

## 定义torcheras模型
```
model = torcheras.Model(model, logdir) # logdir 为模型存储目录
```
torcheras会在logdir目录下以日期为基础建立当前模型的子文件夹，储存参数设置、模型文件和visualdl变量

## 定义损失函数、优化方法和指标
```
model.compile(loss_function, optimizer, ['categorical_acc'], use_cuda=True)
```

## 训练
```
model.fit(train_data, val_data, epochs)]
```

# 展示
## 训练时展示

训练时会以[epoch, batch] metrics1: metric value, metrics2: metric value的形式进行展示

当一个epoch结束时，会分别显示train和val的结果

```
[1  55] loss: 0.55, acc: 0.88        # train
[1  20] loss: 0.57, acc: 0.82        # val
```

## 图形化展示

每执行一个epoch，会将所有的metrics结果添加至visualdl变量中，进入logdir目录，进入子文件夹，执行
```
visualDL --logdir=./ --host=0.0.0.0 --port=8089
```
访问对应网址，会展示train和val的所有metrics结果

# To do list
1. 更多类型的metrics
2. 更丰富的展示结果
3. 已训练模型的自动化选择