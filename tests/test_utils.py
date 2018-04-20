import torcheras
from torch.utils.data import DataLoader
from datasets import CategoricalDataset

print('Start test utils.proportion...')
N = 100
D_in = 10
D_out = 10
categoricalDataset = CategoricalDataset(N, D_in, D_out)
categoricalDataloader = DataLoader(categoricalDataset, batch_size = 16, shuffle = True, num_workers = 1)
label_names = []
for i in range(D_out):
    label_names.append('col' + str(i))
proportion = torcheras.utils.proportion(categoricalDataloader, 1, D_out, label_names)
print(proportion)
print('utils.proportion test done.')