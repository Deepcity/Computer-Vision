#Import the packages needed.
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from torchvision.transforms import ToTensor

data_train = torchvision.datasets.MNIST('data',
        download=True,train=True,transform=ToTensor())
data_test = torchvision.datasets.MNIST('data',
        download=True,train=False,transform=ToTensor())

# 查看对应数据
fig,ax = plt.subplots(1,7)
for i in range(7):
    ax[i].imshow(data_train[i][0].view(28,28))
    ax[i].set_title(data_train[i][1])
    ax[i].axis('off')

#
print('Min intensity value: ',data_train[0][0].min().item())
print('Max intensity value: ',data_train[0][0].max().item())

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchinfo import summary

from pytorchcv import load_mnist, plot_results
load_mnist()
