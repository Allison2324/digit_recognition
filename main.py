import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")
print(train)
print(train["label"])
print(len(train.index))
x_train_set = np.array(train.loc[:, train.columns != "label"])
print(x_train_set)
y_train_set = np.array(train["label"])
print(y_train_set)


def plot_digit(x_set, y_set, idx):
    img = x_set[idx].T.reshape(28, 28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.title('true label: %d' % y_set.T[idx])
    plt.show()


print(x_train_set.shape)
plot_digit(x_train_set, y_train_set, 345)
