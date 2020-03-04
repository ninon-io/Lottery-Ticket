import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
import urllib3
from urllib.request import urlopen, Request

header = {'Mozilla/73.0.1'}
reg_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
req = Request(url=reg_url, headers=header)
html = urlopen(req).read()

transform = transforms.Compose([transforms.ToTensor(),  # converts image into numbers and then into tensor
                                transforms.Normalize((0.5,), (0.5,))])  # norm tensor w/  mean and standard deviation

train_set = datasets.MNIST(root='./dataset_MNIST', download=True, train=True, transform=transform)

test_set = datasets.MNIST(root='./testset_MNIST', download=True, train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

data_iter = iter(train_loader)
images, labels = data_iter.next()

# Display the size of tensors
print(images.shape)
print(labels.shape)

# Plot some images from the database
figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images+1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[0].np().squeeze(), cmap='gray_r')

# Define the neural network
INPUT_SIZE = 784
HIDDEN_SIZE = [128, 64]
OUTPUT_SIZE = 10

model = nn.Sequential(nn.Linear(INPUT_SIZE, HIDDEN_SIZE[0]),
                      nn.Relu(),
                      nn.Linear(HIDDEN_SIZE[0], HIDDEN_SIZE[1]),
                      nn.Relu(),
                      nn.Linear(HIDDEN_SIZE[1], OUTPUT_SIZE),
                      nn.LogSoftmax(dim=1))
print(model)

# Definition of Loss
criterion = nn.NLLLoss()
images, labels = next(iter(train_loader))
