import torch
import matplotlib.pyplot as plt
import numpy

from time import time
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim, functional as F

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
num_of_images = 3
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
    plt.show()

# Define the neural network
INPUT_SIZE = 784
HIDDEN_SIZE = [128, 64]
OUTPUT_SIZE = 10

model = nn.Sequential(nn.Linear(INPUT_SIZE, HIDDEN_SIZE[0]),
                      nn.ReLU(),
                      nn.Linear(HIDDEN_SIZE[0], HIDDEN_SIZE[1]),
                      nn.ReLU(),
                      nn.Linear(HIDDEN_SIZE[1], OUTPUT_SIZE),
                      nn.LogSoftmax(dim=1))
print(model)

# Definition of Loss
criterion = nn.NLLLoss()
images, labels = next(iter(train_loader))
images = images.view(images.shape[0], -1)

logps = model(images)  # log probabilities
loss = criterion(logps, labels)  # calculate the NLL loss

# Comment when Lottery Ticket applied
print('before backward pass: \n', model[0].weight.grad)
loss.backward()
print('after backward pass: \n', model[0].weight.grad)

# Core Training Process
optimizer = optim.SGD(model.parameters(), lr = 0.003, momentum = 0.9)
time0 = time()
EPOCHS = 15
for e in range(EPOCHS):
    running_loss = 0
    for images, labels in train_loader:
        # Flatten MNIST images into a long vector of 784
        images = images.view(images.shape[0], -1)

        # Training Pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        # Learning by back-propagation
        loss.backward()

        # Optimization of the weights
        optimizer.step()

        running_loss += loss.item()
    else:
        print('Epoch {} - Training loss: {}'.format(e, running_loss/len(train_loader)))

print('runing time = ', (time() - time0)/60)

images, labels = next(iter(