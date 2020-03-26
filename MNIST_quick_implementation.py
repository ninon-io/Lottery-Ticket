from __future__ import print_function

import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms

# import torch.nn.utils.prune as prune # super cheat

# https://towardsdatascience.com/everything-you-need-to-know-about-saving-weights-in-pytorch-572651f3f8de
# http://ttt.ircam.fr/openvpn.html
# For freezing weights go to the first website

# https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
# Excellent explanation

# https://adventuresinmachinelearning.com/vanishing-gradient-problem-tensorflow/
# The Vanishing Gradient Problem TO READ

# https://www.learnpython.org/en/ String Formatting

parser = argparse.ArgumentParser(description='Pytorch Mnist Wrapped')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='For Saving the current Model')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}  # Don't understand that

# DataLoader prep
transform = transforms.Compose([transforms.ToTensor(),  # converts image into numbers and then into tensor
                                transforms.Normalize((0.1307,),
                                                     (0.3081,))])  # norm tensor w/  mean and standard deviation
train_set = datasets.MNIST(root='./dataset_MNIST', download=True, train=True, transform=transform)
test_set = datasets.MNIST(root='./testset_MNIST', download=True, train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)


# Skeleton of the network
class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

# How the data flow in the network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        out_put = F.log_softmax(x, dim=1)
        return out_put


# Initialization
model = NNet().to(device=device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


# Keeping track of the progress
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(args.epochs + 1)]


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Training pass
        optimizer.zero_grad()  # Initialization of weights TO MODIFY IN ORDER TO RESET WEIGHTS NOT TO ZEROS
        output = model(data)
        loss = F.nll_loss(output, target)  # Equivalent to criterion: loss = criterion(output, target)
        # Learning with back-propagation
        loss.backward()  # Here the matching size is very important
        # Optimizes weights
        optimizer.step()  # Gradient descend step accordingly to the backward propagation
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # Disable autograd functionality (error gradients and backpropagation calculation) not needed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


ENTIRE_MODEL_FILENAME = "mnist_cnn.pt"
MODEL_WEIGHTS = "mnist_weights_cnn.pt"


def main():
    model = NNet().to(device=device)
    time0 = time()
    test()
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()

    # Saving models
    print("\nTraining Time (in minutes) =", (time() - time0) / 60)
    if args.save_model:
        torch.save(model.state_dict(), MODEL_WEIGHTS)  # saves only the weights
        torch.save(model, ENTIRE_MODEL_FILENAME)  # saves all the architecture


if __name__ == '__main__':
    main()

# Data and results visualisation
# TODO: Plotting style using sns
# Plotting Style
sns.set_style('darkgrid')

fig = plt.figure
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig
plt.show()

# reverse operation
# model_new_weights = NNet()
# model_new_weights.load_state_dict(torch.load(MODEL_WEIGHTS))
# model_new = torch.load(ENTIRE_MODEL_FILENAME)
# model.load_state_dict(torch.load(ENTIRE_MODEL_FILENAME))

