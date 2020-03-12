from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.utils
# import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.optim as optim
import argparse
import collections
from collections import OrderedDict
import seaborn as sns
import numpy
import matplotlib.pyplot as plt


from time import time
from torchvision import datasets
from torchvision import transforms


# Plotting Style
# sns.set_style('darkgrid')


parser = argparse.ArgumentParser(description='Pytorch Mnist Wrapped')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--pruning--', type=int, default=10, metavar='P',
                    help='percentage of pruning for each cycle (default: 10)')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")


class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.sequential = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 32, 3, 1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(32, 64, 3, 1)),
            ('relu2', nn.ReLU()),
            ('max_pool', nn.MaxPool2d(2)),
            ('drop1', nn.Dropout(0.25)),
            ('drop2', nn.Dropout(0.5)),
            ('linear1', nn.Linear(9216, 128)),
            ('linear2', nn.Linear(128, 10))
        ]))
        # if does not work with nn.Sequential
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # find how to sequentially flatt
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        out_put = F.log_softmax(x, dim=1)
        return out_put


# model = NNet().to(device=device)
# module = model.conv1
# print(list(module.named_parameters()))
# print(list(module.named_buffers()))


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Training pass
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        # Learning with back-propagation
        loss.backward()
        # Optimizes weights
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


MODEL_FILENAME = "mnist_cnn.pt"


def main():
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    transform = transforms.Compose([transforms.ToTensor(),  # converts image into numbers and then into tensor
                                    transforms.Normalize((0.5,),
                                                         (0.5,))])  # norm tensor w/  mean and standard deviation
    train_set = datasets.MNIST(root='./dataset_MNIST', download=True, train=True, transform=transform)
    test_set = datasets.MNIST(root='./testset_MNIST', download=True, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    model = NNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    time0 = time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader,  optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    # Experiments
    print("\nTraining Time (in minutes) =", (time() - time0) / 60)
    if args.save_model:
        torch.save(model.state_dict(), MODEL_FILENAME)


# reverse operation
model = torch.load(MODEL_FILENAME)
model.load_state_dict(torch.load(MODEL_FILENAME))
# Print model's state dict
print("Model's State dict:")
for param_tensor in model.state_dict():
    print('--Sizes of tensor--', param_tensor, "\t", model.state_dict()[param_tensor].size())  # Access to sizes
    print('--Weights--', param_tensor, "\t", list(model.state_dict()[param_tensor]))  # Access to weights
    # Normalization of weights
    raw_weights = model.load.state_dict()[param_tensor]
    print('raw weights', raw_weights)

    # Uncomment if experiments on optimizers
    # Print optimizer's state dict
    # print("Optimizer State dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])


if __name__ == '__main__':
    main()
