from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
import torch.optim as optim
import main as main


from masking import Masking

# import torch.nn.utils.prune as prune # super cheat


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
model = NNet().to(device=main.device)
optimizer = optim.SGD(model.parameters(), lr=main.args.lr, momentum=main.args.momentum)


# Keeping track of the progress
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(main.train_loader.dataset) for i in range(main.args.epochs + 1)]


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(main.train_loader):
        data, target = data.to(main.device), target.to(main.device)
        # Training pass
        optimizer.zero_grad()  # Initialization of weights TO MODIFY IN ORDER TO RESET WEIGHTS NOT TO ZEROS
        output = model(data)
        loss = F.nll_loss(output, target)  # Equivalent to criterion: loss = criterion(output, target)
        # Learning with back-propagation
        loss.backward()  # Here the matching size is very important
        # Optimizes weights
        optimizer.step()  # Gradient descend step accordingly to the backward propagation
        if batch_idx % main.args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(main.train_loader.dataset),
                100. * batch_idx / len(main.train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(main.train_loader.dataset)))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # Disable autograd functionality (error gradients and backpropagation calculation) not needed
        for data, target in main.test_loader:
            data, target = data.to(main.device), target.to(main.device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(main.test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(main.test_loader.dataset),
        100. * correct / len(main.test_loader.dataset)))


ENTIRE_MODEL_FILENAME = "mnist_cnn.pt"
MODEL_WEIGHTS = "mnist_weights_cnn.pt"


# reverse operation
# model_new_weights = NNet()
# model_new_weights.load_state_dict(torch.load(MODEL_WEIGHTS))
# model_new = torch.load(ENTIRE_MODEL_FILENAME)
# model.load_state_dict(torch.load(ENTIRE_MODEL_FILENAME))

