import torch
from torch import optim
from torch.nn import functional as F

from mnist import NNet

LEARNING_RATE = 0.01
MOMENTUM = 0.5


class Learn:
    def __init__(self, train_loader, test_loader, batch_size=64, seed=1, cuda=False):
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if cuda else "cpu")
        # Initialization
        self.model = NNet().to(device=self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        self.batch_size = batch_size
        self.epoch = 0
        self.train_loader = train_loader
        self.test_loader = test_loader

    def run(self, data):
        data.to(self.device)
        self.optimizer.zero_grad()
        return self.model(data)

    def train(self, log_interval=10):
        train_losses = []
        train_counter = []
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            # Training pass
            self.optimizer.zero_grad()  # Initialization of weights TO MODIFY IN ORDER TO RESET WEIGHTS NOT TO ZEROS
            output = self.model(data)
            loss = F.nll_loss(output, target)  # Equivalent to criterion: loss = criterion(output, target)
            # Learning with back-propagation
            loss.backward()  # Here the matching size is very important
            # Optimizes weights
            self.optimizer.step()  # Gradient descend step accordingly to the backward propagation
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, batch_idx * len(data), len(self.train_loader.dataset),
                                100. * batch_idx / len(self.train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((self.epoch - 1) * len(self.train_loader.dataset)))

    def test(self):
        test_losses = []
        test_counter = [i * len(self.train_loader.dataset) for i in range(self.epoch + 1)]
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():  # Disable autograd functionality (error gradients and back-propagation calculation)
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

