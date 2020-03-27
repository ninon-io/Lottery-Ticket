from time import time
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
import torch.nn.utils
import mnist
from mnist import NNet
from masking import *

# get the arguments, if not on command line, the arguments are the default
parser = argparse.ArgumentParser(description='Pytorch Mnist Wrapped')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
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


def main():
    model = NNet().to(device=device)
    time0 = time()
    mnist.test()
    for epoch in range(1, args.epochs + 1):
        mnist.train(epoch)
        mnist.test()

    # Saving models
    print("\nTraining Time (in minutes) =", (time() - time0) / 60)
    if args.save_model:
        torch.save(model.state_dict(), MODEL_WEIGHTS)  # saves only the weights
        torch.save(model, ENTIRE_MODEL_FILENAME)  # saves all the architecture

    # Data and results visualisation
    # TODO: Plotting style using sns
    # Plotting Style
    sns.set_style('darkgrid')
    plt.figure()
    plt.plot(mnist.train_counter, mnist.train_losses, color='blue')
    plt.scatter(mnist.test_counter, mnist.test_losses, color='red')
    plt.legend(['Train loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    sns.set_context('paper')  # for saving the figure
    sns.set()
    plt.show()

    # Generate new state dict with masked weights
    Masking.masking()


if __name__ == '__main__':
    main()
