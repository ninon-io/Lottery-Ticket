from time import time
import argparse
from torchvision import datasets
from torchvision import transforms
import torch.nn.utils

from learn import Learn
# from masking import Masking

ENTIRE_MODEL_FILENAME = "mnist_cnn.pt"
MODEL_WEIGHTS = "mnist_weights_cnn.pt"

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
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}  # Don't understand that


if __name__ == "__main__":
    # DataLoader prep
    transform = transforms.Compose([transforms.ToTensor(),  # converts image into numbers and then into tensor
                                    transforms.Normalize((0.1307,),
                                                         (0.3081,))])  # norm tensor w/  mean and standard deviation
    train_set = datasets.MNIST(root='./dataset_MNIST', download=True, train=True, transform=transform)
    test_set = datasets.MNIST(root='./testset_MNIST', download=True, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    learn = Learn(train_loader, test_loader, batch_size=args.batch_size, seed=args.seed, cuda=use_cuda)
    time0 = time()
    learn.test()
    for epoch in range(0, args.epochs):
        learn.train()
        learn.test()
    learn.plot()

    # Saving models
    print("\nTraining Time (in minutes) =", (time() - time0) / 60)
    if args.save_model:
        torch.save(learn.model.state_dict(), MODEL_WEIGHTS)  # saves only the weights
        torch.save(learn.model, ENTIRE_MODEL_FILENAME)  # saves all the architecture


