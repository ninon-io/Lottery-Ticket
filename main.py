from time import time
import argparse
from torchvision import datasets
from torchvision import transforms
import torch.nn.utils

from learn import Learn
from learn import weights_init

INITIAL_MODEL = 'initial_model.pt'

ENTIRE_MODEL_FILENAME = "mnist_cnn.pt"
MODEL_WEIGHTS = "mnist_weights_cnn.pt"

ENTIRE_MODEL_MASKED = "mnist_masked.pt"
MODEL_WEIGHTS_MASKED = "mnist_weights_masked.pt"

# get the arguments, if not on command line, the arguments are the default
parser = argparse.ArgumentParser(description='Pytorch Mnist Wrapped')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='Momentum (defaults: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
parser.add_argument('--local', dest='local', default=False, action='store_true')
parser.add_argument('--pruning_percent', type=int, default=99.7, metavar='P',
                    help='percentage of pruning for each cycle (default: 10)')

args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

if __name__ == "__main__":
    # DataLoader prep
    transform = transforms.Compose([transforms.ToTensor(),  # converts image into numbers and then into tensor
                                    transforms.Normalize((0.1307,),
                                                         (0.3081,))])  # norm tensor w/  mean and standard deviation
    train_set = datasets.MNIST(root='./dataset_MNIST', download=True, train=True, transform=transform)
    test_set = datasets.MNIST(root='./testset_MNIST', download=True, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    learn = Learn(train_loader, test_loader, batch_size=args.batch_size,
                  seed=args.seed, cuda=use_cuda, lr=args.lr, momentum=args.momentum)
    # Set the time
    time0 = time()
    # Initial training of the model
    learn.model.apply(weights_init)  # Random Initialization of weights
    torch.save(learn.model.state_dict(), INITIAL_MODEL)  # Saves this random init
    print(' ==================')
    print('| INITIAL TRAINING |')
    print(' ==================')
    learn.test()  # First test on randomly initialized data
    for epoch in range(0, args.epochs):
        learn.train()
        learn.test()
    learn.plot()

    # Saving models
    print("\nTraining Time (in minutes) =", (time() - time0) / 60)
    if args.save_model:
        torch.save(learn.model.state_dict(), MODEL_WEIGHTS)  # saves only the weights
        torch.save(learn.model, ENTIRE_MODEL_FILENAME)  # saves all the architecture

    from masking import Masking

    # Mask the weights
    print(' ===================')
    print('| MASKING PROCEDURE |')
    print(' ===================')
    mask = Masking(args.pruning_percent)
    if args.local:
        mask.global_masking()
    else:
        mask.local_masking()

    # Retrain the model with masked weight in state dict
    print(' =================')
    print('| MASKED TRAINING |')
    print(' =================')
    model_state_dict = learn.model.load_state_dict(torch.load(INITIAL_MODEL))
    learn.test()
    for epoch in range(0, args.epochs):
        learn.train()
        learn.test()
    learn.plot()

    # Set the time again
    time0 = time()
    # Saving sparser models
    print("\nTraining Time (in minutes) =", (time() - time0) / 60)
    if args.save_model:
        torch.save(learn.model.state_dict(), MODEL_WEIGHTS_MASKED)  # saves only the weights
        torch.save(learn.model, ENTIRE_MODEL_MASKED)  # saves all the architecture
