from __future__ import print_function
import torch
import torch.nn.utils
from mnist import NNet
import math
import argparse
import pickle

ENTIRE_MODEL_FILENAME = "mnist_cnn.pt"
MODEL_WEIGHTS = "mnist_weights_cnn.pt"

# Load the trained model state dict
model_new_weights = NNet()
model_new_weights.load_state_dict(torch.load(MODEL_WEIGHTS))
# model_new = NNet()
# model_new.load_state_dict(torch.load(ENTIRE_MODEL_FILENAME))

parser = argparse.ArgumentParser(description='Pruning Algorithm for MNIST')
parser.add_argument('--global_pruning', type=str, default='global_pruning', metavar='G',
                    help='A masking on all layers will be apply')
parser.add_argument('--local_pruning', type=str, default='global_pruning', metavar='L',
                    help='A masking layer by layer will be apply')
parser.add_argument('--pruning_percent', type=int, default=75, metavar='P',
                    help='percentage of pruning for each cycle (default: 10)')

args = parser.parse_args()


class Masking:
    def __init__(self):
        self.modules = ['conv1', 'conv2', 'fc1', 'fc2']  # TODO: automatic iteration on layer depending the model?
        self.pruning_percent = args.pruning_percent

    def __global_masking__(self):
        # Access of the masking value and construction of masks
        weights_array_global = getattr(model_new_weights, self.modules[0]).weight  # Get the first layer tensor
        weights_array_global = weights_array_global.view(-1)  # Reshape it
        print('Tensors local number of elements: ', weights_array_global.numel())
        # Construction of the global tensor
        for layer in self.modules[1:]:
            weights_array_local = getattr(model_new_weights, layer).weight  # Get the weight of each layer
            abs_weights_array_local = torch.Tensor.abs(weights_array_local)  # Compute the abs values
            global_array_construction = (abs_weights_array_local.view(-1))  # Reshape the tensor
            weights_array_global = torch.cat([weights_array_global, global_array_construction])  # Concatenate tensors
            print('Tensors local number of elements: ', weights_array_local.numel())

        # Computation of the masking limit
        print(torch.sort(weights_array_global))
        print('Tensors global number of elements: ', weights_array_global.numel())
        ranked_tensor = torch.sort(weights_array_global)
        # Get the limit value for masking
        masking_value = ranked_tensor[0][math.floor(ranked_tensor[0].numel() * self.pruning_percent / 100)]
        print('Masking value:', masking_value)

        # Creation of layers' masks and generation of new weights layer by layer
        masked_state_dict = []
        for layer in self.modules:  # TODO: Redundancy of variable?
            weights_array_local = getattr(model_new_weights, layer).weight
            abs_weights_array_local = torch.Tensor.abs(weights_array_local)
            mask_tensor = abs_weights_array_local.ge(masking_value).int()
            masked_weights = weights_array_local*mask_tensor
            masked_state_dict.append(masked_weights)
        print('genius global')

        # Update the model state dict
        model_dict = model_new_weights.state_dict()
        cpt = 0
        for (key, tensor) in model_dict.items():
            if 'weight' in key:
                model_dict[key] = masked_state_dict[cpt]
                cpt += 1
        print('NEW MODEL DICT', model_dict)
        print('genius this is the end')

    def __local_masking__(self):
        masked_state_dict = []
        for layer in self.modules:
            weights_array_local = getattr(model_new_weights, layer).weight  # Get the weight of each layer
            abs_tensor = torch.Tensor.abs(weights_array_local)  # Compute the absolute value
            mask_construction = abs_tensor.view(-1)  # Reshape the tensor
            mask_construction = torch.sort(mask_construction)  # Rank the tensor by value of weights
            # Get the limit of masking value
            masking_value = mask_construction[0][math.floor(mask_construction[0].numel() * self.pruning_percent / 100)]
            print('MASKING VALUE', masking_value)
            mask_tensor = abs_tensor.ge(masking_value).int()  # Get the mask tensor
            masked_weights = weights_array_local*mask_tensor  # Compute the new weights tensors
            masked_state_dict.append(masked_weights)
        pickle.dump(masked_state_dict, open("masked_state_dict_local.pt", "wb"))
        print('genius local')

        # Update the model state dict
        model_dict = model_new_weights.state_dict()
        cpt = 0
        for (key, tensor) in model_dict.items():
            if 'weight' in key:
                model_dict[key] = masked_state_dict[cpt]
                cpt += 1
        print('NEW MODEL DICT', model_dict)
        print('genius this is the end')


def masking():
    if args.global_pruning:
        test = Masking()
        test.__global_masking__()
    else:
        test = Masking()
        test.__local_masking__()


# TODO: Link the new_state_dict to MNIST model and train it again!
# TODO: Print and plot results to compare!

# Somehow better method to change boolean tensor into int tensor => Indeed .int()
# for i in range(len(mask_bool)):
#     for j in range(len(mask_bool[i])):
#         mask_bool[i][j] = not mask_bool[i][j]
# print('mask bool', mask_bool)

# Print model's state dict
# print("-----Model's State dict after training-----")
# for name, param in model_new_weights.named_parameters():
#     print('==========')
#     print(name, ':', param.requires_grad)
#     print('==========')
# for layer in model_new_weights.named_parameters():
#     print('=========', layer, '==========')
#     print('=====', model_new_weights.named_parameters(), '====')


# Uncomment if experiments on optimizers
# Print optimizer's state dict
# print("Optimizer State dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])
