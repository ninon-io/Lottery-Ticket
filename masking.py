from __future__ import print_function
import torch
import torch.nn.utils
from mnist import NNet
import math
import argparse
import pickle

INITIAL_MODEL_FILENAME = 'initial_model.pt'

ENTIRE_MODEL_FILENAME = "mnist_cnn.pt"
MODEL_WEIGHTS_FILENAME = "mnist_weights_cnn.pt"

# Load the initial model state dict
initial_model = NNet()
initial_model.load_state_dict((torch.load(INITIAL_MODEL_FILENAME)))

# Load the trained model state dict
model_new_weights = NNet()
model_new_weights.load_state_dict(torch.load(MODEL_WEIGHTS_FILENAME))
# model_new = NNet()
# model_new.load_state_dict(torch.load(ENTIRE_MODEL_FILENAME))


class Masking:
    def __init__(self, pruning_percent):
        self.modules = ['conv1', 'conv2', 'fc1', 'fc2']  # TODO: automatic iteration on layer depending the model?
        self.pruning_percent = pruning_percent

    def global_masking(self):
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
        ranked_tensor = torch.sort(weights_array_global)  # Sort the global tensor
        # Get the limit value for masking
        masking_value = ranked_tensor[0][math.floor(ranked_tensor[0].numel() * self.pruning_percent / 100)]
        print('Masking value:', masking_value)

        # Creation of layers' masks and generation of new weights layer by layer
        masked_state_dict = []
        for layer in self.modules:
            initial_weights = getattr(initial_model, layer).weight  # Get the initial weights
            weights_array_local = getattr(model_new_weights, layer).weight  # Get the trained weights
            abs_weights_array_local = torch.Tensor.abs(weights_array_local)  # Abs value of trained weights
            mask_tensor = abs_weights_array_local.ge(masking_value).int()  # Mask the lower weights after training
            masked_weights = initial_weights*mask_tensor  # Mask the future lower weights in the initial tensors
            masked_state_dict.append(masked_weights)  # Create dict with new tensors to inject in state dict

        # Update the initial model state dict
        model_dict = initial_model.state_dict()  # Get the all state dict before training
        cpt = 0
        for (key, tensor) in model_dict.items():
            if 'weight' in key:
                model_dict[key] = masked_state_dict[cpt]  # Replace the initial weights tensor by the masked ones
                cpt += 1
        print('NEW MODEL DICT', model_dict)

    def local_masking(self):
        masked_state_dict = []
        for layer in self.modules:
            initial_weights = getattr(initial_model, layer).weight  # Get the initial weights
            weights_array_local = getattr(model_new_weights, layer).weight  # Get the weight of each layer
            abs_tensor = torch.Tensor.abs(weights_array_local)  # Compute the absolute value
            mask_construction = abs_tensor.view(-1)  # Reshape the tensor
            mask_construction = torch.sort(mask_construction)  # Rank the tensor by value of weights
            # Get the limit value for masking
            masking_value = mask_construction[0][math.floor(mask_construction[0].numel() * self.pruning_percent / 100)]
            print('MASKING VALUE', masking_value)
            mask_tensor = abs_tensor.ge(masking_value).int()  # Get the mask tensor
            masked_weights = initial_weights*mask_tensor  # Compute the new weights tensors
            masked_state_dict.append(masked_weights)

        # Update the model state dict
        model_dict = initial_model.state_dict()
        cpt = 0
        for (key, tensor) in model_dict.items():
            if 'weight' in key:
                model_dict[key] = masked_state_dict[cpt]
                cpt += 1
        print('NEW MODEL DICT', model_dict)


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
