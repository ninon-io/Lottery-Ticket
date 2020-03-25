from __future__ import print_function
import torch
import torch.nn.utils
from MNIST_quick_implementation import NNet
import math
import argparse

ENTIRE_MODEL_FILENAME = "mnist_cnn.pt"
MODEL_WEIGHTS = "mnist_weights_cnn.pt"

# Load the trained model
model_new_weights = NNet()
model_new_weights.load_state_dict(torch.load(MODEL_WEIGHTS))
# model_new = torch.load(ENTIRE_MODEL_FILENAME)
# model.load_state_dict(torch.load(ENTIRE_MODEL_FILENAME))

parser = argparse.ArgumentParser(description='Pruning Algorithm for MNIST')
parser.add_argument('--pruning_percent', type=int, default=10, metavar='P',
                    help='percentage of pruning for each cycle (default: 10)')
parser.add_argument('--global_pruning', type=str, default=10, metavar='G',
                    help='percentage of pruning for each cycle (default: 10)')
parser.add_argument('--local_pruning', type=str, default=10, metavar='L',
                    help='percentage of pruning for each cycle (default: 10)')

args = parser.parse_args()


# Architecture problem as I want the call of method being an argument (cf up) associate with the class ?
class Pruning:
    def __init__(self):
        self.modules = ['conv1', 'conv2', 'fc1', 'fc2']
        self.pruning_percent = args.pruning_percent  # How to call here the argparse ????

    def __global_pruning__(self):
        # Work in progress on weights from different layers
        print('========================= LOOP ON LAYERS ================================')
        weights_array_global = getattr(model_new_weights, self.modules[0]).weight
        for layer in self.modules[1:]:
            print('=================LOCAL=================')
            weights_array_local = getattr(model_new_weights, layer).weight
            print('WEIGHT LAYER', weights_array_local)
            print('Tensors shape: ', weights_array_local.shape)
            weights_array_local = (weights_array_local.view(-1))
            print('WEIGHTS ARRAY RESHAPED:', weights_array_local)
            print('Tensors RESHAPE: ', weights_array_local.shape)
            weights_array_local_ranked = torch.sort(weights_array_local)
            print('SORTED LOCAL WEIGHTS', weights_array_local_ranked)
            print('Tensors local number of elements: ', weights_array_local.numel())
            norm_weights = torch.norm(weights_array_local)  # Gives the norm of tensors, useful ? Not sure
            print('NORM TENSORS: ', norm_weights)
            # weights_array_global += torch.cat([weights_array_global, weights_array_local])  # PROBLEM OF SIZES TENSORS
            # masked_weight = torch.scatter() TO CONTINUE HERE
            print('================END LOCAL================')

        print('================GLOBAL================')
        print(torch.sort(weights_array_global))
        print('Tensors global shape', weights_array_global.shape)
        print('Tensors global number of elements: ', weights_array_global.numel())
        print('======================================')

        # print(list(module.named_parameters(layer)))
        # norm_weights = torch.norm(model_new_weights.named_parameters('weight'))
        # print(norm_weights)

        print('genius')

    def __local_pruning__(self):
        print('========================= LOOP ON LAYERS ================================')
        for layer in self.modules:
            print('=================LOCAL=================')
            weights_array_local = getattr(model_new_weights, layer).weight  # Get the weight of each layer
            print('WEIGHT LAYER', weights_array_local)
            abs_tensor = torch.Tensor.abs(weights_array_local)  # Compute the absolute value
            mask_construction = abs_tensor.view(-1)  # Reshape the tensor
            mask_construction = torch.sort(mask_construction)  # Rank the tensor by value of weights
            # Get the limit of masking value
            masking_value = mask_construction[0][math.floor(mask_construction[0].numel() * self.pruning_percent / 100)]
            print('MASKING VALUE', masking_value)
            mask_tensor = abs_tensor.ge(masking_value).int()  # Get the mask tensor
            masked_weights = weights_array_local*mask_tensor  # Compute the new weights tensors
            print('MASKED WEIGHT', masked_weights)
            print('================END LOCAL================')

        print('genius')


test_1 = Pruning()
# test_1.__global_pruning__()
test_1.__local_pruning__()

print("========== TEST PLAYGROUND ==========")
pruning_percentage_test = 50  # To put as an argument for the user
testing_tensor = torch.randn(3, 4)  # Initialization
print(testing_tensor)
abs_test = torch.Tensor.abs(testing_tensor)  # Takes absolute value
print(abs_test)
shaped_tensor = abs_test.view(-1)  # Reshape on one column tensor
print('shaped', shaped_tensor)
rank_tensor = torch.sort(shaped_tensor)  # Sort the weights
print(rank_tensor)
# Takes the pruning value
masking_value_test = rank_tensor[0][math.floor(rank_tensor[0].numel() * pruning_percentage_test / 100)]
print('mask:', masking_value_test)
print("----------")
mask_bool = abs_test.ge(masking_value_test).int()  # Generates a mask from the masking value calculated
print('mask bool', mask_bool)
masked_tensor = testing_tensor*mask_bool  # Gives the new appropriate tensor of weights to train
print('masked tensor', masked_tensor)


# Somehow better method to change boolean tensor into int tensor
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
