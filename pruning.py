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
parser.add_argument('--pruning_percent--', type=int, default=10, metavar='P',
                    help='percentage of pruning for each cycle (default: 10)')


# Architecture problem as I want the call of method being an argument (cf up) associate with the class ?
class Pruning:
    def __init__(self):
        self.modules = ['conv1', 'conv2', 'fc1', 'fc2']
        # self.pruning_percent = args.pruning_percent  # How to call here the argparse ????

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
        # Work in progress on weights from different layers
        print('========================= LOOP ON LAYERS ================================')
        for layer in self.modules:
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
            abs_weights = torch.Tensor.abs(weights_array_local_ranked[0])
            print('ABS', abs_weights, abs_weights.numel())
            print('================END LOCAL================')

        print('genius')


test_1 = Pruning()
# test_1.__global_pruning__()
test_1.__local_pruning__()

print("========== TEST PLAYGROUND ==========")
pruning_percentage = 50  # To put as an argument for the user
testing_tensor = torch.randn(3, 4)
print(testing_tensor)
abs_test = torch.Tensor.abs(testing_tensor)
print(abs_test)
shaped_tensor = abs_test.view(-1)
print('shaped', shaped_tensor)
rank_tensor = torch.sort(shaped_tensor)
print(rank_tensor)
masking_value = rank_tensor[0][math.floor(rank_tensor[0].numel() * pruning_percentage / 100)]
print('mask:', masking_value)
mask_bool = abs_test.ge(masking_value)
print('mask bool', mask_bool[0][0])
# Somehow better method to change boolean tensor into int tensor
for i, j in mask_bool[:][:]:
    if mask_bool[i][j]:
        mask_bool[i][j] = 1
    if not mask_bool[i][j]:
        mask_bool[i][j] = 0
mask = mask_bool
print('mask:', mask)

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
