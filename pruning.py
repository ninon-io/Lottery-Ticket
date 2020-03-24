from __future__ import print_function
import torch.nn.utils
from MNIST_quick_implementation import NNet

ENTIRE_MODEL_FILENAME = "mnist_cnn.pt"
MODEL_WEIGHTS = "mnist_weights_cnn.pt"

# Load the trained model
model_new_weights = NNet()
model_new_weights.load_state_dict(torch.load(MODEL_WEIGHTS))
# model_new = torch.load(ENTIRE_MODEL_FILENAME)
# model.load_state_dict(torch.load(ENTIRE_MODEL_FILENAME))


# Architecture problem as I want the call of method being an argument associate with the class ?
class Pruning:
    def __init__(self):
        self.modules = ['conv1', 'conv2', 'fc1', 'fc2']

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
            # masked_weight = torch.scatter() TO CONTINUE HERE
            print('================END LOCAL================')

        # print(list(module.named_parameters(layer)))
        # norm_weights = torch.norm(model_new_weights.named_parameters('weight'))
        # print(norm_weights)

        print('genius')


test_1 = Pruning()
# test_1.__global_pruning__()
test_1.__local_pruning__()


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
