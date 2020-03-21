from __future__ import print_function
import torch.nn.utils
from MNIST_quick_implementation import NNet

ENTIRE_MODEL_FILENAME = "mnist_cnn.pt"
MODEL_WEIGHTS = "mnist_weights_cnn.pt"

# reverse operation
model_new_weights = NNet()
model_new_weights.load_state_dict(torch.load(MODEL_WEIGHTS))
# model_new = torch.load(ENTIRE_MODEL_FILENAME)
# model.load_state_dict(torch.load(ENTIRE_MODEL_FILENAME))

# Print model's state dict
# print("-----Model's State dict after training-----")
# for name, param in model_new_weights.named_parameters():
#     print('==========')
#     print(name, ':', param.requires_grad)
#     print('==========')
# for layer in model_new_weights.named_parameters():
#     print('=========', layer, '==========')
#     print('=====', model_new_weights.named_parameters(), '====')

# Work in progress on weights from different layers
print('========================= LOOP ON LAYERS ================================')
modules = ['conv1', 'conv2', 'fc1', 'fc2']
weights_array_global = torch.empty([32, 9]).view(-1)
print('empty tensor', weights_array_global.shape)
for layer in modules:
    print('WEIGHT ARRAY GLOBAL', weights_array_global.shape)
    print('=================LOCAL=================')
    weights_array_local = getattr(model_new_weights, layer).weight
    print('WEIGHT LAYER', weights_array_local)
    print('Tensors shape: ', weights_array_local.shape)
    weights_array_local = (weights_array_local.view(-1))
    print('WEIGHTS ARRAY RESHAPED:', weights_array_local)
    print('Tensors RESHAPE: ', weights_array_local.shape)
    weights_array_local_ranked = torch.sort(weights_array_local)
    print('SORTED LOCAL WEIGHTS', weights_array_local_ranked)
    norm_weights = torch.norm(weights_array_local)  # Gives the norm of tensors, useful ? Not sure
    print('NORM TENSORS: ', norm_weights)
    # weights_array_global += torch.cat([weights_array_global, weights_array_local]) PROBLEM OF SIZES TENSORS
    # masked_weight = torch.scatter() TO CONTINUE HERE
    print('================END LOCAL================')

print('================GLOBAL================')
print(torch.sort(weights_array_global))
print(weights_array_global.shape)
print('======================================')

# print(list(module.named_parameters(layer)))
# norm_weights = torch.norm(model_new_weights.named_parameters('weight'))
# print(norm_weights)

print('==================================')
print('Normalization')
print('==================================')

print('genius')


# Uncomment if experiments on optimizers
# Print optimizer's state dict
# print("Optimizer State dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])
