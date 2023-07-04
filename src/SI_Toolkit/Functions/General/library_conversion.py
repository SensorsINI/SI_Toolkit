import torch
from SI_Toolkit.Functions.Pytorch.Network import get_device
from tensorflow import keras
import numpy as np


def convert(net_original, net_info, time_series_length, batch_size, stateful, construct_network):
    if net_info.library == 'Pytorch':  # Now we want to create a network different than the one we have
        from SI_Toolkit.Functions.TF.Network import compose_net_from_net_name
    else:
        from SI_Toolkit.Functions.Pytorch.Network import compose_net_from_net_name

    # Create network architecture
    net, _ = compose_net_from_net_name(net_info.net_name, net_info.inputs, net_info.outputs,
                                              time_series_length=time_series_length,
                                              batch_size=batch_size, stateful=stateful,
                                              construct_network=construct_network)

    if net_info.library == 'Pytorch':
        net = convert_torch_to_tf(net_original, net)
        net_info.library = 'TF'
    else:
        net = convert_tf_to_torch(net_original, net)
        net_info.library = 'Pytorch'

    return net, net_info


# https://discuss.pytorch.org/t/transferring-weights-from-keras-to-pytorch/9889
def convert_tf_to_torch(km, pm):

    device = get_device()

    weight_dict = dict()
    for i in range(len(km.layers)):
        weight_dict[km.layers[i].get_config()['name'] + '.weight'] = np.transpose(km.layers[i].get_weights()[0], (1, 0))
        weight_dict[km.layers[i].get_config()['name'] + '.bias'] = km.layers[i].get_weights()[1]

    pyt_state_dict = pm.state_dict()
    for key in pyt_state_dict.keys():
        pyt_state_dict[key] = torch.from_numpy(weight_dict[key]).to(device)
    pm.load_state_dict(pyt_state_dict)
    return pm


# https://gist.github.com/chirag1992m/4c1f2cb27d7c138a4dc76aeddfe940c2
def convert_torch_to_tf(pm, km):
    pyt_state_dict = pm.state_dict()
    # Iterate over the layers in the Keras model and the weights in the PyTorch model
    for idx, layer in enumerate(km.layers):
        name = layer.name
        weights = np.transpose(pyt_state_dict[name + '.weight'].numpy(), (1, 0))
        bias = pyt_state_dict[name + '.bias'].numpy()
        km.layers[idx].set_weights([weights, bias])
    return km
