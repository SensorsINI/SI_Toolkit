import torch
import torch.nn as nn

import sys
import os

import collections
from types import SimpleNamespace
from copy import deepcopy as dcp


dict_translate = {'rnn.weight_ih_l0': 'network_head.weight_ih_l0',
                     'rnn.weight_hh_l0': 'network_head.weight_hh_l0',
                     'rnn.bias_ih_l0': 'network_head.bias_ih_l0',
                     'rnn.bias_hh_l0': 'network_head.bias_hh_l0',
                     'rnn.weight_ih_l1': 'network_head.weight_ih_l1',
                     'rnn.weight_hh_l1': 'network_head.weight_hh_l1',
                     'rnn.bias_ih_l1': 'network_head.bias_ih_l1',
                     'rnn.bias_hh_l1': 'network_head.bias_hh_l1',
                     'cl.weight': 'cl.weight',
                     'cl.bias': 'cl.bias'}

def load_pretrained_net_weights(net, pt_path):
    """
    A function loading parameters (weights and biases) from a previous training to a net RNN instance
    :param net: An instance of RNN
    :param pt_path: path to .pt file storing weights and biases
    :return: No return. Modifies net in place.
    """

    device = get_device()

    pre_trained_model_dict = torch.load(pt_path, map_location=device)
    if 'state_dict' in pre_trained_model_dict:
        pre_trained_model_dict = pre_trained_model_dict['state_dict']
    print("Loading Model: ", pt_path)
    print('')

    new_state_dict = net.state_dict()

    use_dict_translate = False
    for key1, key2 in zip(pre_trained_model_dict.keys(), new_state_dict.keys()):
        if key1 != key2:
            use_dict_translate = True
            break

    for key, value in new_state_dict.items():
        pretrained_key = key
        if use_dict_translate:
            pretrained_key = dict_translate[key]
        weights = pre_trained_model_dict[pretrained_key]
        new_state_dict[key] = weights
    print('')
    net.load_state_dict(new_state_dict)


def compose_net_from_net_name(net_info,
                              time_series_length,
                              batch_size=None,
                              stateful=False,
                              construct_network='with cells',
                              remove_redundant_dimensions=False):

    if remove_redundant_dimensions == True:
        raise NotImplementedError('Removing redundant dimensions not implemented for Pytorch.')

    net_name = net_info.net_name
    inputs_list = net_info.inputs
    outputs_list = net_info.outputs

    net = Sequence(net_name=net_name, inputs_list=inputs_list, outputs_list=outputs_list,
                   batch_size=batch_size, construct_network=construct_network)

    print('Constructed a neural network of type {}, with {} hidden layers with sizes {} respectively.'
          .format(net.net_type, len(net.h_size), ', '.join(map(str, net.h_size))))

    net_info.net_type = net.net_type
    net_info.delta_gru_dict = net.delta_gru_dict

    return net, net_info


class Sequence(nn.Module):
    """"
    Our Network class.
    """

    def __init__(self, net_name, inputs_list, outputs_list, batch_size, construct_network='with cells'):
        super(Sequence, self).__init__()
        """Initialization of an RNN instance
        We assume that inputs may be both commands and state variables, whereas outputs are always state variables
        """

        # Check if GPU is available. If yes device='cuda:0' if not device='cpu'
        self.device = get_device()

        self.construct_network = construct_network

        self.net_name = net_name
        self.net_full_name = None

        self.batch_size = batch_size

        self.delta_gru_dict = None

        # Get the information about network architecture from the network name
        # Split the names into "LSTM/GRU", "128H1", "64H2" etc.
        names = net_name.split('-')
        layers = ['H1', 'H2', 'H3', 'H4', 'H5']
        self.h_size = []  # Hidden layers sizes
        for name in names:
            for index, layer in enumerate(layers):
                if layer in name:
                    # assign the variable with name obtained from list layers.
                    self.h_size.append(int(name[:-2]))

        if not self.h_size:
            raise ValueError('You have to provide the size of at least one hidden layer in rnn name')

        self.h_number = len(self.h_size)

        if 'DeltaGRU' in names:
            self.net_type = 'DeltaGRU'
        elif 'GRU' in names:
            self.net_type = 'GRU'
        elif 'LSTM' in names:
            self.net_type = 'LSTM'
        elif 'Dense' in names:
            self.net_type = 'Dense'
        else:
            self.net_type = 'RNN-Basic'

        # Construct network
        if self.construct_network == 'with cells':
            if self.net_type == 'Dense':
                self.net_cell = [nn.Linear(len(inputs_list), self.h_size[0]).to(self.device)]
                for i in range(len(self.h_size) - 1):
                    self.net_cell.append(nn.Linear(self.h_size[i], self.h_size[i + 1]).to(self.device))
            elif self.net_type == 'GRU':
                self.net_cell = [nn.GRUCell(len(inputs_list), self.h_size[0]).to(self.device)]
                for i in range(len(self.h_size) - 1):
                    self.net_cell.append(nn.GRUCell(self.h_size[i], self.h_size[i + 1]).to(self.device))
            elif self.net_type == 'DeltaGRU':
                raise ValueError("DeltaGRU can only be created with construct_network == 'with modules'")
            elif self.net_type == 'LSTM':
                self.net_cell = [nn.LSTMCell(len(inputs_list), self.h_size[0]).to(self.device)]
                for i in range(len(self.h_size) - 1):
                    self.net_cell.append(nn.LSTMCell(self.h_size[i], self.h_size[i + 1]).to(self.device))
            elif self.net_type == 'RNN-Basic':
                self.net_cell = [nn.RNNCell(len(inputs_list), self.h_size[0]).to(self.device)]
                for i in range(len(self.h_size) - 1):
                    self.net_cell.append(nn.RNNCell(self.h_size[i], self.h_size[i + 1]).to(self.device))
        elif self.construct_network == 'with modules':
            if self.net_type == 'Dense':
                raise ValueError("Dense can only be created with construct_network == 'with cells'")
            if self.h_size and self.h_size.count(self.h_size[0]) != len(self.h_size):  # List not empty and not elements are the same
                raise ValueError("In the mode construct_network == 'with modules' all hidden layers must have the same size. It is not the case.")

            if self.net_type == 'GRU':
                self.network_head = nn.GRU(input_size=len(inputs_list), hidden_size=self.h_size[0],
                                           num_layers=len(self.h_size))
            elif self.net_type == 'DeltaGRU':
                import yaml
                import SI_Toolkit.Functions.Pytorch as EdgeDRNN_location
                path_to_EdgeDRNN = os.path.join(os.path.dirname(EdgeDRNN_location.__file__), "EdgeDRNN", "python")
                if path_to_EdgeDRNN not in sys.path:
                    sys.path.insert(0, path_to_EdgeDRNN)
                from SI_Toolkit.Functions.Pytorch.EdgeDRNN.python.nnlayers.deltagru import DeltaGRU

                delta_gru_dict = yaml.load(open(os.path.join("SI_Toolkit_ASF", "config_DeltaGRU.yml"), "r"),
                                   Loader=yaml.FullLoader)
                delta_gru_dict['inp_size'] = len(inputs_list)
                delta_gru_dict['rnn_size'] = self.h_size[0]
                delta_gru_dict['rnn_layers'] = len(self.h_size)
                delta_gru_dict['num_classes'] = len(outputs_list)
                self.delta_gru_dict = delta_gru_dict

                self.rnn = DeltaGRU(
                    input_size=delta_gru_dict['inp_size'],
                    hidden_size=delta_gru_dict['rnn_size'],
                    num_layers=delta_gru_dict['rnn_layers'],
                    batch_first=delta_gru_dict['batch_first'],
                    thx=delta_gru_dict['thx'],
                    thh=delta_gru_dict['thh'],
                    qa=delta_gru_dict['qa'],
                    aqi=delta_gru_dict['aqi'],
                    aqf=delta_gru_dict['aqf'],
                    qw=delta_gru_dict['qw'],
                    wqi=delta_gru_dict['wqi'],
                    wqf=delta_gru_dict['wqf'],
                    nqi=delta_gru_dict['afqi'],
                    nqf=delta_gru_dict['afqf'],
                    debug=delta_gru_dict['debug'],
                )
            elif self.net_type == 'LSTM':
                self.network_head = nn.LSTM(input_size=len(inputs_list), hidden_size=self.h_size[0],
                                            num_layers=len(self.h_size))
            elif self.net_type == 'RNN-Basic':
                self.network_head = nn.RNN(input_size=len(inputs_list), hidden_size=self.h_size[0],
                                           num_layers=len(self.h_size))


        if self.construct_network == 'with cells':
            self.layers = nn.ModuleList([])
            self.net_cell.append(nn.Linear(self.h_size[-1], len(outputs_list)).to(self.device))
            for cell in self.net_cell:
                self.layers.append(cell)
        else:
            self.cl = nn.Linear(self.h_size[-1], len(outputs_list)).to(self.device)



        # Declaration of the variables keeping internal state of GRU hidden layers
        self.h = [None] * len(self.h_size)
        self.c = [None] * len(self.h_size)  # Internal state cell - only matters for LSTM

        self.activation_function = nn.Tanh()

        # Send the whole RNN to GPU if available, otherwise send it to CPU
        self.to(self.device)

        print('Constructed a neural network of type {}, with {} hidden layers with sizes {} respectively.'
              .format(self.net_type, len(self.h_size), ', '.join(map(str, self.h_size))))
        print('The inputs are (in this order): {}'.format(', '.join(map(str, inputs_list))))
        print('The outputs are (in this order): {}'.format(', '.join(map(str, outputs_list))))

    def reset(self):
        """
        Reset the network (not the weights!)
        """
        self.reset_internal_states()

    def forward(self, network_input):

        """
        Input is expected to be [batch size, time horizon, features] to make it same as TF
        """

        network_input = torch.transpose(network_input, 0, 1)  # Results in [time horizon, batch size, features (input size)]

        if self.batch_size is None:
            batch_size = network_input.size(1)
            if self.h[0] is None:
                self.reset_internal_states(batch_size=batch_size)

        if self.construct_network == 'with cells':
            outputs = []
            for iteration, input_t in enumerate(network_input.chunk(network_input.size(0), dim=0)):
                if self.net_type == 'Dense':
                    self.h[0] = self.layers[0](input_t.squeeze(0))
                    self.h[0] = self.activation_function(self.h[0])
                    for i in range(len(self.h_size) - 1):
                        self.h[i + 1] = self.layers[i + 1](self.h[i])
                        self.h[i + 1] = self.activation_function(self.h[i + 1])
                elif self.net_type == 'LSTM':
                    self.h[0], self.c[0] = self.layers[0](input_t.squeeze(0), (self.h[0], self.c[0]))
                    for i in range(len(self.h_size) - 1):
                        self.h[i + 1], self.c[i + 1] = self.layers[i + 1](self.h[i], (self.h[i + 1], self.c[i + 1]))
                elif self.net_type == 'GRU' or self.net_type == 'RNN-Basic':
                    self.h[0] = self.layers[0](input_t.squeeze(0), self.h[0])
                    for i in range(len(self.h_size) - 1):
                        self.h[i + 1] = self.layers[i + 1](self.h[i], self.h[i + 1])

                output = self.layers[-1](self.h[-1])

                outputs += [output]

            outputs = torch.stack(outputs, 1)

        elif self.construct_network == 'with modules':
            if self.net_type == 'LSTM':
                outputs, (self.h, self.c) = self.network_head(network_input, (self.h, self.c))
            elif self.net_type == 'GRU' or self.net_type == 'RNN-Basic':
                outputs, self.h = self.network_head(network_input, self.h)
            elif self.net_type == 'DeltaGRU':
                outputs, self.h, _ = self.rnn(network_input, self.h)
            outputs = self.cl(outputs)

            outputs = torch.transpose(outputs, 0, 1)

        return outputs

    def reset_internal_states(self, memory_states_ref=None, batch_size=None):

        if memory_states_ref is not None:
            self.h = dcp(memory_states_ref[0])
            self.c = dcp(memory_states_ref[1])
        else:
            if batch_size is None:
                batch_size = self.batch_size

            if batch_size is None:
                self.h = [None] * len(self.h_size)
                self.c = [None] * len(self.h_size)  # Internal state cell - only matters for LSTM
            else:
                if self.construct_network == 'with cells':
                    for i in range(len(self.h_size)):  # For Dense network h keeps intermediate results
                        self.h[i] = torch.zeros(batch_size, self.h_size[i], dtype=torch.float).to(self.device)  # [Batch size, output of RNN layer]
                        if self.net_type == 'LSTM':
                            self.c[i] = torch.zeros(batch_size, self.h_size[i], dtype=torch.float).to(self.device)
                else:
                    if self.net_type == 'DeltaGRU':
                        self.h = None
                    else:
                        self.h = torch.zeros(len(self.h_size), batch_size, self.h_size[0], dtype=torch.float).to(self.device)  # [Batch size, output of RNN layer]
                    if self.net_type == 'LSTM':
                        self.c = torch.zeros(len(self.h_size), batch_size, self.h_size[0], dtype=torch.float).to(self.device)

    def return_internal_states(self):
        # Different scenarios might happen depending if the network is build with modules or not
        if isinstance(self.h, list):
            h_ref = []
            for i in range(len(self.h)):
                if self.h[i] is None:
                    h_ref.append(None)
                else:
                    h_ref.append(self.h[i].detach().clone())
        else:
            h_ref = self.h.detach().clone()

        if isinstance(self.c, list):
            c_ref = []
            for i in range(len(self.c)):
                if self.c[i] is None:
                    c_ref.append(None)
                else:
                    c_ref.append(self.c[i].detach().clone())
        else:
            c_ref = self.c.detach().clone()


        return [h_ref, c_ref]

# Print parameter count
# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def print_parameter_count(net):
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    pytorch_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('::: # network all parameters: ' + str(pytorch_total_params))
    print('::: # network trainable parameters: ' + str(pytorch_trainable_params))
    print('')


def get_device():
    """
    Small function to correctly send data to GPU or CPU depending what is available
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    # elif torch.backends.mps.is_available(): # Activate M1 GPU - not worth it
    #     device = torch.device("mps")
    else:
        device = torch.device('cpu')
    return device


def _copy_internal_states_to_ref(net, memory_states_ref):
    new_states = net.return_internal_states()
    for i in range(len(memory_states_ref)):
        memory_states_ref[i] = new_states[i]



def _copy_internal_states_from_ref(net, memory_states_ref):
    net.reset_internal_states(memory_states_ref=memory_states_ref)
