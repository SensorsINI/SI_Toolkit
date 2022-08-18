import torch
import torch.nn as nn

import collections
from types import SimpleNamespace


def load_pretrained_net_weights(net, pt_path):
    """
    A function loading parameters (weights and biases) from a previous training to a net RNN instance
    :param net: An instance of RNN
    :param pt_path: path to .pt file storing weights and biases
    :return: No return. Modifies net in place.
    """

    device = get_device()

    pre_trained_model = torch.load(pt_path, map_location=device)
    print("Loading Model: ", pt_path)
    print('')

    pre_trained_model = list(pre_trained_model.items())
    new_state_dict = collections.OrderedDict()
    count = 0
    num_param_key = len(pre_trained_model)
    for key, value in net.state_dict().items():
        if count >= num_param_key:
            break
        layer_name, weights = pre_trained_model[count]
        new_state_dict[key] = weights
        # print("Pre-trained Layer: %s - Loaded into new layer: %s" % (layer_name, key))
        count += 1
    print('')
    net.load_state_dict(new_state_dict)


def compose_net_from_net_name(net_name,
                              inputs_list,
                              outputs_list,
                              time_series_length,
                              batch_size=None,
                              stateful=False):

    net = Sequence(net_name=net_name, inputs_list=inputs_list, outputs_list=outputs_list, batch_size=batch_size)

    print('Constructed a neural network of type {}, with {} hidden layers with sizes {} respectively.'
          .format(net.net_type, len(net.h_size), ', '.join(map(str, net.h_size))))

    # Compose net_info
    net_info = SimpleNamespace()
    net_info.net_name = net_name
    net_info.inputs = inputs_list
    net_info.outputs = outputs_list
    net_info.net_type = net.net_type

    return net, net_info


class Sequence(nn.Module):
    """"
    Our Network class.
    """

    def __init__(self, net_name, inputs_list, outputs_list, batch_size):
        super(Sequence, self).__init__()
        """Initialization of an RNN instance
        We assume that inputs may be both commands and state variables, whereas outputs are always state variables
        """

        # Check if GPU is available. If yes device='cuda:0' if not device='cpu'
        self.device = get_device()

        self.net_name = net_name
        self.net_full_name = None

        self.batch_size = batch_size

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

        if 'GRU' in names:
            self.net_type = 'GRU'
        elif 'LSTM' in names:
            self.net_type = 'LSTM'
        elif 'Dense' in names:
            self.net_type = 'Dense'
        else:
            self.net_type = 'RNN-Basic'

        # Construct network

        if self.net_type == 'GRU':
            self.net_cell = [nn.GRUCell(len(inputs_list), self.h_size[0]).to(self.device)]
            for i in range(len(self.h_size) - 1):
                self.net_cell.append(nn.GRUCell(self.h_size[i], self.h_size[i + 1]).to(self.device))
        elif self.net_type == 'LSTM':
            self.net_cell = [nn.LSTMCell(len(inputs_list), self.h_size[0]).to(self.device)]
            for i in range(len(self.h_size) - 1):
                self.net_cell.append(nn.LSTMCell(self.h_size[i], self.h_size[i + 1]).to(self.device))
        elif self.net_type == 'Dense':
            self.net_cell = [nn.Linear(len(inputs_list), self.h_size[0]).to(self.device)]
            for i in range(len(self.h_size) - 1):
                self.net_cell.append(nn.Linear(self.h_size[i], self.h_size[i + 1]).to(self.device))
        else:
            self.net_cell = [nn.RNNCell(len(inputs_list), self.h_size[0]).to(self.device)]
            for i in range(len(self.h_size) - 1):
                self.net_cell.append(nn.RNNCell(self.h_size[i], self.h_size[i + 1]).to(self.device))

        self.linear = nn.Linear(self.h_size[-1], len(outputs_list))  # RNN out

        self.layers = nn.ModuleList([])
        for cell in self.net_cell:
            self.layers.append(cell)
        self.layers.append(self.linear)

        # Count data samples (=time steps)
        self.sample_counter = 0
        # Declaration of the variables keeping internal state of GRU hidden layers
        self.h = [None] * len(self.h_size)
        self.c = [None] * len(self.h_size)  # Internal state cell - only matters for LSTM

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
        self.sample_counter = 0
        self.reset_internal_states()

    def forward(self, network_input):

        """
        Input is expected to be [batch size, time horizon, features] to make it same as TF
        """

        network_input = torch.transpose(network_input, 0, 1)  # Results in [time horizon, batch size, features (input size)]

        outputs = []

        if self.batch_size is None:
            batch_size = network_input.size(1)
            if self.h[0] is None:
                self.reset_internal_states(batch_size)

        for iteration, input_t in enumerate(network_input.chunk(network_input.size(0), dim=0)):

            # Propagate input through RNN layers
            if self.net_type == 'LSTM':
                self.h[0], self.c[0] = self.layers[0](input_t.squeeze(0), (self.h[0], self.c[0]))
                for i in range(len(self.h_size) - 1):
                    self.h[i + 1], self.c[i + 1] = self.layers[i + 1](self.h[i], (self.h[i + 1], self.c[i + 1]))
            elif self.net_type == 'GRU' or self.net_type == 'RNN-Basic':
                self.h[0] = self.layers[0](input_t.squeeze(0), self.h[0])
                for i in range(len(self.h_size) - 1):
                    self.h[i + 1] = self.layers[i + 1](self.h[i], self.h[i + 1])
            elif self.net_type == 'Dense':
                self.h[0] = self.layers[0](input_t.squeeze(0))
                for i in range(len(self.h_size) - 1):
                    self.h[i + 1] = self.layers[i + 1](self.h[i])

            output = self.layers[-1](self.h[-1])

            outputs += [output]
            self.sample_counter = self.sample_counter + 1

        return torch.stack(outputs, 1)

    def reset_internal_states(self, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

        if batch_size is None:
            self.h = [None] * len(self.h_size)
            self.c = [None] * len(self.h_size)  # Internal state cell - only matters for LSTM
        else:
            for i in range(len(self.h_size)):  # For Dense network h keeps intermediate results
                self.h[i] = torch.zeros(batch_size, self.h_size[i], dtype=torch.float).to(self.device)  # [Batch size, output of RNN layer]
                if self.net_type == 'LSTM':
                    self.c[i] = torch.zeros(batch_size, self.h_size[i], dtype=torch.float).to(self.device)


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