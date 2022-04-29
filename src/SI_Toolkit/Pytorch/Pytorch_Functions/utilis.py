import torch
import torch.nn as nn

import random as rnd

import numpy as np

import os

import shutil
from shutil import copy as shutil_copy

import collections

from types import SimpleNamespace

from datetime import datetime

try:
    # Use gitpython to get a current revision number and use it in description of experimental data
    from git import Repo
except:
    pass

from SI_Toolkit.load_and_normalize import load_normalization_info, get_sampling_interval_from_normalization_info, \
    calculate_normalization_info

def get_device():
    """
    Small function to correctly send data to GPU or CPU depending what is available
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

# Set seeds everywhere required to make results reproducible
def set_seed(args):
    seed = args.seed
    rnd.seed(seed)
    np.random.seed(seed)
    

def get_net(a,
            # If any of arguments provided it overwrite what is given in a
            time_series_length=None,
            batch_size=None,
            stateful=False,
            device='cpu'):
    """
    A quite big (too big?) chunk of creating a network, its associated net_info variable
    and loading associated normalization info.
    It accepts an object a (like SimpleNamespace) which dependent on the mode of operation
        must have following attributes:
    1) Creating new network:
        a.net_name (short name), a.inputs, a.outputs, a.wash_out_len, a.path_to_normalization_info,
    2) Reload:
        a.net_name ('last' or full-name with index suffix), a.path_to_models, a.wash_out_len

    The action to take is decided based on provided net_name.
    It also deletes the folder if txt or ckpt file is missing.
    """

    # region If length of timeseries to be fed into net not provided get it as a sum: wash_out_len + post_wash_out_len
    if time_series_length is None:
        time_series_length = a.wash_out_len + a.post_wash_out_len
    # endregion

    # region Load/create rnn instance, its log and normalization

    # Check if the last part of the name is a sole number
    # If yes user provided full name and this RNN should be loaded

    last_part_of_net_name = a.net_name.split('-')[-1]
    net_name_is_a_full_name = all(c in "0123456789" for c in last_part_of_net_name)

    if net_name_is_a_full_name or a.net_name == 'last':
        load_pretrained = True
    else:
        load_pretrained = False

    # We load a pretrained network
    if load_pretrained:
        net_not_found = True
        # In case net_name is 'last' iterate till a valid file is found
        while net_not_found:
            # region In case net_name is 'last' we have to first find (full) name of the last trained net
            if a.net_name == 'last':
                try:
                    directory = a.path_to_models
                    path_to_latest_model_directory = \
                        max([os.path.join(directory, d) for d in os.listdir(directory)], key=os.path.getctime)
                    # The net full name is the same as folder name in which it is stored
                    parent_net_name = os.path.basename(os.path.normpath(path_to_latest_model_directory))
                except ValueError:
                    raise ValueError('No information about any pretrained network found at {}'.format(a.path_to_models))
            else:
                parent_net_name = a.net_name

            # After above if statement we have parent_net_name and can load it
            print('Loading a pretrained network with the full name: {}'.format(parent_net_name))
            print('')

            # endregion

            # region Ensure that needed txt file are present in the indicated folder
            # They might be missing e.g. if a previous training session was terminated prematurely
            txt_filename = parent_net_name + '.txt'
            txt_path = a.path_to_models + parent_net_name + '/' + txt_filename
            if not os.path.isfile(txt_path):
                txt_not_found_str = 'The corresponding .txt file is missing' \
                                    '(information about inputs and outputs) at the location {}' \
                    .format(txt_path)
                if a.net_name == 'last':
                    print(txt_not_found_str)
                    print('I delete the corresponding folder and try to search again')
                    print('')
                    os.remove(path_to_latest_model_directory)
                    continue
                else:
                    raise FileNotFoundError(txt_not_found_str)

            # endregion

            # region Get information about the pretrained network from the associated txt file
            with open(txt_path, newline='') as f:
                lines = f.read().splitlines()

            for i in range(len(lines)):
                if lines[i] == 'NET NAME:':
                    net_name = lines[i + 1].rstrip("\n")
                    continue
                if lines[i] == 'NET FULL NAME:':
                    net_full_name = lines[i + 1].rstrip("\n")
                    continue
                if lines[i] == 'INPUTS:':
                    inputs = lines[i + 1].rstrip("\n").split(sep=', ')
                    continue
                if lines[i] == 'OUTPUTS:':
                    outputs = lines[i + 1].rstrip("\n").split(sep=', ')
                    continue
                if lines[i] == 'TYPE:':
                    net_type = lines[i + 1].rstrip("\n").split(sep=', ')
                    continue
                if lines[i] == 'NORMALIZATION:':
                    path_to_normalization_info = lines[i + 1].rstrip("\n")
                    continue
                if lines[i] == 'SAMPLING INTERVAL:':
                    net_sampling_interval = float(lines[i + 1].rstrip("\n")[:-2])
                    continue
                if lines[i] == 'WASH_OUT_LENGTH:':
                    net_wash_out_len = float(lines[i + 1].rstrip("\n"))
                    continue

            print('Inputs to the loaded network: {}'.format(', '.join(map(str, inputs))))
            print('Outputs from the loaded network: {}'.format(', '.join(map(str, outputs))))
            print('')

            # endregion

            # region Recreate pretrained network

            # Recreate network architecture
            net, net_info = compose_net_from_net_name(net_name, inputs, outputs,
                                                      time_series_length=time_series_length,
                                                      batch_size=batch_size, stateful=stateful)

            # region Load weights from checkpoint file
            ckpt_filenames = [parent_net_name + '.pt', 'ckpt.pt'] # First is old, second is new way of naming ckpt files. The old way resulted in two long paths for Windows
            ckpt_found = False

            ckpt_path = a.path_to_models + parent_net_name + '/' + ckpt_filenames[0]
            if os.path.isfile(ckpt_path + '.index'):
                ckpt_found = True
            if not ckpt_found:
                ckpt_path = a.path_to_models + parent_net_name + '/' + ckpt_filenames[1]
                if os.path.isfile(ckpt_path + '.index'):
                    ckpt_found = True
            if not ckpt_found:
                ckpt_not_found_str = 'The corresponding .ckpt file is missing' \
                                     '(information about weights and biases). \n' \
                                     'it was not found neither at the location {} nor at {}' \
                    .format(a.path_to_models + parent_net_name + '/' + ckpt_filenames[0], ckpt_path)

                if a.net_name == 'last':
                    print(ckpt_not_found_str)
                    print('I delete the corresponding folder and try to search again')
                    print('')
                    shutil.rmtree(path_to_latest_model_directory)
                    continue
                else:
                    raise FileNotFoundError(ckpt_not_found_str)

            # Load the pretrained weights
            load_pretrained_net_weights(net, ckpt_path, device)

            # net_info.wash_out_len = a.wash_out_len

            # endregion
            print('Model loaded from a checkpoint.')

            # If we got to this point we know that the network was found and we do not need to repeat while loop
            net_not_found = False

            # endregion

            # region Save the path to associated normalization file to net_info
            net_info.path_to_normalization_info = path_to_normalization_info
            # endregion

            net_info.parent_net_name = parent_net_name
            # This is the full name of pretrained net. A new full name will be given if the training is resumed
            net_info.net_full_name = net_full_name

            net_info.path_to_net = a.path_to_models + parent_net_name

            # endregion


    else:

        # region Create a new network according to provided parameters

        print('')
        print('No pretrained network specified. I will train a network from scratch.')
        print('')

        net, net_info = compose_net_from_net_name(a.net_name, a.inputs, a.outputs,
                                                  time_series_length=time_series_length,
                                                  batch_size=batch_size, stateful=stateful)

        # endregion

        # region Save the path to associated normalization file to net_info
        if a.path_to_normalization_info is not None:
            net_info.path_to_normalization_info = a.path_to_normalization_info
        else:
            net_info.path_to_normalization_info = None

        # endregion

        net_info.parent_net_name = 'Network trained from scratch'
        net_info.path_to_net = None  # Folder for net not yer created

    # endregion

    # region Save wash-out length to net_info
    try:
        net_info.wash_out_len = a.wash_out_len
    except AttributeError:
        print('Wash out not defined.')
    # endregion

    return net, net_info



def load_pretrained_net_weights(net, pt_path, device):
    """
    A function loading parameters (weights and biases) from a previous training to a net RNN instance
    :param net: An instance of RNN
    :param pt_path: path to .pt file storing weights and biases
    :return: No return. Modifies net in place.
    """
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
    

class Sequence(nn.Module):
    """"
    Our Network class.
    """

    def __init__(self, net_name, inputs_list, outputs_list):
        super(Sequence, self).__init__()
        """Initialization of an RNN instance
        We assume that inputs may be both commands and state variables, whereas outputs are always state variables
        """

        # Check if GPU is available. If yes device='cuda:0' if not device='cpu'
        self.device = get_device()

        self.net_name = net_name
        self.net_full_name = None

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
            self.net_cell = [nn.GRUCell(len(inputs_list), self.h_size[0]).to(get_device())]
            for i in range(len(self.h_size) - 1):
                self.net_cell.append(nn.GRUCell(self.h_size[i], self.h_size[i + 1]).to(get_device()))
        elif self.net_type == 'LSTM':
            self.net_cell = [nn.LSTMCell(len(inputs_list), self.h_size[0]).to(get_device())]
            for i in range(len(self.h_size) - 1):
                self.net_cell.append(nn.LSTMCell(self.h_size[i], self.h_size[i + 1]).to(get_device()))
        elif self.net_type == 'Dense':
            self.net_cell = [nn.Linear(len(inputs_list), self.h_size[0]).to(get_device())]
            for i in range(len(self.h_size) - 1):
                self.net_cell.append(nn.Linear(self.h_size[i], self.h_size[i + 1]).to(get_device()))
        else:
            self.net_cell = [nn.RNNCell(len(inputs_list), self.h_size[0]).to(get_device())]
            for i in range(len(self.h_size) - 1):
                self.net_cell.append(nn.RNNCell(self.h_size[i], self.h_size[i + 1]).to(get_device()))

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
        # Variable keeping the most recent output of RNN
        self.output = None
        # List storing the history of RNN outputs
        self.outputs = []

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
        self.h = [None] * len(self.h_size)
        self.c = [None] * len(self.h_size)
        self.output = None
        self.outputs = []

    def forward(self, rnn_input):

        """
        Predicts future CartPole states IN "OPEN LOOP"
        (at every time step prediction for the next time step is done based on the true CartPole state)
        """

        # Initialize hidden layers - this change at every call as the batch size may vary
        for i in range(len(self.h_size)):
            self.h[i] = torch.zeros(rnn_input.size(1), self.h_size[i], dtype=torch.float).to(self.device)
            self.c[i] = torch.zeros(rnn_input.size(1), self.h_size[i], dtype=torch.float).to(self.device)

        # The for loop takes the consecutive time steps from input plugs them into RNN and save the outputs into a list
        # THE NETWORK GETS ALWAYS THE GROUND TRUTH, THE REAL STATE OF THE CARTPOLE, AS ITS INPUT
        # IT PREDICTS THE STATE OF THE CARTPOLE ONE TIME STEP AHEAD BASED ON TRUE STATE NOW
        for iteration, input_t in enumerate(rnn_input.chunk(rnn_input.size(0), dim=0)):

            # Propagate input through RNN layers
            if self.net_type == 'LSTM':
                self.h[0], self.c[0] = self.layers[0](input_t.squeeze(0), (self.h[0], self.c[0]))
                for i in range(len(self.h_size) - 1):
                    self.h[i + 1], self.c[i + 1] = self.layers[i + 1](self.h[i], (self.h[i + 1], self.c[i + 1]))
            else:
                self.h[0] = self.layers[0](input_t.squeeze(0), self.h[0])
                for i in range(len(self.h_size) - 1):
                    self.h[i + 1] = self.layers[i + 1](self.h[i], self.h[i + 1])
            self.output = self.layers[-1](self.h[-1])

            self.outputs += [self.output]
            self.sample_counter = self.sample_counter + 1

        # In the train mode we want to continue appending the outputs by calling forward function
        # The outputs will be saved internally in the network instance as a list
        # Otherwise we want to transform outputs list to a tensor and return it
        return self.output

    def return_outputs_history(self):
        return torch.stack(self.outputs, 1)


def compose_net_from_net_name(net_name,
                              inputs_list,
                              outputs_list,
                              time_series_length,
                              batch_size=None,
                              stateful=False):

    net = Sequence(net_name=net_name, inputs_list=inputs_list, outputs_list=outputs_list)

    print('Constructed a neural network of type {}, with {} hidden layers with sizes {} respectively.'
          .format(net.net_type, len(net.h_size), ', '.join(map(str, net.h_size))))

    # Compose net_info
    net_info = SimpleNamespace()
    net_info.net_name = net_name
    net_info.inputs = inputs_list
    net_info.outputs = outputs_list
    net_info.net_type = net.net_type

    return net, net_info


def create_full_name(net_info, path_to_models):
    idx_end_prefix = net_info.net_name.find('-')  # finds first occurrence

    net_full_name = net_info.net_name[:idx_end_prefix + 1] \
                    + str(len(net_info.inputs)) + 'IN-' \
                    + net_info.net_name[idx_end_prefix + 1:] \
                    + '-' + str(len(net_info.outputs)) + 'OUT'

    net_index = 0
    while True:
        path_to_dir = path_to_models + net_full_name + '-' + str(net_index)
        if os.path.isdir(path_to_dir):
            pass
        else:
            net_full_name += '-' + str(net_index)
            os.makedirs(path_to_dir)
            break

        net_index += 1

    print('Full name given to the currently trained network is {}.'.format(net_full_name))
    print('')
    net_info.net_full_name = net_full_name
    net_info.path_to_net = path_to_dir + '/'


def create_log_file(net_info, a):
    date_now = datetime.now().strftime('%Y-%m-%d')
    time_now = datetime.now().strftime('%H:%M:%S')
    try:
        repo = Repo()
        git_revision = repo.head.object.hexsha
    except:
        git_revision = 'unknown'

    txt_path = a.path_to_models + net_info.net_full_name + '/' + net_info.net_full_name + '.txt'
    f = open(txt_path, 'w')
    f.write('CREATED:\n')
    f.write(date_now + ' at time ' + time_now)
    f.write('\n\nWITH GIT REVISION:\n')
    f.write(git_revision)
    f.write('\n\nNET NAME:\n')
    f.write(net_info.net_name)
    f.write('\n\nNET FULL NAME:\n')
    f.write(net_info.net_full_name)
    f.write('\n\nINPUTS:\n')
    f.write(', '.join(map(str, net_info.inputs)))
    f.write('\n\nOUTPUTS:\n')
    f.write(', '.join(map(str, net_info.outputs)))
    f.write('\n\nTYPE:\n')
    f.write(net_info.net_type)
    f.write('\n\nNORMALIZATION:\n')
    f.write(net_info.path_to_normalization_info)
    f.write('\n\nSAMPLING INTERVAL:\n')
    f.write('{} s'.format(net_info.sampling_interval))
    f.write('\n\nPARENT NET:\n')
    f.write(net_info.parent_net_name)
    f.write('\n\nWASH OUT LENGTH:\n')
    f.write(str(net_info.wash_out_len))

    f.write('\n\nTRAINING_FILES:\n')
    if type(a.training_files) is list:
        for path in a.training_files:
            f.write('     ' + path + '\n')
    else:
        f.write(a.training_files)

    f.write('\n\nVALIDATION_FILES:\n')
    if type(a.validation_files) is list:
        for path in a.validation_files:
            f.write('     ' + path + '\n')
    else:
        f.write(a.training_files)

    f.write('\n\nTEST_FILES:\n')
    if type(a.test_files) is list:
        for path in a.test_files:
            f.write('     ' + path + '\n')
    else:
        f.write(a.training_files)

    f.close()


def get_norm_info_for_net(net_info, files_for_normalization=None):
    if net_info.parent_net_name == 'Network trained from scratch':
        # In this case I can either calculate a new normalization info based on training data
        if net_info.path_to_normalization_info is None:
            if files_for_normalization is None:
                raise ValueError('You have to provide either normalization info or data files based in which it should be calculated.')
            normalization_info, net_info.path_to_normalization_info = calculate_normalization_info(files_for_normalization,
                                                                                                   plot_histograms=False,
                                                                                                   user_correction=False,
                                                                                                   path_to_norm_info=net_info.path_to_net)
        else:
            normalization_info = load_normalization_info(net_info.path_to_normalization_info)
            shutil_copy(net_info.path_to_normalization_info, net_info.path_to_net)
            net_info.path_to_normalization_info = net_info.path_to_net + os.path.basename(net_info.path_to_normalization_info)
    else:
        # In this case (retraining) we need to provide a normalization info.
        # This normalization info should in general come from the folder of retrained network,
        #  however it is also compatible with older version of the program with normalization info placed in a different folder
        if net_info.path_to_normalization_info is None:
            raise ValueError('You must provide normalization info for retraining existing network')
        normalization_info = load_normalization_info(net_info.path_to_normalization_info)
        try: shutil_copy(net_info.path_to_normalization_info, net_info.path_to_net)
        except: pass
        net_info.path_to_normalization_info = os.path.join(net_info.path_to_net, os.path.basename(
            net_info.path_to_normalization_info))

    # region Get sampling interval from normalization info
    # TODO: this does not really fits here put is too small for me to create separate function
    try:
        net_info.sampling_interval = get_sampling_interval_from_normalization_info(net_info.path_to_normalization_info)
    except ValueError:
        net_info.sampling_interval = None
        print('sampling_interval unknown')
    # endregion

    return normalization_info


# Print parameter count
# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def print_parameter_count(net):
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    pytorch_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('::: # network all parameters: ' + str(pytorch_total_params))
    print('::: # network trainable parameters: ' + str(pytorch_trainable_params))
    print('')