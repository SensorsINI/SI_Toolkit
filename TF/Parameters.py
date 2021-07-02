# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:29:29 2020

@author: Marcin
"""
import argparse
import glob
import yaml, os

config = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config.yml'), 'r'), Loader=yaml.FullLoader)

net_name = config['modeling']['NET_NAME']

# net_name = 'GRU-6IN-16H1-16H2-5OUT-0'
# net_name = 'Dense-6IN-16H1-16H2-5OUT-0'
# net_name = 'Dense-16H1-16H2'
# Path to trained models and their logs
PATH_TO_MODELS = config["paths"]["PATH_TO_EXPERIMENT_RECORDINGS"] + config['paths']['path_to_experiment'] + "Models/"

PATH_TO_NORMALIZATION_INFO = config["paths"]["PATH_TO_EXPERIMENT_RECORDINGS"] + config['paths']['path_to_experiment'] + "NormalizationInfo/"
PATH_TO_NORMALIZATION_INFO += os.listdir(PATH_TO_NORMALIZATION_INFO)[0]

# The following paths to dictionaries may be replaced by the list of paths to data files.
TRAINING_FILES = config["paths"]["PATH_TO_EXPERIMENT_RECORDINGS"] + config['paths']['path_to_experiment'] + "/Recordings/Train/"
VALIDATION_FILES = config["paths"]["PATH_TO_EXPERIMENT_RECORDINGS"] + config['paths']['path_to_experiment'] + "/Recordings/Validate/"
TEST_FILES = config["paths"]["PATH_TO_EXPERIMENT_RECORDINGS"] + config['paths']['path_to_experiment'] + "/Recordings/Test/"


# region Set inputs and outputs

control_inputs = config['training_default']['control_inputs']
state_inputs = config['training_default']['state_inputs']
outputs = config['training_default']['outputs']

# For l2race
# control_inputs = ['u1', 'u2']
# state_inputs = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
# outputs = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']

# endregion

def args():
    parser = argparse.ArgumentParser(description='Train a GRU network.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Defining the model
    parser.add_argument('--net_name', default=net_name, type=str,
                        help='Name defining the network.'
                             'It has to have the form:'
                             '(RNN type [GRU/LSTM]/Dense)-(size first hidden layer)H1-(size second hidden layer)H2-...'
                             'e.g. GRU-64H1-64H2-32H3')

    parser.add_argument('--training_files', default=TRAINING_FILES, type=str,
                        help='File name of the recording to be used for training the RNN'
                             'e.g. oval_easy.csv ')
    parser.add_argument('--validation_files', default=VALIDATION_FILES, type=str,
                        help='File name of the recording to be used for validating the RNN'
                             'e.g. oval_easy_test.csv ')
    parser.add_argument('--test_files', default=TEST_FILES, type=str,
                        help='File name of the recording to be used for validating the RNN'
                             'e.g. oval_easy_test.csv ')

    parser.add_argument('--control_inputs', default=control_inputs,
                        help='List of control inputs to neural network')
    parser.add_argument('--state_inputs', default=state_inputs,
                        help='List of state inputs to neural network')
    parser.add_argument('--outputs', default=outputs,
                        help='List of outputs from neural network')

    # Only valid for graphical testing:
    parser.add_argument('--test_len', default=50, type=int,
                        help='For graphical testing only test_len samples from first test file is taken.')
    parser.add_argument('--test_start_idx', default=100, type=int, help='Indicates from which point data from test file should be taken.')
    parser.add_argument('--test_max_horizon', default=5, type=int,
                        help='Indicates prediction horizon for testing.')

    # Training only:
    parser.add_argument('--wash_out_len', default=10, type=int, help='Number of timesteps for a wash-out sequence')
    parser.add_argument('--post_wash_out_len', default=50, type=int,
                        help='Number of timesteps after wash-out sequence (this is used to calculate loss)')

    # Training parameters
    parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs of training')
    parser.add_argument('--batch_size', default=16, type=int, help='Size of a batch')
    parser.add_argument('--seed', default=1873, type=int, help='Set seed for reproducibility')
    parser.add_argument('--lr', default=1.0e-3, type=float, help='Learning rate')

    parser.add_argument('--path_to_models', default=PATH_TO_MODELS, type=str,
                        help='Path where to save/ from where to load models')

    parser.add_argument('--path_to_normalization_info', default=PATH_TO_NORMALIZATION_INFO, type=str,
                        help='Path where the cartpole data is saved')

    parser.add_argument('--on_fly_data_generation', default=False, type=bool,
                        help='Generate data for training during training, instead of loading previously saved data')
    parser.add_argument('--normalize', default=True, type=bool, help='Make all data between 0 and 1')

    args = parser.parse_args()

    # Make sure that the provided lists of inputs and outputs are in alphabetical order

    if args.control_inputs is not None:
        args.control_inputs = sorted(args.control_inputs)

    if args.state_inputs is not None:
        args.state_inputs = sorted(args.state_inputs)

    if args.control_inputs is not None and args.state_inputs is not None:
        args.inputs = args.control_inputs+args.state_inputs
    elif args.control_inputs is not None:
        args.inputs = args.control_inputs
    elif args.state_inputs is not None:
        args.inputs = args.state_inputs
    else:
        args.inputs = None

    if args.outputs is not None:
        args.outputs = sorted(args.outputs)
    return args

