# -*- coding: utf-8 -*-

import argparse
import glob
import yaml, os

config_training = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_training.yml'), 'r'), Loader=yaml.FullLoader)
config_testing = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_testing.yml'), 'r'), Loader=yaml.FullLoader)


net_name = config_training['modeling']['NET_NAME']

# net_name = 'GRU-6IN-16H1-16H2-5OUT-0'
# net_name = 'Dense-6IN-16H1-16H2-5OUT-0'
# net_name = 'Dense-16H1-16H2'
# Path to trained models and their logs
PATH_TO_MODELS = config_training["paths"]["PATH_TO_EXPERIMENT_FOLDERS"] + config_training['paths']['path_to_experiment'] + "Models/"

PATH_TO_NORMALIZATION_INFO = config_training["paths"]["PATH_TO_EXPERIMENT_FOLDERS"] + config_training['paths']['path_to_experiment'] + "NormalizationInfo/"
PATH_TO_NORMALIZATION_INFO += os.listdir(PATH_TO_NORMALIZATION_INFO)[0]

# The following paths to dictionaries may be replaced by the list of paths to data files.
TRAINING_FILES = config_training["paths"]["PATH_TO_EXPERIMENT_FOLDERS"] + config_training['paths']['path_to_experiment'] + "/Recordings/Train/"
VALIDATION_FILES = config_training["paths"]["PATH_TO_EXPERIMENT_FOLDERS"] + config_training['paths']['path_to_experiment'] + "/Recordings/Validate/"
TEST_FILES = config_training["paths"]["PATH_TO_EXPERIMENT_FOLDERS"] + config_training['paths']['path_to_experiment'] + "/Recordings/Test/"


# region Set inputs and outputs

control_inputs = config_training['training_default']['control_inputs']
state_inputs = config_training['training_default']['state_inputs']
setpoint_inputs = config_training['training_default']['setpoint_inputs']
outputs = config_training['training_default']['outputs']

EPOCHS = config_training['training_default']['EPOCHS']
BATCH_SIZE = config_training['training_default']['BATCH_SIZE']
SEED = config_training['training_default']['SEED']
LR = config_training['training_default']['LR']

WASH_OUT_LEN = config_training['training_default']['WASH_OUT_LEN']
POST_WASH_OUT_LEN = config_training['training_default']['POST_WASH_OUT_LEN']
ON_FLY_DATA_GENERATION = config_training['training_default']['ON_FLY_DATA_GENERATION']
NORMALIZE = config_training['training_default']['NORMALIZE']

TEST_LEN = config_testing['testing']['TEST_LEN']
START_IDX = config_testing['testing']['START_IDX']
MAX_HORIZON = config_testing['testing']['MAX_HORIZON']

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
    parser.add_argument('--setpoint_inputs', default=setpoint_inputs,
                        help='List of setpoint inputs to neural network')
    parser.add_argument('--outputs', default=outputs,
                        help='List of outputs from neural network')

    # Only valid for graphical testing:
    # parser.add_argument('--test_len', default=TEST_LEN,
    #                     help='For graphical testing only test_len samples from first test file is taken.')
    # parser.add_argument('--test_start_idx', default=START_IDX, type=int, help='Indicates from which point data from test file should be taken.')
    # parser.add_argument('--test_max_horizon', default=MAX_HORIZON, type=int,
    #                     help='Indicates prediction horizon for testing.')

    # Training only:
    parser.add_argument('--wash_out_len', default=WASH_OUT_LEN, type=int, help='Number of timesteps for a wash-out sequence, min is 0')
    parser.add_argument('--post_wash_out_len', default=POST_WASH_OUT_LEN, type=int,
                        help='Number of timesteps after wash-out sequence (this is used to calculate loss), min is 1')

    # Training parameters
    parser.add_argument('--num_epochs', default=EPOCHS, type=int, help='Number of epochs of training')
    parser.add_argument('--batch_size', default=1, type=int, help='Size of a batch')
    parser.add_argument('--seed', default=SEED, type=int, help='Set seed for reproducibility')
    parser.add_argument('--lr', default=LR, type=float, help='Learning rate')

    parser.add_argument('--path_to_models', default=PATH_TO_MODELS, type=str,
                        help='Path where to save/ from where to load models')

    parser.add_argument('--path_to_normalization_info', default=PATH_TO_NORMALIZATION_INFO, type=str,
                        help='Path where the cartpole data is saved')

    parser.add_argument('--on_fly_data_generation', default=ON_FLY_DATA_GENERATION, type=bool,
                        help='Generate data for training during training, instead of loading previously saved data')
    parser.add_argument('--normalize', default=NORMALIZE, type=bool, help='Make all data between 0 and 1')

    args = parser.parse_args()

    # Make sure that the provided lists of inputs and outputs are in alphabetical order

    if args.post_wash_out_len < 1:
        raise ValueError('post_wash_out_len, the part relevant for loss calculation must be at least 1, also for dense network')

    if args.control_inputs is not None:
        args.control_inputs = sorted(args.control_inputs)

    if args.state_inputs is not None:
        args.state_inputs = sorted(args.state_inputs)

    if args.setpoint_inputs is not None:
        args.setpoint_inputs = sorted(args.setpoint_inputs)

    args.inputs = args.control_inputs + args.state_inputs + args.setpoint_inputs

    if args.outputs is not None:
        args.outputs = sorted(args.outputs)
    return args

