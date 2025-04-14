# -*- coding: utf-8 -*-
"""
This function loads parameters for training (TF/Pytorch/GPs) from config,
adds however an option to overwrite config values with command line
"""
import argparse
import glob
import yaml, os

def args():
    parser = argparse.ArgumentParser(description='Train a GRU network.', formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)

    parser.add_argument('-i', '--config_index', default=0, type=int,
                        help='Index of the config file to use (default: 0, which maps to config_training.yml)')

    # Parse known arguments to get config_index
    args, remaining_argv = parser.parse_known_args()

    # Dynamically construct the config file name
    if args.config_index == 0:
        config_file = 'config_training.yml'
    else:
        config_file = f'config_training_{args.config_index}.yml'

    # Load the selected config file
    config_path = os.path.join('SI_Toolkit_ASF', config_file)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The specified config file '{config_path}' does not exist.")

    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    # Now set variables from config
    library = config['library']
    net_name = config['modeling']['NET_NAME']

    # Path to trained models and their logs
    PATH_TO_MODELS = os.path.join(config["paths"]["PATH_TO_EXPERIMENT_FOLDERS"], config['paths']['path_to_experiment'], "Models")

    PATH_TO_NORMALIZATION_INFO = os.path.join(config["paths"]["PATH_TO_EXPERIMENT_FOLDERS"], config['paths']['path_to_experiment'], "NormalizationInfo")

    # Get path to normalisation info as to a newest csv file in indicated folder
    paths = sorted([os.path.join(PATH_TO_NORMALIZATION_INFO, d) for d in os.listdir(PATH_TO_NORMALIZATION_INFO)], key=os.path.getctime)
    for path in paths:
        if path[-4:] == '.csv':
            PATH_TO_NORMALIZATION_INFO = path

    # The following paths to dictionaries may be replaced by the list of paths to data files.
    TRAINING_FILES = os.path.join(config["paths"]["PATH_TO_EXPERIMENT_FOLDERS"], config['paths']['path_to_experiment'], config['paths']['DATA_FOLDER'], "Train")
    VALIDATION_FILES = os.path.join(config["paths"]["PATH_TO_EXPERIMENT_FOLDERS"], config['paths']['path_to_experiment'], config['paths']['DATA_FOLDER'], "Validate")
    TEST_FILES = os.path.join(config["paths"]["PATH_TO_EXPERIMENT_FOLDERS"], config['paths']['path_to_experiment'], config['paths']['DATA_FOLDER'], "Test")

    # region Set inputs and outputs

    control_inputs = config['training_default']['control_inputs']
    state_inputs = config['training_default']['state_inputs']
    setpoint_inputs = config['training_default']['setpoint_inputs']
    outputs = config['training_default']['outputs']
    translation_invariant_variables = config['training_default']['translation_invariant_variables']

    EPOCHS = config['training_default']['EPOCHS']
    BATCH_SIZE = config['training_default']['BATCH_SIZE']
    SEED = config['training_default']['SEED']
    SHIFT_LABELS = config['training_default']['SHIFT_LABELS']

    LR_INITIAL = config['training_default']['LR']['INITIAL']
    REDUCE_LR_ACTIVATED = config['training_default']['LR']['REDUCE_LR_ON_PLATEAU']
    MIN_DELTA = config['training_default']['LR']['MIN_DELTA']
    LR_MINIMAL = config['training_default']['LR']['MINIMAL']
    LR_PATIENCE = config['training_default']['LR']['PATIENCE']
    LR_DECREASE_FACTOR = config['training_default']['LR']['DECREASE_FACTOR']

    WASH_OUT_LEN = config['training_default']['WASH_OUT_LEN']
    POST_WASH_OUT_LEN = config['training_default']['POST_WASH_OUT_LEN']
    ON_FLY_DATA_GENERATION = config['training_default']['ON_FLY_DATA_GENERATION']
    NORMALIZE = config['training_default']['NORMALIZE']
    USE_NNI = config['training_default']['USE_NNI']
    CONSTRUCT_NETWORK = config['training_default']['CONSTRUCT_NETWORK']

    VALIDATE_ALSO_ON_TRAINING_SET = config['training_default']['VALIDATE_ALSO_ON_TRAINING_SET']

    REGULARIZATION = config['REGULARIZATION']

    QUANTIZATION = config['QUANTIZATION']

    PRUNING_ACTIVATED = config['PRUNING']['ACTIVATED']
    PRUNING_SCHEDULE = config['PRUNING']['PRUNING_PARAMS']['PRUNING_SCHEDULE']
    PRUNING_SCHEDULES = config['PRUNING']['PRUNING_SCHEDULES']

    PLOT_WEIGHTS_DISTRIBUTION = config['training_default']['PLOT_WEIGHTS_DISTRIBUTION']

    FILTERS = config['FILTERS']

    CONFIG_SERIES_MODIFICATION = config['CONFIG_SERIES_MODIFICATION']

    # For l2race
    # control_inputs = ['u1', 'u2']
    # state_inputs = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
    # outputs = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']

    # endregion

    # Now create a new parser with the rest of the arguments, using defaults from config
    parser = argparse.ArgumentParser(description='Train a GRU network.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--config_index', default=0, type=int,
                        help='Index of the config file to use (default: 0, which maps to config_training.yml)')

    parser.add_argument('--library', default=library, type=str,
                        help='Decide if you want to use TF or Pytorch for training.')

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
    parser.add_argument('--translation_invariant_variables', default=translation_invariant_variables,
                        help='List of translation_invariant_variables to neural network - shift of the whole series does not change the result')

    # Training only:
    parser.add_argument('--wash_out_len', default=WASH_OUT_LEN, type=int, help='Number of timesteps for a wash-out sequence, min is 0')
    parser.add_argument('--post_wash_out_len', default=POST_WASH_OUT_LEN, type=int,
                        help='Number of timesteps after wash-out sequence (this is used to calculate loss), min is 1')

    # Training parameters
    parser.add_argument('--num_epochs', default=EPOCHS, type=int, help='Number of epochs of training')
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='Size of a batch')
    parser.add_argument('--seed', default=SEED, type=int, help='Set seed for reproducibility')

    parser.add_argument('--lr_initial', default=LR_INITIAL, type=float, help='Initial learning rate')
    parser.add_argument('--reduce_lr_on_plateau', default=REDUCE_LR_ACTIVATED, type=bool,
                        help='Use the reduce_lr_on_plateau callback from tensorflow')
    parser.add_argument('--min_delta', default=MIN_DELTA, type=float, help='Minimum delta to use with reduce_lr_on_plateau')
    parser.add_argument('--lr_minimal', default=LR_MINIMAL, type=float, help='Minimal learning rate scheduler can decrease lr to')
    parser.add_argument('--lr_patience', default=LR_PATIENCE, type=int, help='How many epochs to wait before decreasing learning rate if validation loss is not decreasing')
    parser.add_argument('--lr_decrease_factor', default=LR_DECREASE_FACTOR, type=float, help='By which factor to decrease learning rate in a single step of a scheduler ')


    parser.add_argument('--shift_labels', default=SHIFT_LABELS, type=int, help='How much to shift labels/targets with respect to features while reading data file for training')

    parser.add_argument('--path_to_models', default=PATH_TO_MODELS, type=str,
                        help='Path where to save/ from where to load models')

    parser.add_argument('--path_to_normalization_info', default=PATH_TO_NORMALIZATION_INFO, type=str,
                        help='Path where the cartpole data is saved')

    parser.add_argument('--on_fly_data_generation', default=ON_FLY_DATA_GENERATION, type=bool,
                        help='Generate data for training during training, instead of loading previously saved data')
    parser.add_argument('--normalize', default=NORMALIZE, type=bool, help='Make all data between 0 and 1')
    parser.add_argument('--use_nni', default=USE_NNI, type=bool, help='Use NNI package to search hyperparameter space')
    parser.add_argument('--construct_network', default=CONSTRUCT_NETWORK, type=str,
                        help='For Pytorch you can decide if you want to construct network with modules or cells.'
                             'First is needed for DeltaRNN, second gives more flexibility in specifying layers sizes.')

    parser.add_argument('--validate_also_on_training_set', default=VALIDATE_ALSO_ON_TRAINING_SET, type=bool,
                        help='If you want to validate also on training dataset, after finished training epoch. Can make training considerably longer!')


    args = parser.parse_args()

    args.config_path = config_path

    # Make sure that the provided lists of inputs and outputs are in alphabetical order

    if args.post_wash_out_len < 1:
        raise ValueError('post_wash_out_len, the part relevant for loss calculation must be at least 1, also for dense network')

    if args.control_inputs is not None:
        args.control_inputs = sorted(args.control_inputs)

    args.control_inputs = generate_column_names(args.control_inputs)

    if args.state_inputs is not None:
        args.state_inputs = sorted(args.state_inputs)

    args.state_inputs = generate_column_names(args.state_inputs)

    if args.setpoint_inputs is not None:
        args.setpoint_inputs = sorted(args.setpoint_inputs)

    args.inputs = args.control_inputs + args.state_inputs + args.setpoint_inputs

    if args.outputs is not None:
        args.outputs = sorted(args.outputs)

    args.plot_weights_distribution = PLOT_WEIGHTS_DISTRIBUTION

    args.loss_mode = config['training_default']['loss_mode']

    args.regularization = REGULARIZATION

    args.quantization = QUANTIZATION

    args.pruning_activated = PRUNING_ACTIVATED
    args.pruning_schedule = PRUNING_SCHEDULE
    args.pruning_schedules = PRUNING_SCHEDULES

    args.filters = FILTERS

    args.config_series_modification = CONFIG_SERIES_MODIFICATION

    return args




def generate_column_names(inputs):
    """
    Generate a list of column names based on input patterns with zero-padded indices.

    Parameters:
    - inputs (list of str): List containing column names or patterns like 'prefix(n)'.

    Returns:
    - selected_columns (list of str): List of generated column names.
    """
    selected_columns = []

    for item in inputs:
        if '(' in item and ')' in item:
            try:
                # Extract prefix and number
                prefix, num_str = item.split('(')
                num = int(num_str.rstrip(')'))
                prefix = prefix.strip()

                # Determine the padding width based on the number of digits in 'num'
                padding_width = len(str(num-1))

                # Generate zero-padded column names from 0 to 'num-1'
                for i in range(num):
                    padded_index = str(i).zfill(padding_width)
                    column_name = f"{prefix}_{padded_index}"
                    selected_columns.append(column_name)
            except ValueError:
                print(f"Warning: Invalid format for input '{item}'. Skipping.")
        else:
            # Direct column name
            selected_columns.append(item)

    return selected_columns
