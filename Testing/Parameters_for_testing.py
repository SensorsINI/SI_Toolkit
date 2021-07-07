# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:29:29 2020

@author: Marcin
"""

import argparse
import numpy as np

import yaml, os
config = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config.yml'), 'r'), Loader=yaml.FullLoader)

PATH_TO_MODELS = config["paths"]["PATH_TO_EXPERIMENT_RECORDINGS"] + config['paths']['path_to_experiment'] + "Models/"

tests = config['testing']['tests']

PATH_TO_NORMALIZATION_INFO = config["paths"]["PATH_TO_EXPERIMENT_RECORDINGS"] + config['paths']['path_to_experiment'] + "NormalizationInfo/"
PATH_TO_NORMALIZATION_INFO += os.listdir(PATH_TO_NORMALIZATION_INFO)[0]
norm_infos = [PATH_TO_NORMALIZATION_INFO]*len(tests) # Norm info for each test, for Euler has no effect, can be None or whatever

dt_euler = [0.002]*len(tests)  # Timestep of Euler (printed are only values, for which ground truth value exists), for neural network has no effect
titles = tests  # Titles of tests to be printed in GUI

TEST_FILE = [config["paths"]["PATH_TO_EXPERIMENT_RECORDINGS"] + config['paths']['path_to_experiment'] + "Recordings/Test/" + config['testing']['TEST_FILE']]
# TODO: For consistency features should be "state inputs" probably. Think about it once more before implementing
# For CartPole
features = list(np.sort(
    ['angle',
     'angleD',
     'angle_cos',
     'angle_sin',
     'position',
     'positionD',
     ]
))
features_units = [' (deg)', ' (deg/s)', '', '', ' (m)', ' (m/s)']

# For l2race
# features = list(['x1','x2','x3','x4','x5','x6','x7'])

# For CartPole
control_inputs = ['Q']

# For l2race
# control_inputs = ['u1', 'u2']

# TEST_FILE = ['./Experiment_Recordings/PCP-1/Test/Dance-Test-cartpole-2021-05-26-17-17-13.csv']

# PATH_TO_NORMALIZATION_INFO = config['modeling']['PATH_TO_NORMALIZATION_INFO']


def args():
    parser = argparse.ArgumentParser(description='Parameters for Brunton GUI', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Only valid for graphical testing:
    parser.add_argument('--test_file', default=TEST_FILE, type=str,
                        help='File name of the recording to be used for validating the RNN'
                             'e.g. oval_easy_test.csv ')
    parser.add_argument('--tests', default=tests,
                        help='List of tests which should be performed')
    parser.add_argument('--norm_infos', default=norm_infos,
                        help='List of norm_infos for neural nets')
    parser.add_argument('--dt_euler', default=dt_euler,
                        help='List of timestep lengths for Euler experiments')
    parser.add_argument('--features', default=features,
                        help='List of features (= state_inputs) which can be plotted in GUI')
    parser.add_argument('--control_inputs', default=control_inputs,
                        help='List of control inputs')
    parser.add_argument('--titles', default=titles,
                        help='List of titles of tests.')

    parser.add_argument('--path_to_models', default=PATH_TO_MODELS, type=str,
                        help='Path to the NN trained models ')
    parser.add_argument('--test_len', default="max",
                        help='For graphical testing only test_len samples from first test file is taken.')
    parser.add_argument('--test_start_idx', default=0, type=int, help='Indicates from which point data from test file should be taken.')
    parser.add_argument('--test_max_horizon', default=40, type=int,
                        help='Indicates prediction horizon for testing.')

    args = parser.parse_args()

    return args