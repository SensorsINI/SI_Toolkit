# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:29:29 2020

@author: Marcin
"""

import argparse
from CartPole.state_utilities import STATE_VARIABLES_REDUCED, STATE_VARIABLES

import yaml, os
config = yaml.load(open(os.path.join('SI_Toolkit', 'config.yml'), 'r'), Loader=yaml.FullLoader)

PATH_TO_MODELS = config['modeling']['PATH_TO_MODELS']

features = list(STATE_VARIABLES_REDUCED)

tests = config['testing']['tests']
norm_infos = [config['modeling']['PATH_TO_NORMALIZATION_INFO']]*len(tests) # Norm info for each test, for Euler has no effect, can be None or whatever
dt_euler = [0.002]*len(tests)  # Timestep of Euler (printed are only values, for which ground truth value exists), for neural network has no effect
titles = tests  # Titles of tests to be printed in GUI

TEST_FILE = config['testing']['TEST_FILE']

PATH_TO_NORMALIZATION_INFO = config['modeling']['PATH_TO_NORMALIZATION_INFO']


def args():
    parser = argparse.ArgumentParser(description='Parameters for Brunton GUI')
    # Only valid for graphical testing:
    parser.add_argument('--test_file', default=TEST_FILE, type=str,
                        help='File name of the recording to be used for validating the RNN'
                             'e.g. oval_easy_test.csv ')
    parser.add_argument('--tests', default=tests,
                        help='List of tests whcih should be performed')
    parser.add_argument('--norm_infos', default=norm_infos,
                        help='List of norm_infos for neural nets')
    parser.add_argument('--dt_euler', default=dt_euler,
                        help='List of timestep lengths for Euler experiments')
    parser.add_argument('--features', default=features,
                        help='List of features which can be plotted in GUI')
    parser.add_argument('--titles', default=titles,
                        help='List of titles of tests.')

    parser.add_argument('--path_to_models', default=PATH_TO_MODELS, type=str,
                        help='Path to the NN trained models ')
    parser.add_argument('--test_len', default=500, type=int,
                        help='For graphical testing only test_len samples from first test file is taken.')
    parser.add_argument('--test_start_idx', default=100, type=int, help='Indicates from which point data from test file should be taken.')
    parser.add_argument('--test_max_horizon', default=40, type=int,
                        help='Indicates prediction horizon for testing.')

    args = parser.parse_args()

    return args