# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:29:29 2020

@author: Marcin
"""

import argparse
import numpy as np

import yaml
import os

config_testing = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_testing.yml'), 'r'), Loader=yaml.FullLoader)

PATH_TO_NN = config_testing['testing']["PATH_TO_NN"]

tests = config_testing['testing']['tests']

titles = tests  # Titles of tests to be printed in GUI

TEST_FILE = config_testing['testing']['TEST_FILE']
MAX_HORIZON = config_testing['testing']['MAX_HORIZON']
START_IDX = config_testing['testing']['START_IDX']

default_locations_for_testfile = [config_testing['testing']['PATH_TO_TEST_FILE']]

# TODO: For consistency features should be "state inputs" probably. Think about it once more before implementing
# For CartPole
features = list(np.sort(config_testing['testing']['features']))

# For l2race
# features = list(['x1','x2','x3','x4','x5','x6','x7'])

# For CartPole
control_inputs = config_testing['testing']['control_inputs']

# For l2race
# control_inputs = ['u1', 'u2']

TEST_LEN = config_testing['testing']['TEST_LEN']

DECIMATION = config_testing['testing']['DECIMATION']

def args():
    parser = argparse.ArgumentParser(description='Parameters for Brunton GUI', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Only valid for graphical testing:
    parser.add_argument('--test_file', default=TEST_FILE, type=str,
                        help='File name of the recording to be used for validating the RNN'
                             'e.g. oval_easy_test.csv ')
    parser.add_argument('--default_locations_for_testfile', default=default_locations_for_testfile, type=str,
                        help='Where to search for test file if only name and not the path specified'
                             'e.g. oval_easy_test.csv ')
    parser.add_argument('--tests', default=tests,
                        help='List of tests which should be performed')
    parser.add_argument('--features', default=features,
                        help='List of features (= state_inputs) which can be plotted in GUI')
    parser.add_argument('--control_inputs', default=control_inputs,
                        help='List of control inputs')
    parser.add_argument('--titles', default=titles,
                        help='List of titles of tests.')

    parser.add_argument('--path_to_models', default=PATH_TO_NN, type=str,
                        help='Path to the NN trained models ')
    parser.add_argument('--test_len', default=TEST_LEN,
                        help='For graphical testing only test_len samples from first test file is taken.')
    parser.add_argument('--test_start_idx', default=START_IDX, type=int, help='Indicates from which point data from test file should be taken.')
    parser.add_argument('--test_max_horizon', default=MAX_HORIZON, type=int,
                        help='Indicates prediction horizon for testing.')
    parser.add_argument('--decimation', default=DECIMATION, type=int,
                        help='How much to decimate the dataset - make it to fit the network timestep')

    args = parser.parse_args()

    return args