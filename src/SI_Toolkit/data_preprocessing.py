"""
Various functions to preprocess data:
- duplicating column in a dataset but with shifted rows corresponding to a time shift
- quantizing sensor data
- adding derivatives to the selected features of dataset
"""

import csv
import os

import numpy as np
import pandas as pd
from derivative import dxdt
from tqdm import trange

from SI_Toolkit.Functions.set_working_directory import set_working_directory

from SI_Toolkit.load_and_normalize import get_paths_to_datafiles, load_data


def transform_dataset(get_files_from, save_files_to, transformation='add_shifted_columns', **kwargs):

    set_working_directory()

    paths_to_recordings = get_paths_to_datafiles(get_files_from)

    if not paths_to_recordings:
        Exception('No files found')

    try:
        os.makedirs(save_files_to)
    except FileExistsError:
        pass

    for i in trange(len(paths_to_recordings)):
        current_path = paths_to_recordings[i]
        df = load_data(list_of_paths_to_datafiles=[current_path], verbose=False)[0]

        processed_file_name = os.path.basename(current_path)

        if transformation == 'add_shifted_columns':
            df_processed = add_shifted_columns_single_file(df, variables_to_shift=kwargs['variables_to_shift'], indices_by_which_to_shift=kwargs['indices_by_which_to_shift'])
        elif transformation == 'apply_sensor_quantization':
            df_processed = apply_sensors_quantization(df, variables_quantization_dict=kwargs['variables_quantization_dict'])
        elif transformation == 'add_derivatives':
            df_processed = append_derivatives(df, variables_for_derivative=kwargs['variables_for_derivative'], derivative_algorithm=kwargs['derivative_algorithm'], df_name=processed_file_name)
        else:
            raise NotImplemented('Transformation {} is not implemented'.format(transformation))

        if df_processed is None:
            print('Dropping {}, transformation not successful. '.format(current_path))
            if transformation == 'add_derivatives':
                print('Probably too short to calculate derivatives.')
            continue

        processed_file_path = os.path.join(save_files_to, processed_file_name)
        with open(processed_file_path, 'w', newline=''):  # Overwrites if existed
            pass
        with open(current_path, "r", newline='') as f_input, \
                open(processed_file_path, "a", newline='') as f_output:
            for line in f_input:
                if line[0:len('#')] == '#':
                    csv.writer(f_output).writerow([line.strip()])
                else:
                    break

        df_processed.to_csv(processed_file_path, index=False, mode='a')


def append_derivatives_to_df(df, variables_for_derivative, derivative_algorithm, cut=1):

    df = df.reset_index(drop=True)

    try:
        t = df['time'].values
    except KeyError:
        t = np.arange(df.shape[0])
    y = df[variables_for_derivative].values
    dy = np.zeros_like(y)
    for j in range(len(variables_for_derivative)):
        if derivative_algorithm == 'finite_difference':
            dy[:, j] = dxdt(y[:, j], t, kind=derivative_algorithm, k=cut)
        elif derivative_algorithm == 'single_difference':
            cut = 1
            dy[:-1, j] = (y[1:, j]-y[:-1, j])/(t[1:]-t[:-1])
        elif derivative_algorithm == 'backward_difference':
            cut = 1
            dy[1:, j] = (y[1:, j]-y[:-1, j])/(t[1:]-t[:-1])
        else:
            raise NotImplemented('{} is not a recognised name for derivative algorithm'.format(derivative_algorithm))

    derivatives_names = ['D_' + x for x in variables_for_derivative]
    derivatives = pd.DataFrame(data=dy, columns=derivatives_names)

    df = pd.concat([df, derivatives], axis=1)

    # cut first and last k where derivative is not well determined
    df = df.iloc[cut:-cut, :].reset_index(drop=True)

    return df


def append_derivatives(df, variables_for_derivative, derivative_algorithm, df_name, verbose=False):
    """
    Takes list of dataframes dfs
    and augment it with derivatives of columns indicated in variables_for_derivative
    using algorithm indicated in derivative_algorithm.
    The output file is shorter - first and last indices for which it is difficult to get good derivative are dropped.
    """

    cut = 1

    paths_with_derivatives = []

    dfs_split = []
    if 'experiment_index' in df.columns:
        grouped = df.groupby(df.experiment_index)
        for i in df.experiment_index.unique():
            dfs_split.append(grouped.get_group(i))
    else:
        dfs_split.append(df)

    dfs_processed = []
    for df_partial in dfs_split:
        # print(file_names)
        if df_partial.shape[0] < 2 * cut:
            continue
        df_partial = append_derivatives_to_df(df_partial, variables_for_derivative, derivative_algorithm, cut)
        dfs_processed.append(df_partial)

    if dfs_processed:
        paths_with_derivatives.append(df_name)
    else:
        print('Dropping {} as too short'.format(df_name))

    if dfs_processed:
        dfs_processed = pd.concat(dfs_processed, axis=0).reset_index(drop=True)
    else:
        dfs_processed = None


    # The remaining part of the function keeps track of names of all the processed files and checks if all have time axis
    if not hasattr(append_derivatives, 'no_time_axis_in_files'):
        # At the first call of the function, create the lists
        append_derivatives.all_processed_files = []
        append_derivatives.no_time_axis_in_files = []
    append_derivatives.all_processed_files.append(df_name)  # Keep track of all processed files
    if 'time' not in df:
        append_derivatives.no_time_axis_in_files.append(df_name)  # Keep track of files without time axis

    if append_derivatives.no_time_axis_in_files:  # True if the list of files without time axis is not empty
        if set(append_derivatives.no_time_axis_in_files) != set(append_derivatives.all_processed_files):
            print('\033[91mWARNING!!!! \n Some data files have time axes, but some not. \n'
                  'The derivative calculation across files is inconsistent!\033[0m')
            print('The files without time axis are:')
            for filename in append_derivatives.no_time_axis_in_files:
                print(filename)
        elif verbose and set(append_derivatives.no_time_axis_in_files) == set(append_derivatives.all_processed_files):
            print('No time axis provided. Calculated increments dx for all the files.')

    return dfs_processed


def add_shifted_columns_single_file(df, variables_to_shift, indices_by_which_to_shift):
    length_original_df = len(df)
    for j in range(len(indices_by_which_to_shift)):
        index_by_which_to_shift = int(indices_by_which_to_shift[j])
        if index_by_which_to_shift == 0:
            continue
        new_names = [variable_to_shift + '_' + str(index_by_which_to_shift) for variable_to_shift in variables_to_shift]
        subset = df.loc[:, variables_to_shift]
        if index_by_which_to_shift > 0:
            subset.index += -index_by_which_to_shift

        else:
            subset.index += abs(index_by_which_to_shift)

        subset.columns = new_names
        df = pd.concat((df, subset), axis=1)

    bound_low = min(indices_by_which_to_shift)
    if bound_low >= 0:
        bound_low = 0
    else:
        bound_low = abs(bound_low)

    bound_high = max(indices_by_which_to_shift)
    if bound_high <= 0:
        bound_high = length_original_df
    else:
        bound_high = length_original_df - bound_high - 1  # indexing from 0!

    df_processed = df.loc[bound_low:bound_high, :]

    return df_processed


def apply_sensors_quantization(df, variables_quantization_dict):
    for variable, precision in variables_quantization_dict.items():
        df[variable] = (df[variable] / precision).round() * precision

    return df
