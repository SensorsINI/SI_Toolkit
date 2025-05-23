"""
Various functions to preprocess data:
- duplicating column in a dataset but with shifted rows corresponding to a time shift
- quantizing sensor data
- adding derivatives to the selected features of dataset
"""

import os
import numpy as np
import pandas as pd
from derivative import dxdt
from tqdm import trange

import sys
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

from SI_Toolkit.load_and_normalize import get_paths_to_datafiles, load_data

try:
    from SI_Toolkit_ASF.ToolkitCustomization.preprocess_data import *
except ImportError:
    pass

from SI_Toolkit.General.preprocess_data_add_control_along_trajectories import add_control_along_trajectories


def transform_dataset(get_files_from, save_files_to, transformation='add_shifted_columns', **kwargs):

    if os.path.exists(get_files_from):
        # If the path is a file
        if os.path.isfile(get_files_from):
            paths_to_recordings = [get_files_from]
            get_files_from = os.path.dirname(get_files_from)
        # If the path is a directory
        elif os.path.isdir(get_files_from):
            paths_to_recordings = get_paths_to_datafiles(get_files_from)
        else:
            # Path exists but is neither a file nor a directory (rare cases)
            raise ValueError('Path exists but is neither a file nor a directory')
    else:
        # Path does not exist
        raise FileNotFoundError(f"Path {get_files_from} does not exist")

    if not paths_to_recordings:
        raise Exception('No files found')

    if save_files_to is not None:
        try:
            os.makedirs(save_files_to)
        except FileExistsError:
            pass

    for i in trange(len(paths_to_recordings), leave=True, position=0, desc='Processed datafiles'):
        current_path = paths_to_recordings[i]
        relative_path_to_search_root = os.path.relpath(current_path, get_files_from)
        processed_file_name = os.path.basename(current_path)

        df = load_data(list_of_paths_to_datafiles=[current_path], verbose=False)[0]

        # Apply transformation
        if callable(transformation):
            # If `transformation` is a function, call it directly
            df_processed = transformation(df, **kwargs)
        else:
            # If `transformation` is a string, retrieve the function and call it
            # Available transformations: add_control_along_trajectories, append_derivatives, apply_sensors_quantization, add_shifted_columns
            # Plus the application specific transformations
            try:
                # Retrieve the function by name and call it with the appropriate arguments
                transformation_function = globals()[transformation]
            except KeyError:
                raise NotImplementedError(f'Transformation {transformation} is not implemented')
            df_processed = transformation_function(df, **kwargs)

        if df_processed is None:
            print('Dropping {}, transformation not successful.'.format(current_path))
            if transformation == 'append_derivatives':
                print('Probably too short to calculate derivatives.')
            continue

        if save_files_to:
            processed_file_path = os.path.join(save_files_to, relative_path_to_search_root)
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)

            # Read and store comments first
            comments = []
            with open(current_path, "r", newline='') as f_input:
                for line in f_input:
                    if line.lstrip().startswith('#'):
                        comments.append(line)
                    else:
                        break

            # Open the file for writing and write comments
            with open(processed_file_path, 'w', newline='') as f_output:
                f_output.write(f'# Original file transformed with "{transformation}" transformation\n#\n')
                f_output.writelines(comments)
            df_processed.to_csv(processed_file_path, index=False, mode='a')


def decimate_datasets(df, keep_every_nth_row, **kwargs):
    """
    Decimates the dataset by keeping only every keep_every_nth_row row.
    """
    df_processed = df.iloc[::keep_every_nth_row, :]

    return df_processed


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


def append_derivatives(df, variables_for_derivative, derivative_algorithm, df_name, verbose=False, **kwargs):
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


def add_shifted_columns(df, variables_to_shift, indices_by_which_to_shift, **kwargs):
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

    max_shift = max(abs(shift) for shift in indices_by_which_to_shift)
    df_processed = df.iloc[max_shift:-max_shift]

    return df_processed


def apply_sensors_quantization(df, variables_quantization_dict):
    for variable, precision in variables_quantization_dict.items():
        df[variable] = (df[variable] / precision).round() * precision

    return df