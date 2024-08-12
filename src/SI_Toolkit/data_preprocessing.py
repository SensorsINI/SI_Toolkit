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

import sys
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

from SI_Toolkit.load_and_normalize import get_paths_to_datafiles, load_data

try:
    from SI_Toolkit_ASF.ToolkitCustomization.preprocess_data import *
except ImportError:
    pass

def transform_dataset(get_files_from, save_files_to, transformation='add_shifted_columns', **kwargs):

    if os.path.exists(get_files_from):
        # If the path is a file
        if os.path.isfile(get_files_from):
            paths_to_recordings  = [get_files_from]
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
        Exception('No files found')

    try:
        os.makedirs(save_files_to)
    except FileExistsError:
        pass

    for i in trange(len(paths_to_recordings), leave=True, position=0, desc='Processed datafiles'):
        current_path = paths_to_recordings[i]
        relative_path_to_search_root = os.path.relpath(current_path, get_files_from)
        processed_file_name = os.path.basename(current_path)

        df = load_data(list_of_paths_to_datafiles=[current_path], verbose=False)[0]

        # Available transformations: add_control_along_trajectories, append_derivatives, apply_sensors_quantization, add_shifted_columns
        # Plus the application specific transformations
        if transformation == 'append_derivatives':
            df_processed = append_derivatives(df, df_name=processed_file_name, **kwargs)
        else:
            try:
                # Retrieve the function by name and call it with the appropriate arguments
                transformation_function = globals()[transformation]
            except KeyError:
                raise NotImplementedError(f'Transformation {transformation} is not implemented')
            df_processed = transformation_function(df, **kwargs)

        if df_processed is None:
            print('Dropping {}, transformation not successful. '.format(current_path))
            if transformation == 'append_derivatives':
                print('Probably too short to calculate derivatives.')
            continue

        processed_file_path = os.path.join(save_files_to, relative_path_to_search_root)
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)

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


from Control_Toolkit.others.globals_and_utils import get_controller_name, get_optimizer_name, import_controller_by_name
from Control_Toolkit.Controllers import template_controller


def add_control_along_trajectories(df, controller, controller_output_variable_name='Q_calculated', **kwargs):
    """
    Adds controller to the trajectory data.
    :param df: trajectory data
    :param controller: controller to evaluate on data
    :param controller_output_variable_name: name of the column with controller output
    :return: trajectory data with controller
    """

    controller_name = controller['controller_name']
    optimizer_name = controller.get('optimizer_name', None)

    environment_name = controller['environment_name']
    action_space = controller['action_space']
    state_components = controller['state_components']
    environment_attributes_dict = controller['environment_attributes_dict']

    initial_environment_attributes = {key: df[value].iloc[0] for key, value in environment_attributes_dict.items()}

    controller_name, _ = get_controller_name(
        controller_name=controller_name
    )

    Controller: "type[template_controller]" = import_controller_by_name(controller_name)
    controller = Controller(
        environment_name=environment_name,
        initial_environment_attributes=initial_environment_attributes,
        control_limits=(action_space.low, action_space.high),
    )
    # Final configuration of controller
    if controller.has_optimizer:
        controller.configure(optimizer_name)
        optimizer_name, _ = get_optimizer_name(
            optimizer_name=controller.optimizer.optimizer_name
        )

    else:
        controller.configure()

    Q_calculated_list = []

    s = np.array(df[state_components])
    time = np.array(df['time'])
    environment_attributes_array = np.array(df[environment_attributes_dict.values()])


    for i in trange(len(df), leave=False, position=1, desc='Processing current datafile'):
        environment_attributes = {key: environment_attributes_array[i, idx] for idx, key in enumerate(environment_attributes_dict.keys())}
        Q_calculated = float(controller.step(
            s=s[i],
            time=time[i],
            updated_attributes=environment_attributes,
        ))
        Q_calculated_list.append(Q_calculated)

    df[controller_output_variable_name] = Q_calculated_list

    return df
