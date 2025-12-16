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
from SI_Toolkit.General.preprocess_data_back_to_front_trajectories import back_to_front_trajectories


def _regenerate_time_column(df, dt=None):
    """
    Regenerate the 'time' column to ensure consistent dt after row removal.
    
    This is critical for Brunton test which computes dt from time differences.
    After add_shifted_columns removes rows, the time column has gaps that cause
    incorrect dt calculation.
    
    The time column is regenerated as simple sequential time (0, dt, 2*dt, ...)
    regardless of experiment_index. This ensures consistent dt throughout.
    
    Args:
        df: DataFrame with 'time' column to regenerate
        dt: Time step. If None, inferred from existing time differences (using median).
    
    Returns:
        DataFrame with regenerated 'time' column
    """
    if 'time' not in df.columns:
        return df
    
    df = df.copy()
    
    # Try to infer dt from existing time differences WITHIN trajectories
    # (excluding jumps at experiment_index boundaries)
    if dt is None:
        time_diffs = df['time'].diff()
        
        # Create mask for within-trajectory rows (experiment_index doesn't change)
        if 'experiment_index' in df.columns:
            same_trajectory = df['experiment_index'].diff().fillna(0) == 0
            within_traj_diffs = time_diffs[same_trajectory].iloc[1:]
        else:
            within_traj_diffs = time_diffs.iloc[1:]
        
        # Use median of positive within-trajectory differences
        positive_diffs = within_traj_diffs[within_traj_diffs > 0]
        if len(positive_diffs) > 0:
            dt = positive_diffs.median()
        
        if dt is None or dt <= 0:
            raise ValueError(
                "Could not infer dt from time column. "
                "The time column has no valid positive differences within trajectories. "
                "Please provide dt explicitly or ensure the time column has proper values."
            )
    
    # Regenerate time column as simple sequential time
    # This ensures consistent dt throughout, regardless of experiment_index
    df['time'] = np.arange(len(df)) * dt
    
    return df


def split_by_column(df, column_name):
    """
    Split a DataFrame by a specified column if it exists.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        column_name (str): Name of the column to split by
    
    Returns:
        list of pd.DataFrame: List of DataFrames split by the specified column,
                              or [df] if the column doesn't exist
    """
    if column_name and column_name in df.columns:
        dfs_split = []
        grouped = df.groupby(column_name, sort=False)
        for idx in df[column_name].unique():
            dfs_split.append(grouped.get_group(idx))
        return dfs_split
    else:
        return [df]


def transform_dataset(get_files_from, save_files_to, transformation='add_shifted_columns', split_on_column=None, **kwargs):

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

        # Split by specified column if provided and present
        dfs_split = split_by_column(df, split_on_column)
        
        # Apply transformation to each split
        dfs_processed = []
        for df_partial in dfs_split:
            if callable(transformation):
                # If `transformation` is a function, call it directly
                df_transformed = transformation(df_partial, **kwargs)
            else:
                # If `transformation` is a string, retrieve the function and call it
                # Available transformations: back_to_front_trajectories, add_control_along_trajectories, append_derivatives, apply_sensors_quantization, add_shifted_columns
                # Plus the application specific transformations
                try:
                    # Retrieve the function by name and call it with the appropriate arguments
                    transformation_function = globals()[transformation]
                except KeyError:
                    raise NotImplementedError(f'Transformation {transformation} is not implemented')
                df_transformed = transformation_function(df_partial, df_name=processed_file_name, current_path=current_path, **kwargs)
            
            if df_transformed is not None:
                dfs_processed.append(df_transformed)
        
        # Concatenate results
        if dfs_processed:
            df_processed = pd.concat(dfs_processed, axis=0, ignore_index=True)
        else:
            df_processed = None

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
                
                # Add transformation-specific metadata from DataFrame attrs
                if hasattr(df_processed, 'attrs') and df_processed.attrs:
                    f_output.write('#\n# Transformation metadata:\n')
                    for key, value in df_processed.attrs.items():
                        f_output.write(f'# {key}: {value}\n')
                    f_output.write('#\n')
                    
            df_processed.to_csv(processed_file_path, index=False, mode='a')


def decimate_datasets(df, keep_every_nth_row, **kwargs):
    """
    Decimates the dataset by keeping only every keep_every_nth_row row.
    """
    df_processed = df.iloc[::keep_every_nth_row, :]

    return df_processed


def keep_section(df, section_to_keep, mode='percent', **kwargs):
    """
    Extract a section of the DataFrame.

    Parameters:
    - df: input DataFrame
    - section_to_keep: tuple of (start, end)
        * If mode='percent': values must be between 0 and 1
        * If mode='lines': start and end are row indices; use -1 as end to indicate "until end"
    - mode: 'percent' or 'lines'
    """
    n = len(df)

    if mode == 'percent':
        if not (0 <= section_to_keep[0] < section_to_keep[1] <= 1):
            raise ValueError("For 'percent' mode, section_to_keep must be within [0, 1] and in increasing order")
        start_idx = int(n * section_to_keep[0])
        end_idx = int(n * section_to_keep[1])

    elif mode == 'lines':
        start_idx, end_idx = section_to_keep
        if end_idx == -1:
            end_idx = n
        if not (0 <= start_idx < end_idx <= n):
            raise ValueError(f"For 'lines' mode, indices must be within [0, {n}] and in increasing order (use -1 for end)")
    else:
        raise ValueError("mode must be either 'percent' or 'lines'")

    return df.iloc[start_idx:end_idx].reset_index(drop=True)


def minimum_filter(df, window, features, thresholds, **kwargs):
    """
    Replace values whose absolute value exceeds the threshold with the value
    (from a centered rolling window) that has the smallest absolute value.

    Uses built-in rolling + apply for efficiency.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        window (int): Size of the centered rolling window.
        features (list): List of feature names to filter.
        thresholds (list): List of corresponding thresholds.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df_processed = df.copy()

    def min_abs_value(x):
        return x[np.argmin(np.abs(x))]

    for feature, threshold in zip(features, thresholds):
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame.")

        # Apply custom rolling function
        rolling_min_abs = df[feature].rolling(window=window, center=True, min_periods=1).apply(min_abs_value, raw=True)

        # Replace only where |value| > threshold
        condition = df[feature].abs() > threshold
        df_processed[feature] = np.where(condition, rolling_min_abs, df[feature])

    return df_processed


def time_reverse(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Return a copy of df with rows in reversed order."""
    return df.iloc[::-1].reset_index(drop=True)


def append_derivatives_to_df(df, variables_for_derivative, derivative_algorithm, cut=1, compute_differences=False):
    """
    Append derivative columns (D_*) to a dataframe.
    
    Args:
        df: Input dataframe
        variables_for_derivative: List of column names to compute derivatives for
        derivative_algorithm: Algorithm to use ('finite_difference', 'single_difference', 'backward_difference')
        cut: Number of rows to cut from start/end (default: 1)
        compute_differences: If True, compute increments (Δx) instead of rates (dx/dt).
                           When False (default), requires 'time' column and computes rates.
    
    Returns:
        DataFrame with D_* columns appended
    """
    df = df.reset_index(drop=True)

    if compute_differences:
        # Explicit mode: compute increments (Δx) with dt=1
        t = np.arange(df.shape[0])
    elif 'time' not in df.columns:
        raise KeyError(
            "'time' column required for rate calculation (dx/dt). "
            "Use compute_differences=True if you want increments (Δx) instead."
        )
    else:
        t = df['time'].values
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


def append_derivatives(df, variables_for_derivative, derivative_algorithm, df_name, verbose=False, compute_differences=False, **kwargs):
    """
    Takes list of dataframes dfs
    and augment it with derivatives of columns indicated in variables_for_derivative
    using algorithm indicated in derivative_algorithm.
    The output file is shorter - first and last indices for which it is difficult to get good derivative are dropped.
    
    Args:
        df: Input dataframe
        variables_for_derivative: List of column names to compute derivatives for
        derivative_algorithm: Algorithm to use ('finite_difference', 'single_difference', 'backward_difference')
        df_name: Name of the file being processed (for logging)
        verbose: If True, print warnings about missing time axis
        compute_differences: If True, compute increments (Δx) instead of rates (dx/dt).
                           When False (default), requires 'time' column and computes rates.
        **kwargs: Additional arguments (ignored)
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
        df_partial = append_derivatives_to_df(df_partial, variables_for_derivative, derivative_algorithm, cut, compute_differences=compute_differences)
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

    # Regenerate time column after derivative computation removed edge rows
    # This ensures proper dt calculation in downstream tests
    if 'time' in df.columns:
        dfs_processed = _regenerate_time_column(dfs_processed)

    return dfs_processed


def add_shifted_columns(df, variables_to_shift, indices_by_which_to_shift, **kwargs):

    orig_index = df.index  # keep original row set & order for final trimming

    for j in range(len(indices_by_which_to_shift)):
        index_by_which_to_shift = int(indices_by_which_to_shift[j])
        if index_by_which_to_shift == 0:
            continue
        new_names = [variable_to_shift + '_' + str(index_by_which_to_shift) for variable_to_shift in variables_to_shift]
        subset = df.loc[:, variables_to_shift].copy()  # copy to avoid chained-index pitfalls

        # Reindex the subset so values align to shifted rows on concat.
        if index_by_which_to_shift > 0:
            subset.index += -index_by_which_to_shift

        else:
            subset.index += abs(index_by_which_to_shift)

        subset.columns = new_names
        df = pd.concat((df, subset), axis=1)

    # --- Correct, minimal trimming logic ---
    # Head rows are incomplete if there are NEGATIVE shifts (values pushed forward),
    # Tail rows are incomplete if there are POSITIVE shifts (values pulled from the future).
    pos_max = max((int(s) for s in indices_by_which_to_shift if int(s) > 0), default=0)
    neg_max = max((abs(int(s)) for s in indices_by_which_to_shift if int(s) < 0), default=0)

    # First, restrict back to the original index to drop any extra rows created by shifting
    df = df.reindex(orig_index)

    # If there is nothing to trim (all shifts are 0), return as-is
    if pos_max == 0 and neg_max == 0:
        return df

    # Trim asymmetrically: drop 'neg_max' from the head, 'pos_max' from the tail.
    start = neg_max
    end = len(df) - pos_max
    df_processed = df.iloc[start:end].copy()
    
    # Regenerate time column to ensure proper dt calculation after row removal
    # This is critical for Brunton test which computes dt from time differences
    if 'time' in df_processed.columns:
        df_processed = _regenerate_time_column(df_processed)

    return df_processed


def flip_column_signs(df, variables_to_flip, **kwargs):
    df_processed = df.copy()

    for var in variables_to_flip:
        if var in df_processed.columns:
            df_processed[var] = -df_processed[var]
        else:
            raise ValueError(f"Column '{var}' does not exist in dataframe.")

    return df_processed


def subtract_columns(df, variables_to_subtract, **kwargs):
    """
    Appends new columns to the dataframe by subtracting two existing columns.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        variables_to_subtract (list of lists): Each sublist has three elements [column_a, column_b, column_c],
            where column_c will be calculated as column_a - column_b and appended to df.
        kwargs: Additional parameters for future extensions.

    Returns:
        pd.DataFrame: Processed dataframe with new columns added.
    """
    df_processed = df.copy()

    for var_a, var_b, var_c in variables_to_subtract:
        # Verify columns exist in DataFrame to avoid errors
        if var_a not in df_processed.columns or var_b not in df_processed.columns:
            raise ValueError(f"One or both columns '{var_a}' or '{var_b}' do not exist in dataframe.")

        # Perform the subtraction and create the new column
        df_processed[var_c] = df_processed[var_a] - df_processed[var_b]

    return df_processed


def apply_sensors_quantization(df, variables_quantization_dict):
    for variable, precision in variables_quantization_dict.items():
        df[variable] = (df[variable] / precision).round() * precision

    return df