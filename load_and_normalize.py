import pandas as pd
import numpy as np
from tqdm import trange
from time import sleep

import glob
import os

import matplotlib.pyplot as plt

# Import module to get a current time and date used to name the files containing normalization information
from datetime import datetime
import csv

try:
    # Use gitpython to get a current revision number and use it in description of experimental data
    from git import Repo
except:
    pass

try:
    from SI_Toolkit_ApplicationSpecificFiles.user_defined_normalization_correction import apply_user_defined_normalization_correction
except:
    print('SI_Toolkit_ApplicationSpecificFiles not created yet')

import yaml, os

config = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config_training.yml'), 'r'), Loader=yaml.FullLoader)


PATH_TO_NORMALIZATION_INFO = config["paths"]["PATH_TO_EXPERIMENT_FOLDERS"] + config["paths"]["path_to_experiment"] + 'NormalizationInfo/'
normalization_rounding_decimals = 5


def get_paths_to_datafiles(paths_to_data_information):
    """
    There are three options to load data:
    1. provide path_to_normalization_information
        and the program will load the datafiles with which normalization was computed
    2. provide path_to_folder_with_data
        and program will load all the csv files it encounters at this location
        (nested folders are not searched though)
    3. provide list_of_paths_to_datafiles
        and datafiles will be loaded

    Options 1., 2. and 3. are exclusive. get_paths_to_datafiles distinguishes if it got a list (does nothing),
    path to a folder (list csv files in this folder), or path to a csv file (it assumes it is normalization file)
    and try to find list of paths in it.
    """

    list_of_paths_to_datafiles = []

    def list_of_paths_from_norminfo():

        with open(paths_to_data_information, 'r', newline='') as cmt_file:  # open file
            reached_path_list = False
            for line in cmt_file:  # read each line
                if reached_path_list:
                    if line == '#':  # Empty line means we reached end of path list
                        break
                    path = line[len('#     '):]  # remove first '#     '
                    list_of_paths_to_datafiles.append(path)

                # After this line paths are listed
                if line == '# Data files used to calculate normalization information:':
                    reached_path_list = True


    if isinstance(paths_to_data_information, list):
        for path in paths_to_data_information:
            if path[-4:] == '.csv':
                list_of_paths_to_datafiles.append(path)
            else:
                # Assume that path to folder was provided
                folder_content = glob.glob(path + '*.csv')
                list_of_paths_to_datafiles += folder_content

    elif isinstance(paths_to_data_information, str):
        if paths_to_data_information[-4:] == '.csv':
            # Get list of paths from normalization_information
            list_of_paths_from_norminfo(paths_to_data_information)
        else:
            # Assume that path to folder was provided
            for dirpath, dirnames, filenames in os.walk(paths_to_data_information):
                for filename in [f for f in filenames if f.endswith(".csv")]:
                    list_of_paths_to_datafiles.append(os.path.join(dirpath, filename))

            # list_of_paths_to_datafiles = glob.glob(paths_to_data_information + '*.csv')
    else:
        raise TypeError('Unsupported type of input argument to get_paths_to_datafiles')

    return sorted(list_of_paths_to_datafiles)


def get_full_paths_to_csvs(default_locations='', csv_names=None):
    """
    This super cool function takes as the argument
    default locations where the csv files are normally find
    and
    the names of csv files.
    It returns the list of absolute paths to the csv files.

    It is probably mostly useful for loading single csv file at a variable location.

    Both the default_locations and csv_names are graciously allowed to be a string
    if there is only one default location respectively one csv file.
    But they can always be list of strings with one or more elements.

    csv_names can be either the name of a file (with or without ".csv" suffix - isn't it delightful?)
    or the absolute or relative path to it.
    csv_names is None, '' or [] the path to the most recent files over default locations will be returned

    If there are two csv with name = csv_names at two places listed as default_location,
    and only name, not the path is specified,
    the exception will be raised notifying user about the problem. I find it also fantastic.
    """

    file_paths = []

    # If not already a list pack default location into a list
    if not isinstance(default_locations, list):
        default_locations = [default_locations]

    # If empty, load the most recent file from ANY(!) of the default locations
    if csv_names is None or csv_names == '' or csv_names == []:
        if default_locations[0] != [] and (default_locations[0] is not None):
            # get the latest file from the default location
            try:
                list_of_files = []
                for default_location in default_locations:
                    list_of_files.extend(glob.glob(default_location + '/*.csv'))
                file_paths = [max(list_of_files, key=os.path.getctime)]
            except FileNotFoundError:
                print('Cannot load: No experiment recording found in data folders: {}'.format(default_locations))
        else:
            raise Exception('Cannot load: Tried loading most recent recording, but no default locations specified')

    else:
        # If not already a list csv_names location into a list
        if not isinstance(csv_names, list):
            csv_names = [csv_names]

        for filename in csv_names:

            if filename[-4:] != '.csv':
                filename += '.csv'

            # check if file found in DATA_FOLDER_NAME or at local starting point
            if os.path.isfile(filename):
                file_path = [filename]
            elif default_locations is None or default_locations == [] or default_locations == '':
                file_path = []
                print(
                    'Cannot load: There is no experiment recording file with name {} in root and no default location is specified'.format(
                        filename, default_locations))
            else:
                file_path = []
                one_file_already_found = False
                for default_location in default_locations:
                    file_path_trial = os.path.join(default_location, filename)
                    if os.path.isfile(file_path_trial):
                        if one_file_already_found:
                            raise Exception('There is more than one csv file with specified name at default location')
                        file_path.append(file_path_trial)
                        one_file_already_found = True
                if not file_path:
                    print(
                        'Cannot load: There is no experiment recording file with name {} at local folder or in {}'.format(
                            filename, default_locations))

            file_paths.extend(file_path)

    return file_paths


# load csv file with experiment recording (e.g. for replay)
def load_csv_recording(file_path):
    if isinstance(file_path, list):
        file_path = file_path[0]

    # Get race recording
    print('Loading file {}'.format(file_path))
    try:
        data: pd.DataFrame = pd.read_csv(file_path, comment='#')  # skip comment lines starting with #
    except Exception as e:
        print('Cannot load: Caught {} trying to read CSV file {}'.format(e, file_path))
        return False

    # Change to float32 wherever numeric column
    cols = data.columns
    data[cols] = data[cols].apply(pd.to_numeric, errors='ignore', downcast='float')

    return data

def load_data(list_of_paths_to_datafiles=None):

    all_dfs = []  # saved separately to get normalization
    print('Loading data files:')
    sleep(0.1)
    for file_number in trange(len(list_of_paths_to_datafiles)):
        filepath = list_of_paths_to_datafiles[file_number]
        # print(filepath)
        # Read column names from file
        df = pd.read_csv(filepath, comment='#')

        # Change to float32 wherever numeric column
        cols = df.columns
        df[cols] = df[cols].apply(pd.to_numeric, errors='ignore', downcast='float')

        all_dfs.append(df)

    return all_dfs


# This function returns the saving interval of datafile
# Used to ensure that datafiles used for training save data with the same frequency
def get_sampling_interval_from_datafile(path_to_datafile):
    preceding_text = '# Saving: '
    dt_save = None
    with open(path_to_datafile, 'r') as cmt_file:  # open file
        for line in cmt_file:  # read each line
            if line[0:len(preceding_text)] == preceding_text:
                dt_save = float(line[len(preceding_text):-2])
                return dt_save
    return dt_save


# This function returns the saving interval of datafile
# Used to ensure that datafiles used for training save data with the same frequency
# def get_sampling_interval_from_normalization_info(path_to_normalization_info):
#     preceding_text = '# Sampling interval of data used to calculate normalization: '
#     with open(path_to_normalization_info, 'r', newline='') as cmt_file:  # open file
#        for line in cmt_file:  # read each line
#             if line[0:len(preceding_text)] == preceding_text:
#                 dt_information = line[len(preceding_text):]
#                 if dt_information == 'Not constant!':
#                     print('The normalization information was calculated with data with varying sampling frequency.')
#                     dt_save = None
#                 else:
#                     dt_save = float(dt_information[:-2])
#                 return dt_save


def calculate_normalization_info(paths_to_data_information=None, plot_histograms=True, user_correction=True, path_to_norm_info=None):
    """
    This function creates csv file with information about dataset statistics which may be used for normalization.
    The statistics are calculated from provided datafiles
    BUT may include user corrections to account for prior knowledge about distribution (e.g. 0 mean)
    """

    list_of_paths_to_datafiles = get_paths_to_datafiles(paths_to_data_information)

    # print paths to datafiles
    print('')
    print('# Datafiles used to calculate normalization:')
    for i in range(len(list_of_paths_to_datafiles)):
        print('    - ' + list_of_paths_to_datafiles[i])

    # endregion

    # region Check if all datafile have the same sampling interval

    # dts_save = []
    # dt_save = None
    # for path in list_of_paths_to_datafiles:
    #     dt_save = get_sampling_interval_from_datafile(path)
    #     dts_save.append(dt_save)
    # dts_save = np.array(dt_save)

    # tol = 1.0e-6
    # sampling_interval_str = '# Sampling interval of data used to calculate normalization: '
    # try:
    #     if np.all(abs(dts_save - dt_save) < tol):
    #         sampling_interval_str += '{} s'.format(dt_save)
    #     else:
    #         sampling_interval_str += 'Not constant!'
    # except TypeError:
    #     print('Save interval unknown.')

    # endregion

    # region Load data
    df = load_data(list_of_paths_to_datafiles=list_of_paths_to_datafiles)
    # endregion

    # region Concatinate all data frames into one
    if type(df) is list:
        df_total = pd.concat(df)
    else:
        df_total = df

    del df
    # endregion

    # region Exclude time from normalization
    if 'time' in df_total.columns:
        df_total.drop('time',
                      axis='columns', inplace=True)

    # endregion

    # region Calculate normalization values from data
    df_mean = df_total.mean(axis=0)
    df_std = df_total.std(axis=0)
    df_max = df_total.max(axis=0)
    df_min = df_total.min(axis=0)
    frame = {'mean': df_mean, 'std': df_std, 'max': df_max, 'min': df_min}
    df_norm_info = pd.DataFrame(frame).transpose()

    # endregion

    # region User correction to calculated normalization values

    # This way user can impose prior knowledge of the distribution and
    # e.g. impose 0 mean even if data used for normalization does not show it.

    # Make a copy of the original normalization values (from calculations)
    # to see which changes was done by user
    df_norm_info_from_data = df_norm_info.copy()

    # User defined normalization values:

    if user_correction:
        df_norm_info = apply_user_defined_normalization_correction(df_norm_info)

    if df_norm_info.equals(df_norm_info_from_data):
        modified = 'No'
        # print to file also original dataframe, so that anybody can check changes done by user
    else:
        modified = 'Yes'

    df_norm_info = df_norm_info.round(normalization_rounding_decimals)

    # endregion

    # region Transform original dataframe to comment by adding "comment column" and "space columns"
    df_norm_info_from_data = df_norm_info_from_data.reindex(sorted(df_norm_info_from_data.columns), axis=1)
    df_norm_info_from_data = df_norm_info_from_data.round(normalization_rounding_decimals)
    df_index = df_norm_info_from_data.index
    df_norm_info_from_data.insert(0, "      ", df_index, True)
    df_norm_info_from_data.insert(0, "#", 4 * ['#'], True)
    for i in range(len(df_norm_info_from_data.columns)):
        df_norm_info_from_data.insert(2 * i + 1, '         ', 4 * ['           '], True)
    # endregion

    # region Make folder to save normalization info (if not yet existing)
    try:
        os.makedirs(PATH_TO_NORMALIZATION_INFO[:-1])
    except FileExistsError:
        pass

    # endregion

    # region Write the .csv file
    date_now = datetime.now().strftime('%Y-%m-%d')
    time_now = datetime.now().strftime('%H-%M-%S')
    if path_to_norm_info is None:
        csv_filepath = PATH_TO_NORMALIZATION_INFO + 'NI_' + date_now + '_' + time_now + '.csv'
    else:
        csv_filepath = path_to_norm_info + 'NI_' + date_now + '_' + time_now + '.csv'

    with open(csv_filepath, "a", newline='') as outfile:
        writer = csv.writer(outfile)

        writer.writerow(['# ' + 'This is normalization information calculated {} at time {}'
                        .format(date_now, time_now)])
        try:
            repo = Repo()
            git_revision = repo.head.object.hexsha
        except:
            git_revision = 'unknown'
        writer.writerow(['# ' + 'Done with git-revision: {}'
                        .format(git_revision)])

        # writer.writerow(['#'])

        # writer.writerow([sampling_interval_str])

        # writer.writerow(['#'])

        writer.writerow(['# Data files used to calculate normalization information:'])
        for path in list_of_paths_to_datafiles:
            writer.writerow(['#     {}'.format(path)])

        writer.writerow(['#'])

        writer.writerow(['# Original (calculated from data) Normalization Information:'])

    df_norm_info_from_data.to_csv(csv_filepath, index=False, header=True, mode='a')  # Mode (a)ppend

    with open(csv_filepath, "a", newline='') as outfile:

        writer = csv.writer(outfile)

        writer.writerow(['#'])

        writer.writerow(['# Does user modified normalization info calculated from data?: {}'.format(modified)])

        writer.writerow(['#'])

        writer.writerow(['# Normalization Information:'])

    df_norm_info = df_norm_info.reindex(sorted(df_norm_info.columns), axis=1)
    df_norm_info.to_csv(csv_filepath, index=True, header=True, mode='a')  # Mode (a)ppend

    # endregion

    # region Plot histograms of data used for normalization
    if plot_histograms:
        # Plot historgrams to make the firs check about gaussian assumption
        for feature in df_total.columns:
            plt.hist(df_total[feature].to_numpy(), 50, density=True, facecolor='g', alpha=0.75)
            plt.title(feature)
            plt.show()

    # endregion

    return df_norm_info, csv_filepath


def load_normalization_info(path_to_normalization_info):
    return pd.read_csv(path_to_normalization_info, index_col=0, comment='#')


def normalize_feature(feature, normalization_info, normalization_type='minmax_sym', name=None):
    """feature needs to have atribute name!!!"""

    if hasattr(feature, 'name'):
        name = feature.name
    else:
        pass

    if name in normalization_info.columns:
        pass
    else:
        return feature

    if normalization_type == 'gaussian':
        col_mean = normalization_info.loc['mean', name]
        col_std = normalization_info.loc['std', name]
        if col_std == 0:
            return 0
        else:
            return (feature - col_mean) / col_std
    elif normalization_type == 'minmax_pos':
        col_min = normalization_info.loc['min', name]
        col_max = normalization_info.loc['max', name]
        if (col_max - col_min) == 0:
            return 0
        else:
            return (feature - col_min) / (col_max - col_min)
    elif normalization_type == 'minmax_sym':
        col_min = normalization_info.loc['min', name]
        col_max = normalization_info.loc['max', name]
        if (col_max - col_min) == 0:
            return 0
        else:
            return -1.0 + 2.0 * (feature - col_min) / (col_max - col_min)


def normalize_df(dfs, normalization_info, normalization_type='minmax_sym'):
    if type(dfs) is list:
        for i in range(len(dfs)):
            dfs[i] = dfs[i].apply(normalize_feature, axis=0,
                                  normalization_info=normalization_info,
                                  normalization_type=normalization_type)
    else:
        dfs = dfs.apply(normalize_feature, axis=0,
                        normalization_info=normalization_info,
                        normalization_type=normalization_type)

    return dfs


def denormalize_feature(feature, normalization_info, normalization_type='minmax_sym', name=None):
    """feature needs to have atribute name!!!"""

    if hasattr(feature, 'name'):
        name = feature.name
    else:
        pass

    if normalization_type == 'gaussian':
        # col_mean = normalization_info.loc['mean', name]
        # col_std = normalization_info.loc['std', name]
        # return feature * col_std + col_mean
        return feature * normalization_info.loc['std', name] + normalization_info.loc['mean', name]
    elif normalization_type == 'minmax_pos':
        # col_min = normalization_info.loc['min', name]
        # col_max = normalization_info.loc['max', name]
        # return feature * (col_max - col_min) + col_min
        # return feature * col_std + col_mean
        return feature * (normalization_info.loc['max', name] - normalization_info.loc['min', name]) + \
               normalization_info.loc['min', name]
    elif normalization_type == 'minmax_sym':
        # col_min = normalization_info.loc['min', name]
        # col_max = normalization_info.loc['max', name]
        # return ((feature + 1.0) / 2.0) * (col_max - col_min) + col_min
        return ((feature + 1.0) / 2.0) * (normalization_info.loc['max', name] - normalization_info.loc['min', name]) \
               + normalization_info.loc['min', name]


def denormalize_df(dfs, normalization_info, normalization_type='minmax_sym'):
    if type(dfs) is list:
        for i in range(len(dfs)):
            dfs[i] = dfs[i].apply(denormalize_feature, axis=0,
                                  normalization_info=normalization_info,
                                  normalization_type=normalization_type)
    else:
        dfs = dfs.apply(denormalize_feature, axis=0,
                        normalization_info=normalization_info,
                        normalization_type=normalization_type)

    return dfs


def denormalize_numpy_array(normalized_array,
                            features,
                            normalization_info,
                            normalization_type='minmax_sym'):
    denormalized_array = np.zeros_like(normalized_array)
    for feature_idx in range(len(features)):
        if normalization_type == 'gaussian':
            denormalized_array[..., feature_idx] = normalized_array[..., feature_idx] * \
                                                   normalization_info.at['std', features[feature_idx]] + \
                                                   normalization_info.at['mean', features[feature_idx]]
        elif normalization_type == 'minmax_pos':
            denormalized_array[..., feature_idx] = normalized_array[..., feature_idx] \
                                                   * (normalization_info.at['max', features[feature_idx]] -
                                                      normalization_info.at['min', features[feature_idx]]) + \
                                                   normalization_info.at['min', features[feature_idx]]
        elif normalization_type == 'minmax_sym':
            denormalized_array[..., feature_idx] = ((normalized_array[..., feature_idx] + 1.0) / 2.0) * \
                                                   (normalization_info.at['max', features[feature_idx]] -
                                                    normalization_info.at['min', features[feature_idx]]) \
                                                   + normalization_info.at['min', features[feature_idx]]

    return denormalized_array


def normalize_numpy_array(denormalized_array,
                          features,
                          normalization_info,
                          normalization_type='minmax_sym',
                          normalized_array = None,):  # If you want to write to an existing array instead of creating your own

    if normalized_array is None: # This option gives a possibility of using preallocated array
        normalized_array = np.zeros_like(denormalized_array)

    for feature_idx in range(len(features)):
        if normalization_type == 'gaussian':
            normalized_array[..., feature_idx] = (denormalized_array[..., feature_idx]
                                                  - normalization_info.at['mean', features[feature_idx]]) / \
                                                  normalization_info.at['std', features[feature_idx]]

        elif normalization_type == 'minmax_pos':
            normalized_array[..., feature_idx] = (denormalized_array[..., feature_idx]
                                                  - normalization_info.at['min', features[feature_idx]])\
                                                  / (normalization_info.at['max', features[feature_idx]] -
                                                  normalization_info.at['min', features[feature_idx]])


        elif normalization_type == 'minmax_sym':
            normalized_array[..., feature_idx] = -1.0 + 2.0 * (
                    (denormalized_array[..., feature_idx]-normalization_info.at['min', features[feature_idx]])
                    /
                    (normalization_info.at['max', features[feature_idx]] - normalization_info.at['min', features[feature_idx]])
            )

    return normalized_array


if __name__ == '__main__':
    folder_with_data_to_calculate_norm_info = config["paths"]["PATH_TO_EXPERIMENT_FOLDERS"] + config["paths"]["path_to_experiment"] + "Recordings/Train/"

    calculate_normalization_info(folder_with_data_to_calculate_norm_info)
