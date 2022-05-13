from SI_Toolkit.TF.Parameters import args
from SI_Toolkit.TF.TF_Functions.Dataset import Dataset
from SI_Toolkit.load_and_normalize import load_data, normalize_df, get_paths_to_datafiles
import pandas as pd

"""Gets the training, validation and test files that are specified in config_training file to import for training 
in Google Colaboratory."""


def get_training_files():

    a = args()
    normalization_info = pd.read_csv(a.path_to_normalization_info, index_col=0, comment='#')

    paths_to_datafiles_training = get_paths_to_datafiles(a.training_files)
    training_dfs = load_data(paths_to_datafiles_training)
    training_dfs_norm = normalize_df(training_dfs, normalization_info)
    training_dataset = Dataset(training_dfs_norm, a, shuffle=True, inputs=a.inputs, outputs=a.outputs)

    paths_to_datafiles_validation = get_paths_to_datafiles(a.validation_files)
    validation_dfs = load_data(paths_to_datafiles_validation)
    validation_dfs_norm = normalize_df(validation_dfs, normalization_info)
    validation_dataset = Dataset(validation_dfs_norm, a, shuffle=True, inputs=a.inputs, outputs=a.outputs)

    paths_to_datafiles_test = get_paths_to_datafiles(a.test_files)
    test_dfs = load_data(paths_to_datafiles_test)
    test_dfs_norm = normalize_df(test_dfs, normalization_info)
    test_set = Dataset(test_dfs_norm, a, shuffle=False, inputs=a.inputs, outputs=a.outputs)

    return training_dataset, validation_dataset, test_set
