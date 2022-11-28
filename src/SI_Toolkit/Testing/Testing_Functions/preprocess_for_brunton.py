from SI_Toolkit.load_and_normalize import \
    load_data, get_sampling_interval_from_datafile, get_full_paths_to_csvs
import numpy as np

def preprocess_for_brunton(
        test_file: str,
        path_to_testfile: str,
        test_len: int,
        test_max_horizon: int,
        test_start_idx: int,
        decimation: int,
        **kwargs,
):

    # Get dataset:
    path_to_testfile = get_full_paths_to_csvs(default_locations=path_to_testfile, csv_names=test_file)
    test_dfs = load_data(path_to_testfile)
    if not test_dfs:  # Tests for empty list
        raise FileNotFoundError('The load_data(path_to_testfile) function returned an empty list of files. \n'
                                'Probably the requested experiment recording is not found at the indicated location or under indicated name. \n'
                                'path_to_testfile value is {}, \n'
                                'path_to_testfile is {}, \n'
                                'a.test_file is {}'.format(path_to_testfile, path_to_testfile, test_file))
    if test_len == 'max':
        test_len = len(test_dfs[0]) - test_max_horizon - test_start_idx  # You could have +1; then, for last prediction you don not have ground truth to compare with, but you can still calculate it.
    dataset = test_dfs[0]
    dataset = dataset.iloc[::decimation, :]
    dataset = dataset.iloc[test_start_idx:test_start_idx + test_len + test_max_horizon, :]
    dataset.reset_index(drop=True, inplace=True)

    if dataset.shape[0]-test_max_horizon < test_len:
        raise ValueError(
            '\nThe test datafile is too small for the requested test length.\n'
            'For this datafile TEST_LEN can be {} at most.\n'
            'You requested {}.'.format(dataset.shape[0]-test_max_horizon, test_len))

    # Get sampling interval
    dataset_sampling_dt = get_sampling_interval_from_datafile(path_to_testfile[0])
    if dataset_sampling_dt is None:
        print('No information about sampling interval found')
        # raise ValueError ('No information about sampling interval found')

    time_axis = dataset['time'].to_numpy()[:test_len]
    ground_truth_features = np.array(dataset.columns)
    ground_truth = [dataset[ground_truth_features].to_numpy()[:test_len, :], ground_truth_features]

    return dataset, time_axis, dataset_sampling_dt, ground_truth
