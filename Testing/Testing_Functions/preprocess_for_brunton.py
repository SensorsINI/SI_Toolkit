from SI_Toolkit.load_and_normalize import \
    load_data, get_sampling_interval_from_datafile, get_full_paths_to_csvs

def preprocess_for_brunton(a):
    # Get dataset:
    path_to_testfile = get_full_paths_to_csvs(default_locations=a.default_locations_for_testfile, csv_names=a.test_file)
    test_dfs = load_data(path_to_testfile)
    if not test_dfs:  # Tests for empty list
        raise FileNotFoundError('The load_data(path_to_testfile) function returned an empty list of files. \n'
                                'Probably the requested experiment recording is not found at the indicated location or under indicated name. \n'
                                'path_to_testfile value is {}, \n'
                                'a.default_locations_for_testfile is {}, \n'
                                'a.test_file is {}'.format(path_to_testfile, a.default_locations_for_testfile, a.test_file))
    if a.test_len == 'max':
        a.test_len = len(test_dfs[0]) - a.test_max_horizon - a.test_start_idx  # You could have +1; then, for last prediction you don not have ground truth to compare with, but you can still calculate it.
    dataset = test_dfs[0].iloc[a.test_start_idx:a.test_start_idx + a.test_len + a.test_max_horizon, :]
    dataset.reset_index(drop=True, inplace=True)

    # Get sampling interval
    dataset_sampling_dt = get_sampling_interval_from_datafile(path_to_testfile[0])
    if dataset_sampling_dt is None:
        print('No information about sampling interval found')
        # raise ValueError ('No information about sampling interval found')

    time_axis = dataset['time'].to_numpy()[:a.test_len]
    ground_truth_features = a.features+a.control_inputs
    ground_truth = dataset[ground_truth_features].to_numpy()[:a.test_len, :]

    return dataset, time_axis, dataset_sampling_dt, ground_truth
