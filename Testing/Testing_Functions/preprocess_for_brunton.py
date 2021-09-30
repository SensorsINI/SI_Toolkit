from SI_Toolkit.load_and_normalize import \
    load_data, get_sampling_interval_from_datafile, get_full_paths_to_csvs

def preprocess_for_brunton(a):
    # Get dataset:
    path_to_testfile = get_full_paths_to_csvs(default_locations=a.default_locations_for_testfile, csv_names=a.test_file)
    test_dfs = load_data(path_to_testfile)
    if a.test_len == 'max':
        a.test_len = len(test_dfs[
                             0]) - a.test_max_horizon - a.test_start_idx  # You could have +1; then, for last prediction you don not have ground truth to compare with, but you can still calculate it.
    dataset = test_dfs[0].iloc[a.test_start_idx:a.test_start_idx + a.test_len + a.test_max_horizon, :]
    dataset.reset_index(drop=True, inplace=True)

    # Get sampling interval
    dataset_sampling_dt = get_sampling_interval_from_datafile(path_to_testfile[0])
    if dataset_sampling_dt is None:
        raise ValueError ('No information about sampling interval found')

    # # Add noise to position, angle
    # if a.noise == 'add_noise':
    #     add_noise(dataset, a.noise_level)



    time_axis = dataset['time'].to_numpy()[:a.test_len]
    ground_truth_features = a.features+a.control_inputs
    ground_truth = dataset[ground_truth_features].to_numpy()[:a.test_len, :]

    return dataset, time_axis, dataset_sampling_dt, ground_truth
